// IndexTTS2Generator.swift — the production generation driver (IndexTTS-2.5 `infer_v2_5.py`
// port), tying the parity-locked components into one reusable pipeline:
// text frontend → reference conditioning → GPT AR → EnhancedCodec decode → length regulator
// → S2Mel CFM → BigVGAN → 22.05 kHz waveform.
//
// Engine-free by design (no MLXToolKit import): the MLXIndexTTS2TTS wrapper owns the engine
// contract, PCM decode/resample, and metaData routing; this class owns the kernels.
//
// Dtype policy: components load AS-SHIPPED (fp16 main checkpoint incl. the codec, fp32
// w2v-BERT / CAMPPlus) — the dtype the Python reference runs on Metal. Parity gates run fp32
// CPU lanes. Watchdog rules: weight loads on the CPU stream with `eval(model)` materialized
// post-update; every forward runs on the caller's (GPU) stream; int8/int4 quantize on CPU at
// load, forwards GPU-only (quant matmul is Metal-only).
//
// Duration control (E12): the length-regulator target is the native lever — reference
// `int(len(S_infer) · 1.72 · duration_factor)`; `speechRate` is 1/duration_factor;
// `targetDuration` pins the total mel-frame budget (22 050 / 256 ≈ 86.13 frames/s),
// distributed across segments proportional to their content length.

import Foundation
import MLX
import MLXNN
import MLXRandom

public enum IndexTTS2Error: Error, CustomStringConvertible {
    case missingWeights(String)
    case weightContract(String)
    case missingResource(String)
    case emptyGeneration
    case audioTooShort

    public var description: String {
        switch self {
        case .missingWeights(let path): return "missing weights: \(path)"
        case .weightContract(let detail): return "weight key contract violated: \(detail)"
        case .missingResource(let name): return "missing baked resource: \(name)"
        case .emptyGeneration: return "no mel codes generated"
        case .audioTooShort: return "reference audio shorter than one analysis frame"
        }
    }
}

/// The assembled IndexTTS-2.5 pipeline. Construct with `load(...)` (heavy — pages all
/// weights), then `prepareReference` once per voice and `synthesize` per utterance.
public final class IndexTTS2Generator {

    public static let outputSampleRate = 22_050
    /// Front-end conditioning rate (w2v-BERT / CAMPPlus).
    public static let conditioningSampleRate = 16_000
    /// Mel frames per second at the S2Mel hop (22050 / 256).
    static let melFramesPerSecond = 22_050.0 / 256.0
    /// The reference's length-regulator expansion factor (codec 50 Hz → mel 86.13 Hz).
    static let defaultLengthFactor = 1.72
    /// The checkpoint's tiktoken vocabulary file name.
    public static let tokenizerFile = "multilingual_zh_ja_yue_char_del.tiktoken"

    // Components are public for the parity-gate lane (`indextts2-gate`, a separate target).
    public let gpt: UnifiedVoiceV25
    public let s2mel: S2Mel
    public let bigvgan: BigVGANV2
    public let codec: EnhancedCodecDecoder
    public let w2v: Wav2Vec2BertModel
    public let campplus: CAMPPlus
    public let frontend: IndexTTSTextFrontend
    public let semanticMean: MLXArray
    public let semanticStd: MLXArray

    /// GPT-backbone quant applied at load (nil = as-shipped fp16).
    public let quantBits: Int?

    /// Test-facing seam for the engine's **INF gate** (C14): every loaded component, keyed by
    /// role. `campplus` is the BatchNorm carrier; the rest are in scope so a future
    /// training-mode-sensitive layer cannot slip in unwatched.
    public var inferenceModeGraphs: [String: MLXNN.Module?] {
        ["gpt": gpt, "s2mel": s2mel, "bigvgan": bigvgan, "codec": codec, "w2v": w2v, "campplus": campplus]
    }

    // MARK: - Loading

    private init(gpt: UnifiedVoiceV25, s2mel: S2Mel, bigvgan: BigVGANV2, codec: EnhancedCodecDecoder,
                 w2v: Wav2Vec2BertModel, campplus: CAMPPlus, frontend: IndexTTSTextFrontend,
                 semanticMean: MLXArray, semanticStd: MLXArray, quantBits: Int?) {
        self.gpt = gpt
        self.s2mel = s2mel
        self.bigvgan = bigvgan
        self.codec = codec
        self.w2v = w2v
        self.campplus = campplus
        self.frontend = frontend
        self.semanticMean = semanticMean
        self.semanticStd = semanticStd
        self.quantBits = quantBits
    }

    /// Parity-gate lane: upcast every component to `dtype` (fp32 CPU goldens) in place.
    public func upcast(to dtype: DType) {
        for module in [gpt, s2mel, bigvgan, codec, w2v, campplus] as [Module] {
            module.update(parameters: module.parameters().mapValues { $0.asType(dtype) })
            eval(module)
        }
    }

    static func bakedNPY(_ name: String) throws -> MLXArray {
        guard let url = Bundle.module.url(forResource: name, withExtension: "npy", subdirectory: "Resources") else {
            throw IndexTTS2Error.missingResource("\(name).npy")
        }
        return try NPY.load(url)
    }

    /// Load one component with the full weight-key contract (0-missing / 0-unused).
    ///
    /// Internal rather than private so the C14 INF gate can exercise this choke point directly —
    /// it is where inference mode is set for all six components.
    static func loadComponent<M: Module>(
        _ model: M, url: URL, sanitize: ([String: MLXArray]) -> [String: MLXArray]
    ) throws -> M {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw IndexTTS2Error.missingWeights(url.path)
        }
        let declared = Set(model.parameters().flattened().map(\.0))
        let sanitized = sanitize(try loadArrays(url: url))
        let missing = declared.subtracting(sanitized.keys)
        let unused = Set(sanitized.keys).subtracting(declared)
        guard missing.isEmpty, unused.isEmpty else {
            throw IndexTTS2Error.weightContract(
                "\(url.lastPathComponent): missing \(missing.count) (\(missing.sorted().prefix(4))) "
                + "unused \(unused.count) (\(unused.sorted().prefix(4)))")
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)

        // INFERENCE MODE — load-bearing, not hygiene (engine C14 / the INF gate). `MLXNN.Module`
        // defaults to training mode, where `BatchNorm` normalizes by the CURRENT batch and
        // overwrites the checkpoint's running stats on every forward. CAMPPlus — the speaker
        // encoder whose embedding conditions the whole clone — is dense with BatchNorms, so
        // without this the speaker embedding drifts run to run. Every component loads through
        // here, so this is the one choke point — call sites deliberately do NOT repeat it.
        model.train(false)
        eval(model)
        return model
    }

    /// Page in the full pipeline. Weight loads run on the CPU stream (watchdog rule);
    /// `quantBits` (8|4) quantizes the `gpt.h.*` Linears in place (donor scope, group 64).
    /// `progress` receives a coarse [0, 1] fraction across the 6 components.
    public static func load(
        modelDirectory: URL, w2vBertDirectory: URL,
        quantBits: Int? = nil, progress: ((Double) -> Void)? = nil
    ) throws -> IndexTTS2Generator {
        let vocabURL = modelDirectory.appending(path: tokenizerFile)
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw IndexTTS2Error.missingWeights(vocabURL.path)
        }
        let frontend = IndexTTSTextFrontend(tokenizer: try TiktokenBPE(vocabularyURL: vocabURL))
        let semanticMean = try bakedNPY("semantic_mean").asType(.float32)
        let semanticStd = try bakedNPY("semantic_std").asType(.float32)
        guard let campplusURL = Bundle.module.url(forResource: "campplus_cn_common", withExtension: "safetensors",
                                                  subdirectory: "Resources") else {
            throw IndexTTS2Error.missingResource("campplus_cn_common.safetensors")
        }

        return try Device.withDefaultDevice(Device(.cpu)) { () -> IndexTTS2Generator in
            var step = 0.0
            func tick() { step += 1; progress?(step / 6.0) }

            let gpt = try loadComponent(
                UnifiedVoiceV25(), url: modelDirectory.appending(path: "gpt.safetensors"),
                sanitize: UnifiedVoiceV25.sanitize)
            if let bits = quantBits {
                // Donor scope: ONLY the GPT2 backbone Linears; embeddings / heads / norms /
                // conditioners stay full precision. Quantize on CPU; forwards must be GPU.
                quantize(model: gpt, groupSize: 64, bits: bits) { path, module in
                    path.hasPrefix("gpt.h.") && module is Linear
                }
                eval(gpt)
            }
            tick()
            let s2mel = try loadComponent(
                S2Mel(), url: modelDirectory.appending(path: "s2mel.safetensors"), sanitize: S2Mel.sanitize)
            tick()
            let bigvgan = try loadComponent(
                BigVGANV2(), url: modelDirectory.appending(path: "bigvgan.safetensors"), sanitize: { $0 })
            tick()
            let codec = try loadComponent(
                EnhancedCodecDecoder(), url: modelDirectory.appending(path: "codec.safetensors"),
                sanitize: EnhancedCodecDecoder.sanitize)
            tick()
            let w2v = try loadComponent(
                Wav2Vec2BertModel(), url: w2vBertDirectory.appending(path: "model.safetensors"),
                sanitize: Wav2Vec2BertModel.sanitize)
            tick()
            let campplus = try loadComponent(CAMPPlus(), url: campplusURL, sanitize: CAMPPlus.sanitize)
            tick()

            return IndexTTS2Generator(
                gpt: gpt, s2mel: s2mel, bigvgan: bigvgan, codec: codec, w2v: w2v, campplus: campplus,
                frontend: frontend, semanticMean: semanticMean, semanticStd: semanticStd, quantBits: quantBits)
        }
    }

    // MARK: - Reference conditioning

    /// Everything `synthesize` needs from one reference voice — prepare once, reuse per line.
    public struct Reference {
        public let style: MLXArray             // (1, 192) CAMPPlus embedding → spk_emb_proj + CFM style
        public let baseEmovec: MLXArray        // (1, 1280) reference-audio emotion vector
        public let promptCondition: MLXArray   // (1, T_ref_mel, 512) length-regulated w2v-BERT features
        public let refMel: MLXArray            // (1, 80, T_ref_mel) CFM prompt
    }

    /// Build the reference conditioning from mono PCM at the two pipeline rates
    /// (resampling stays the caller's job — the wrapper owns PCM decode/resample).
    public func prepareReference(samples16k: [Float], samples22k: [Float]) throws -> Reference {
        let wav16k = MLXArray(samples16k)
        let wav22k = MLXArray(samples22k)

        guard let (features, mask) = SeamlessFeatureExtractor.callAsFeatures(wav16k) else {
            throw IndexTTS2Error.audioTooShort
        }
        let (_, hs) = w2v(inputFeatures: features, attentionMask: mask)
        let spkCondEmb = Wav2Vec2BertModel.semanticTap(hs, mean: semanticMean, std: semanticStd)  // (1, T, 1024)
        eval(spkCondEmb)

        guard let cmn = CampPlusFbank.fbankCMN(wav16k) else { throw IndexTTS2Error.audioTooShort }
        let style = campplus(cmn.expandedDimensions(axis: 0))
        eval(style)

        let refMel = RefMel.melSpectrogram(wav22k)
        // 2.5 conditions the CFM prompt on the RAW semantic features (no codec round trip).
        let promptCondition = s2mel.lengthRegulatorModule(spkCondEmb, ylens: MLXArray([Int32(refMel.dim(2))]))
        eval(refMel, promptCondition)

        let baseEmovec = gpt.getEmovec(spkCondEmb.transposed(0, 2, 1))
        eval(baseEmovec)

        return Reference(style: style, baseEmovec: baseEmovec, promptCondition: promptCondition, refMel: refMel)
    }

    // MARK: - Synthesis

    /// Reference defaults (`infer_v2_5.py`).
    public struct SynthesisParams {
        public var maxMelTokens = 1500
        public var maxTextTokensPerSegment = 120
        public var temperature: Float = 0.8
        public var topK = 30
        public var topP: Float = 0.8
        public var repetitionPenalty: Float = 10.0
        public var diffusionSteps = 25
        public var cfgRate: Float = 0.7
        public var intervalSilenceMs = 200
        public init() {}
    }

    /// Synthesize one utterance. `language` nil = script detection (Latin → English; pass
    /// `.es` explicitly for Spanish). `emotionWeights` are the 8 category weights in
    /// `EmotionPresets.categories` order, ALREADY emo_alpha-scaled (nil = reference emotion
    /// as-is). `speechRate` scales pace (1.0 natural, >1 faster); `targetDurationSeconds`
    /// pins the total output length via the native length-regulator lever (wins over rate).
    /// `cancelCheck` is called between pipeline stages; a throw aborts.
    public func synthesize(
        text: String,
        reference: Reference,
        language: IndexTTSLanguage? = nil,
        emotionWeights: [Float]? = nil,
        targetDurationSeconds: Double? = nil,
        speechRate: Double? = nil,
        params: SynthesisParams = SynthesisParams(),
        cancelCheck: (() throws -> Void)? = nil
    ) throws -> [Float] {
        // Emotion blend: preset weights ⇒ emovec_mat + (1−Σw)·base; else base.
        let emoVec: MLXArray
        if let weights = emotionWeights {
            emoVec = EmotionPresets.blend(weights: weights, style: reference.style, baseEmovec: reference.baseEmovec)
        } else {
            emoVec = reference.baseEmovec
        }
        let conditioning = gpt.prepareConditioningLatents(style: reference.style, emoVec: emoVec)
        eval(conditioning)

        let prepared = try frontend.prepare(text, language: language, maxTokensPerSegment: params.maxTextTokensPerSegment)
        let languageID = prepared.language.languageID

        // Per-segment AR → codes → codec features (needed up front for the targetDuration budget).
        var generated: [MLXArray] = []
        for ids in prepared.tokenIDs where !ids.isEmpty {
            try cancelCheck?()
            let result = gpt.generateMelCodes(
                conditioning: conditioning, textTokens: ids, languageID: languageID,
                maxMelTokens: params.maxMelTokens, temperature: params.temperature,
                topK: params.topK, topP: params.topP, repetitionPenalty: params.repetitionPenalty)
            let codes = compressSilence(result.melCodes)
            guard !codes.isEmpty else { continue }
            let sInfer = codec(MLXArray(codes.map(Int32.init)).expandedDimensions(axis: 0))   // (1, 2T, 1024)
            eval(sInfer)
            generated.append(sInfer)
        }
        // Cancellation must win over emptyGeneration: a per-token bail inside generateMelCodes
        // (CAN gate) can leave `generated` empty/partial — surface the CancellationError.
        try cancelCheck?()
        guard !generated.isEmpty else { throw IndexTTS2Error.emptyGeneration }

        // Length-regulator targets (the E12 duration lever).
        let totalContent = generated.reduce(0) { $0 + $1.dim(1) }
        let silenceSamples = generated.count > 1 && params.intervalSilenceMs > 0
            ? Int(Double(Self.outputSampleRate) * Double(params.intervalSilenceMs) / 1000.0) : 0
        func targetFrames(for contentLen: Int) -> Int {
            if let duration = targetDurationSeconds {
                let silenceSeconds = Double(silenceSamples * (generated.count - 1)) / Double(Self.outputSampleRate)
                let speechFrames = max(1.0, (duration - silenceSeconds) * Self.melFramesPerSecond)
                return max(4, Int(speechFrames * Double(contentLen) / Double(totalContent)))
            }
            let rate = speechRate.map { max(0.25, min(4.0, $0)) } ?? 1.0
            return max(4, Int(Double(contentLen) * Self.defaultLengthFactor / rate))
        }

        // Per-segment S2Mel + vocoder.
        var audioSegments: [[Float]] = []
        for sInfer in generated {
            try cancelCheck?()
            let cond = s2mel.lengthRegulatorModule(sInfer, ylens: MLXArray([Int32(targetFrames(for: sInfer.dim(1)))]))
            let catCondition = concatenated([reference.promptCondition, cond], axis: 1)
            eval(catCondition)

            try cancelCheck?()
            let mel = s2mel.cfmModule.inference(
                mu: catCondition, xLens: MLXArray([Int32(catCondition.dim(1))]),
                prompt: reference.refMel, style: reference.style,
                nTimesteps: params.diffusionSteps, temperature: 1.0, inferenceCfgRate: params.cfgRate)
            eval(mel)

            try cancelCheck?()
            let wav = bigvgan(mel[0..., 0..., reference.refMel.dim(2)...])
            var audio = wav[0, 0]
            let peak = MLX.abs(audio).max().item(Float.self)
            if peak > 1.0 { audio = audio / max(peak, 1e-6) }
            audio = clip(audio, min: -0.99, max: 0.99)
            eval(audio)
            audioSegments.append(audio.asArray(Float.self))

            // Bound the denoise/vocode working set across segments (cache-discipline rule).
            Memory.clearCache()
        }

        if audioSegments.count == 1 { return audioSegments[0] }
        var out: [Float] = []
        for (index, segment) in audioSegments.enumerated() {
            out += segment
            if index < audioSegments.count - 1 && silenceSamples > 0 {
                out += [Float](repeating: 0, count: silenceSamples)
            }
        }
        return out
    }
}
