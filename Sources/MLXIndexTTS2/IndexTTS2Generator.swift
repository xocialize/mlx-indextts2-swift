// IndexTTS2Generator.swift — the production generation driver (generate_v2.py port), tying the
// parity-locked components into one reusable pipeline: tokenize → reference conditioning →
// GPT AR → S2Mel CFM → BigVGAN → 22.05 kHz waveform.
//
// Engine-free by design (no MLXToolKit import): the MLXIndexTTS2TTS wrapper owns the engine
// contract, PCM decode/resample, and metaData routing; this class owns the kernels.
//
// Dtype policy: components load AS-SHIPPED (fp16 main checkpoint, fp32 w2v-BERT/codec/campplus)
// — exactly the dtype the Python reference ran on Metal to produce the Stage-0 goldens. The
// parity gates (P2–P7) ran fp32-upcast lanes; production quality is quantified in the app
// harness (dBFS + |STFT| + listen). Watchdog rules: weight loads on the CPU stream with
// `eval(model)` materialized post-update; every forward runs on the caller's (GPU) stream;
// int8/int4 quantize on CPU at load, forwards GPU-only (quant matmul is Metal-only).
//
// Duration control (E12): the length-regulator target length is the native lever —
// default ylens = code_len · 1.72 (generate_v2); `speechRate` divides it; `targetDuration`
// pins the total mel-frame budget (22 050 / 256 ≈ 86.13 frames/s), distributed across
// segments proportional to their code length.

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

/// The assembled IndexTTS2 pipeline. Construct with `load(...)` (heavy — pages all weights),
/// then `prepareReference` once per voice and `synthesize` per utterance.
public final class IndexTTS2Generator {

    public static let outputSampleRate = 22_050
    /// Front-end conditioning rate (w2v-BERT / CampPlus).
    public static let conditioningSampleRate = 16_000
    /// Mel frames per second at the S2Mel hop (22050 / 256).
    static let melFramesPerSecond = 22_050.0 / 256.0
    /// generate_v2's default length-regulator expansion factor.
    static let defaultLengthFactor = 1.72

    let gpt: UnifiedVoiceV2
    let s2mel: S2Mel
    let bigvgan: BigVGANV2
    let vq2emb: Vq2Emb
    let w2v: Wav2Vec2BertModel
    let repcodec: RepCodec
    let campplus: CAMPPlus
    public let tokenizer: IndexTTSTextTokenizer
    let semanticMean: MLXArray
    let semanticStd: MLXArray

    /// GPT-backbone quant applied at load (nil = as-shipped fp16).
    public let quantBits: Int?

    // MARK: - Loading

    private init(gpt: UnifiedVoiceV2, s2mel: S2Mel, bigvgan: BigVGANV2, vq2emb: Vq2Emb,
                 w2v: Wav2Vec2BertModel, repcodec: RepCodec, campplus: CAMPPlus,
                 tokenizer: IndexTTSTextTokenizer, semanticMean: MLXArray,
                 semanticStd: MLXArray, quantBits: Int?) {
        self.gpt = gpt
        self.s2mel = s2mel
        self.bigvgan = bigvgan
        self.vq2emb = vq2emb
        self.w2v = w2v
        self.repcodec = repcodec
        self.campplus = campplus
        self.tokenizer = tokenizer
        self.semanticMean = semanticMean
        self.semanticStd = semanticStd
        self.quantBits = quantBits
    }

    static func bakedNPY(_ name: String) throws -> MLXArray {
        guard let url = Bundle.module.url(forResource: name, withExtension: "npy",
                                          subdirectory: "Resources") else {
            throw IndexTTS2Error.missingResource("\(name).npy")
        }
        return try NPY.load(url)
    }

    /// Load one component with the full weight-key contract (0-missing / 0-unused).
    private static func loadComponent<M: Module>(
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
                "\(url.lastPathComponent): missing \(missing.count) "
                + "(\(missing.sorted().prefix(4))) unused \(unused.count) "
                + "(\(unused.sorted().prefix(4)))")
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)
        return model
    }

    /// Page in the full pipeline. Weight loads run on the CPU stream (watchdog rule);
    /// `quantBits` (8|4) quantizes the `gpt.h.*` Linears in place (donor scope, group 64).
    /// `progress` receives a coarse [0, 1] fraction across the 7 components.
    public static func load(
        modelDirectory: URL, w2vBertDirectory: URL, semanticCodecDirectory: URL,
        quantBits: Int? = nil, progress: ((Double) -> Void)? = nil
    ) throws -> IndexTTS2Generator {
        guard let vocabURL = Bundle.module.url(forResource: "tokenizer_vocab",
                                               withExtension: "json",
                                               subdirectory: "Resources") else {
            throw IndexTTS2Error.missingResource("tokenizer_vocab.json")
        }
        let tokenizer = try IndexTTSTextTokenizer(vocabURL: vocabURL)
        let semanticMean = try bakedNPY("semantic_mean").asType(.float32)
        let semanticStd = try bakedNPY("semantic_std").asType(.float32)
        guard let campplusURL = Bundle.module.url(forResource: "campplus_cn_common",
                                                  withExtension: "safetensors",
                                                  subdirectory: "Resources") else {
            throw IndexTTS2Error.missingResource("campplus_cn_common.safetensors")
        }

        return try Device.withDefaultDevice(Device(.cpu)) { () -> IndexTTS2Generator in
            var step = 0.0
            func tick() { step += 1; progress?(step / 7.0) }

            let gpt = try loadComponent(
                UnifiedVoiceV2(), url: modelDirectory.appending(path: "gpt.safetensors"),
                sanitize: UnifiedVoiceV2.sanitize)
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
                S2Mel(), url: modelDirectory.appending(path: "s2mel.safetensors"),
                sanitize: S2Mel.sanitize)
            tick()
            let bigvgan = try loadComponent(
                BigVGANV2(), url: modelDirectory.appending(path: "bigvgan.safetensors"),
                sanitize: { $0 })
            tick()
            let vq2emb = try loadComponent(
                Vq2Emb(), url: modelDirectory.appending(path: "vq2emb.safetensors"),
                sanitize: Vq2Emb.sanitize)
            tick()
            let w2v = try loadComponent(
                Wav2Vec2BertModel(),
                url: w2vBertDirectory.appending(path: "model.safetensors"),
                sanitize: Wav2Vec2BertModel.sanitize)
            tick()
            let repcodec = try loadComponent(
                RepCodec(),
                url: semanticCodecDirectory.appending(path: "semantic_codec/model.safetensors"),
                sanitize: RepCodec.sanitize)
            tick()
            let campplus = try loadComponent(
                CAMPPlus(), url: campplusURL, sanitize: CAMPPlus.sanitize)
            campplus.train(false)  // BatchNorms use running stats
            tick()

            return IndexTTS2Generator(
                gpt: gpt, s2mel: s2mel, bigvgan: bigvgan, vq2emb: vq2emb, w2v: w2v,
                repcodec: repcodec, campplus: campplus, tokenizer: tokenizer,
                semanticMean: semanticMean, semanticStd: semanticStd, quantBits: quantBits)
        }
    }

    // MARK: - Reference conditioning

    /// Everything `synthesize` needs from one reference voice — prepare once, reuse per line
    /// (the dub/long-form pattern).
    public struct Reference {
        public let speechCond: MLXArray        // (1, 32, 1280) perceiver speaker conditioning
        public let baseEmovec: MLXArray        // (1, 1280) reference-audio emotion vector
        public let style: MLXArray             // (1, 192) CampPlus embedding
        public let promptCondition: MLXArray   // (1, T_ref, 512) length-regulated S_ref
        public let refMel: MLXArray            // (1, 80, T_ref) CFM prompt
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
        let spkCondEmb = Wav2Vec2BertModel.semanticTap(hs, mean: semanticMean, std: semanticStd)
        eval(spkCondEmb)

        let (_, sRef) = repcodec.quantize(spkCondEmb)
        eval(sRef)

        guard let cmn = CampPlusFbank.fbankCMN(wav16k) else {
            throw IndexTTS2Error.audioTooShort
        }
        let style = campplus(cmn[.newAxis, 0..., 0...])
        eval(style)

        let refMel = RefMel.melSpectrogram(wav22k)
        let promptCondition = s2mel.lengthRegulatorModule(
            sRef, ylens: MLXArray([Int32(refMel.dim(2))]))
        eval(refMel, promptCondition)

        let spkNCL = spkCondEmb.transposed(0, 2, 1)
        let speechCond = gpt.getConditioning(spkNCL)
        let baseEmovec = gpt.getEmovec(spkNCL)
        eval(speechCond, baseEmovec)

        return Reference(speechCond: speechCond, baseEmovec: baseEmovec, style: style,
                         promptCondition: promptCondition, refMel: refMel)
    }

    // MARK: - Synthesis

    /// generate_v2 defaults.
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

    /// Synthesize one utterance. `emotionWeights` are the 8 category weights in
    /// `EmotionPresets.categories` order, ALREADY emo_alpha-scaled (nil = reference emotion
    /// as-is). `speechRate` scales pace (1.0 natural, >1 faster); `targetDurationSeconds`
    /// pins the total output length via the native length-regulator lever (wins over rate).
    /// `cancelCheck` is called between pipeline stages; a throw aborts.
    public func synthesize(
        text: String,
        reference: Reference,
        emotionWeights: [Float]? = nil,
        targetDurationSeconds: Double? = nil,
        speechRate: Double? = nil,
        params: SynthesisParams = SynthesisParams(),
        cancelCheck: (() throws -> Void)? = nil
    ) throws -> [Float] {
        // Emotion blend (generate_v2): preset weights ⇒ emovec_mat + (1−Σw)·base; else base.
        let emoVec: MLXArray
        if let weights = emotionWeights {
            emoVec = EmotionPresets.blend(weights: weights, style: reference.style,
                                          baseEmovec: reference.baseEmovec)
        } else {
            emoVec = reference.baseEmovec
        }
        let conditioning = gpt.prepareConditioningLatents(
            speechConditioning: reference.speechCond, emoVec: emoVec, batchSize: 1)
        eval(conditioning)

        // Tokenize + split long text. splitSegments partitions the piece sequence
        // order-preserving, so the paired ids are recovered by walking a cursor.
        let pairs = tokenizer.encodeWithPieces(text)
        let segments = tokenizer.splitSegments(
            pairs.map(\.surface), maxTokensPerSegment: params.maxTextTokensPerSegment)
        var cursor = 0
        let segmentIds: [[Int]] = segments.map { segment in
            let ids = pairs[cursor ..< cursor + segment.count].map(\.id)
            cursor += segment.count
            return ids
        }
        precondition(cursor == pairs.count, "splitSegments dropped tokens")

        // Per-segment AR → codes (needed up front for the targetDuration frame budget).
        var generated: [(ids: [Int], codes: [Int])] = []
        for ids in segmentIds where !ids.isEmpty {
            try cancelCheck?()
            let textTokens = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
            let result = gpt.generateMelCodes(
                conditioning: conditioning, textTokens: textTokens,
                maxMelTokens: params.maxMelTokens, temperature: params.temperature,
                topK: params.topK, topP: params.topP,
                repetitionPenalty: params.repetitionPenalty)
            let codes = compressSilence(result.melCodes)
            if !codes.isEmpty { generated.append((ids: ids, codes: codes)) }
        }
        // Cancellation must win over emptyGeneration: a per-token bail inside generateMelCodes
        // (CAN gate) can leave `generated` empty/partial — surface the CancellationError
        // unchanged here rather than laundering it into IndexTTS2Error.emptyGeneration.
        try cancelCheck?()
        guard !generated.isEmpty else { throw IndexTTS2Error.emptyGeneration }

        // Length-regulator targets (the E12 duration lever).
        let totalCodes = generated.reduce(0) { $0 + $1.codes.count }
        let silenceSamples = generated.count > 1 && params.intervalSilenceMs > 0
            ? Int(Double(Self.outputSampleRate) * Double(params.intervalSilenceMs) / 1000.0)
            : 0
        func targetFrames(for codeLen: Int) -> Int {
            if let duration = targetDurationSeconds {
                // Total speech frames = duration − inserted silence, split ∝ code length.
                let silenceSeconds = Double(silenceSamples * (generated.count - 1))
                    / Double(Self.outputSampleRate)
                let speechFrames = max(1.0, (duration - silenceSeconds) * Self.melFramesPerSecond)
                return max(4, Int(speechFrames * Double(codeLen) / Double(totalCodes)))
            }
            let rate = speechRate.map { max(0.25, min(4.0, $0)) } ?? 1.0
            return max(4, Int(Double(codeLen) * Self.defaultLengthFactor / rate))
        }

        // Per-segment S2Mel + vocoder.
        var audioSegments: [[Float]] = []
        for (ids, codes) in generated {
            try cancelCheck?()
            let textTokens = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
            let melCodes = MLXArray(codes.map(Int32.init)).reshaped(1, codes.count)

            let latent = gpt.forwardLatent(
                conditioning: conditioning, textTokens: textTokens, melCodes: melCodes)
            let gptlayerOut = s2mel.gptLayerModule(latent)
            let sInfer = vq2emb(melCodes).transposed(0, 2, 1) + gptlayerOut
            let cond = s2mel.lengthRegulatorModule(
                sInfer, ylens: MLXArray([Int32(targetFrames(for: codes.count))]))
            let catCondition = concatenated([reference.promptCondition, cond], axis: 1)
            eval(catCondition)

            try cancelCheck?()
            let mel = s2mel.cfmModule.inference(
                mu: catCondition, xLens: MLXArray([Int32(catCondition.dim(1))]),
                prompt: reference.refMel, style: reference.style,
                nTimesteps: params.diffusionSteps, temperature: 1.0,
                inferenceCfgRate: params.cfgRate)
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

        // Concatenate with interval silence (generate_v2's default multi-segment path).
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
