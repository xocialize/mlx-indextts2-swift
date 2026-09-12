import Foundation
import MLX
import MLXNN
import MLXIndexTTS2
import MLXRandom
import MLXToolKit

/// IndexTTS-2.5 on the canonical `tts` surface: zero-shot voice cloning from reference audio
/// in five languages (zh / en / ja / es / ar) with the two E12 control levers no other fleet
/// TTS has natively — **emotion decoupled from speaker identity** (8-category preset plane)
/// and **explicit duration control** (the length-regulator target length). Returns the
/// canonical `Audio` (.wav, 22.05 kHz mono).
///
/// Engine-owned lifecycle (C13): the engine constructs from an `IndexTTS2Configuration`,
/// materializes the two declared sources under its models root, pages weights in with
/// `load()`, drives `run(_:)`, and reclaims with `unload()`.
///
/// Voice: `.referenceAudio` only (a zero-shot cloner — no preset voices; `.auto`/`.named`
/// reject legibly via `unsupportedRequestFeature`). `referenceTranscript` is not consumed.
///
/// `metaData` keys (package-specific, C5 — the E12 param plane; canonical `TTSControls`
/// promotion is AB-A-0049):
/// - `language` (string: zh | en | ja | es | ar, or a common name): the text language. Omit to
///   detect from script — Latin script defaults to English, so pass `es` for Spanish.
/// - `emotion` (string | array): preset name ("happy"), weighted list ("happy:0.8,calm:0.2"),
///   or an 8-number array in `EmotionPresets.categories` order
///   (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm).
/// - `emoAlpha` (double, default 0.6): emotion intensity — scales the preset weights;
///   the remainder (1 − Σw) stays on the reference audio's own emotion.
/// - `targetDuration` (double, seconds): native duration fit — pins the output length via
///   the length regulator (the dub cue-window lever). Wins over `speechRate`.
/// - `speechRate` (double, default 1.0): pace lever (>1 faster, <1 slower) — the reference's
///   `duration_factor` is 1 / speechRate.
/// - `seed` (int): reproducible sampling. Clamped to 32 bits — large 64-bit seeds produce
///   Gumbel-noise patterns that never favor EOS → runaway generation (Qwen3 precedent).
@InferenceActor
public final class IndexTTS2Package: ModelPackage {
    public typealias Configuration = IndexTTS2Configuration

    /// Split footprints — MEASURED (M5 Max, 2026-09-02; `indextts2-gate footprint [--bits 8|4]`,
    /// MLX-active memory): post-load floor fp16 4457 MB / int8 4017 MB / int4 3781 MB; run peak on
    /// a 15 s utterance 9250 / 8990 / 8743 MB ⇒ transient 4.8–5.0 GB on every tier (CFM + BigVGAN
    /// dominated, so quant tiers move residents, not the peak). Declared with headroom.
    nonisolated static let fp16ResidentBytes: UInt64 = 4_600_000_000
    nonisolated static let int8ResidentBytes: UInt64 = 4_200_000_000
    nonisolated static let int4ResidentBytes: UInt64 = 3_900_000_000
    nonisolated static let peakActivationBytes: UInt64 = 5_100_000_000

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the bilibili Model Use License Agreement (IndexTTS-2.5) — commercially
            // permissive; separate license only above 100 M MAU / RMB 1 B revenue (engine
            // `permissiveAllowlist`, LTX-2 / LFM precedent). C8: this port is Apache-2.0
            // (donor vanch007/mlx-indextts2, MIT).
            license: LicenseDeclaration(weightLicense: .bilibiliModelUse, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "IndexTeam/IndexTTS-2.5",
                revision: "d0aa86e75bb6f3437f3831e95056fa72842d89ef", tier: 3),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes: fp16ResidentBytes,
                                   peakActivationBytes: peakActivationBytes),
                    QuantFootprint(quant: .int8, residentBytes: int8ResidentBytes,
                                   peakActivationBytes: peakActivationBytes),
                    QuantFootprint(quant: .int4, residentBytes: int4ResidentBytes,
                                   peakActivationBytes: peakActivationBytes),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [
                SpecialtyWeight(.voiceClone, strength: 1.0),
                SpecialtyWeight(.emotionControl, strength: 1.0),
                SpecialtyWeight(.durationControl, strength: 1.0),
            ],
            surfaces: [
                TTSContract.descriptor(
                    name: "indextts2",
                    summary: "IndexTTS-2.5 zero-shot voice-cloning TTS (.wav, 22.05 kHz; zh/en/ja/es/ar "
                        + "via metaData.language) with native per-request emotion control (8-category "
                        + "preset plane, decoupled from the cloned speaker) and native duration control "
                        + "(targetDuration / speechRate) — the fit-to-cue lever for dubbing. "
                        + "Requires voice.referenceAudio (no preset voices). E12 typed plane: "
                        + "`emotion` (.categorical over the shared 8-name vocabulary, or an 8-number "
                        + ".vector) and `targetDuration`; metaData strings remain the compat path, and "
                        + "`emoAlpha` / `speechRate` stay on metaData.",
                    modes: [.expressive, .neutral],
                    // E12 declaration (contract 1.38.0, AB-A-0049 part 3). Only what the PORT
                    // implements: the preset/vector head. `.referenceAudio` (upstream
                    // emo_audio_prompt) and `.textDescription` (upstream use_emo_text) are not
                    // ported, so they are not declared and the engine refuses them pre-admission.
                    controls: TTSControls(emotionModes: [.categorical, .vector],
                                          supportsTargetDuration: true)
                )
            ]
        )
    }

    private let configuration: Configuration
    private var generator: IndexTTS2Generator?

    /// Test-facing seam for the engine's **INF gate** (C14): the loaded component graphs, keyed by
    /// role. `nil` before `load()`, so an unloaded package reports an empty graph — which INF-1
    /// fails, by design. `campplus` is the BatchNorm carrier.
    var inferenceModeGraphs: [String: MLXNN.Module?] { generator?.inferenceModeGraphs ?? [:] }
    // Reference-conditioning reuse: long-form/dub synthesis sends the SAME reference for every
    // line; preparing it re-runs w2v-BERT + CAMPPlus + ref-mel + length regulator. Memoize
    // keyed by the reference bytes (InferenceActor serializes run(); Reference is read-only).
    private var cachedReference: (key: Int, reference: IndexTTS2Generator.Reference)?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard generator == nil else { return }
        // The ENGINE materializes dir-less configs from `weightSources` before load() (contract
        // 1.24+); this guard is the offline backstop, never a download.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw IndexTTS2Error.missingWeights(
                "sources not materialized: \(missing.map(\.role).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()

        let resolved = configuration.resolved(storeRoot: storeRoot)
        guard let modelDir = resolved.modelDirectory, let w2vDir = resolved.w2vBertDirectory else {
            throw IndexTTS2Error.missingWeights("unresolved weight directories (no store root)")
        }

        // Quant tier: configured, with the BudgetAware near-lossless drop — a tight stamped
        // budget downgrades fp16 → int8 instead of failing to fit.
        var quant = configuration.quant
        if quant == .fp16, let budget = configuration.availableBudgetBytes,
           budget < Self.fp16ResidentBytes + Self.peakActivationBytes {
            quant = .int8
        }
        let quantBits: Int?
        switch quant {
        case .fp16, .bf16, .fp32: quantBits = nil   // as-shipped fp16 weights
        case .int8: quantBits = 8
        case .int4: quantBits = 4
        default:
            throw PackageError.unsupportedRequestFeature(
                "quant \(quant.rawValue) — IndexTTS2 supports fp16 | int8 | int4")
        }

        // Heavy: pages ~4.4 GB across 6 components (CPU-stream loads inside).
        generator = try IndexTTS2Generator.load(modelDirectory: modelDir, w2vBertDirectory: w2vDir, quantBits: quantBits)
    }

    public func unload() async {
        generator = nil
        cachedReference = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation.
        // Mid-run cadence: the GPT AR loop bails per generated mel token, and the throwing
        // `cancelCheck` closure checkpoints between every pipeline stage, rethrowing
        // CancellationError unchanged.
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // Voice: zero-shot cloning only.
        guard case .referenceAudio(let referenceClip) = tts.voice.selection else {
            throw PackageError.unsupportedRequestFeature(
                "voice — IndexTTS2 has no preset voices; provide voice.referenceAudio")
        }

        // Reference conditioning (memoized per reference clip).
        let key = Self.referenceKey(referenceClip.data)
        let reference: IndexTTS2Generator.Reference
        if let cached = cachedReference, cached.key == key {
            reference = cached.reference
        } else {
            RunProgress.report(.encode)
            let (mono, sourceRate) = try AudioSupport.decodeToMono(referenceClip)
            let samples16k = SincResampler.resample(
                audio: mono, from: sourceRate, to: IndexTTS2Generator.conditioningSampleRate)
            let samples22k = SincResampler.resample(
                audio: mono, from: sourceRate, to: IndexTTS2Generator.outputSampleRate)
            reference = try generator.prepareReference(samples16k: samples16k, samples22k: samples22k)
            cachedReference = (key, reference)
        }
        try Task.checkCancellation()

        // E12 metaData plane + language.
        var language: IndexTTSLanguage? = nil
        if case .string(let code)? = tts.metaData["language"] {
            guard let parsed = IndexTTSLanguage(parsing: code) else {
                throw PackageError.unsupportedRequestFeature(
                    "language '\(code)' — IndexTTS-2.5 speaks zh | en | ja | es | ar")
            }
            language = parsed
        }
        // E12 typed plane (contract 1.38.0): `tts.emotion` / `tts.targetDuration` win; the
        // metaData strings are the compatibility path. `emoAlpha` (intensity pre-scale) and
        // `speechRate` are not in the typed plane and stay on metaData.
        let emotionWeights = try Self.resolveEmotionWeights(
            typed: tts.emotion, meta: tts.metaData["emotion"],
            alpha: tts.metaData.doubleValue("emoAlpha") ?? 0.6)
        let targetDuration = tts.targetDuration ?? tts.metaData.doubleValue("targetDuration")
        let speechRate = tts.metaData.doubleValue("speechRate")

        if let seed = tts.metaData.intValue("seed") {
            MLXRandom.seed(UInt64(bitPattern: Int64(seed)) & 0xFFFF_FFFF)
        }

        RunProgress.report(.generate)
        let samples: [Float]
        do {
            samples = try generator.synthesize(
                text: tts.text, reference: reference, language: language,
                emotionWeights: emotionWeights, targetDurationSeconds: targetDuration,
                speechRate: speechRate, cancelCheck: { try Task.checkCancellation() })
        } catch let error as IndexTTSTextFrontend.FrontendError {
            throw PackageError.unsupportedRequestFeature(error.description)
        }

        try Task.checkCancellation()
        RunProgress.report(.decode)
        let wav = AudioSupport.encodeWAV16(samples: samples, sampleRate: IndexTTS2Generator.outputSampleRate)
        return TTSResponse(audio: Audio(format: .wav, data: wav, sampleRate: IndexTTS2Generator.outputSampleRate, channels: 1))
    }

    // MARK: - E12 typed plane (contract 1.38.0) → preset weights

    /// The typed `TTSRequest.emotion` wins; `metaData["emotion"]` is the compatibility path.
    /// `.categorical` resolves through the shared E12 vocabulary (canonical names + the 9→8
    /// aliases) onto ONE preset at weight 1 × alpha; `.vector` is the 8-number head in
    /// `EmotionPresets.categories` order, clamped and alpha-scaled exactly like the metaData array
    /// form. `.referenceAudio` / `.textDescription` are NOT declared — the port has no
    /// emo-audio-prompt / emo-text path yet — so the engine refuses them before admission, and a
    /// direct caller is refused here, legibly, never silently.
    nonisolated static func resolveEmotionWeights(typed: TTSEmotion?, meta: MetaValue?,
                                                  alpha: Double) throws -> [Float]? {
        guard let typed else { return try parseEmotion(meta, alpha: alpha) }
        let scale = Float(max(0.0, min(1.0, alpha)))
        var weights = [Float](repeating: 0, count: EmotionPresets.categories.count)
        switch typed {
        case .categorical(let label):
            guard let emotion = E12Emotion.resolve(label) else {
                throw PackageError.unsupportedRequestFeature(
                    "emotion '\(label)' — known: \(E12Emotion.knownLabels)")
            }
            weights[emotion.presetIndex] = 1.0
        case .vector(let values):
            guard values.count == weights.count else {
                throw PackageError.unsupportedRequestFeature(
                    "emotion vector — want \(weights.count) weights in order "
                        + "(\(EmotionPresets.categories.joined(separator: ", "))), got \(values.count)")
            }
            for (index, v) in values.enumerated() { weights[index] = max(0, min(1.2, v)) }
        case .referenceAudio:
            throw PackageError.unsupportedRequestFeature(
                "emotion.referenceAudio — not in this port (upstream emo_audio_prompt); "
                    + "send .categorical or .vector")
        case .textDescription:
            throw PackageError.unsupportedRequestFeature(
                "emotion.textDescription — not in this port (upstream use_emo_text); "
                    + "send .categorical or .vector")
        @unknown default:
            throw PackageError.unsupportedRequestFeature("emotion — mode not known to this package")
        }
        return weights.map { $0 * scale }
    }

    // MARK: - E12 emotion parsing (`parse_emotion` + emo_alpha pre-scale)

    /// `emotion` accepts a preset name ("happy"), a weighted list ("happy:0.8,calm:0.2"),
    /// or an 8-number array in `EmotionPresets.categories` order. Returns the alpha-scaled
    /// 8-vector, or nil (no emotion override → reference emotion as-is).
    nonisolated static func parseEmotion(_ value: MetaValue?, alpha: Double) throws -> [Float]? {
        guard let value else { return nil }
        let scale = Float(max(0.0, min(1.0, alpha)))
        var weights = [Float](repeating: 0, count: EmotionPresets.categories.count)

        switch value {
        case .string(let spec):
            for part in spec.split(separator: ",") {
                let pair = part.split(separator: ":", maxSplits: 1)
                let name = pair[0].trimmingCharacters(in: .whitespaces).lowercased()
                guard let index = E12Emotion.resolve(name)?.presetIndex else {
                    throw PackageError.unsupportedRequestFeature(
                        "emotion '\(name)' — known: \(E12Emotion.knownLabels)")
                }
                let weight = pair.count == 2 ? Float(pair[1].trimmingCharacters(in: .whitespaces)) ?? 1.0 : 1.0
                weights[index] = max(0, min(1.2, weight))
            }
        case .array(let values):
            guard values.count == weights.count else {
                throw PackageError.unsupportedRequestFeature(
                    "emotion array — want \(weights.count) weights (\(EmotionPresets.categories.joined(separator: ", ")))")
            }
            for (index, v) in values.enumerated() {
                switch v {
                case .double(let d): weights[index] = max(0, min(1.2, Float(d)))
                case .int(let i): weights[index] = max(0, min(1.2, Float(i)))
                default: throw PackageError.unsupportedRequestFeature("emotion array — numeric weights only")
                }
            }
        default:
            throw PackageError.unsupportedRequestFeature("emotion — string or 8-number array")
        }
        return weights.map { $0 * scale }
    }

    /// In-memory cache key for a prepared reference (Hasher is per-process seeded, which is
    /// all we need — reuse happens within one long-form run).
    nonisolated static func referenceKey(_ data: Data) -> Int {
        var hasher = Hasher()
        hasher.combine(data)
        return hasher.finalize()
    }
}

extension IndexTTS2Package {
    public nonisolated static var registration: PackageRegistration { .of(IndexTTS2Package.self) }
}

extension MetaData {
    /// Convenience: read an int-valued metaData key (e.g. the sampling seed).
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }

    /// Convenience: read a double-valued key, accepting ints (JSON 1 vs 1.0).
    func doubleValue(_ key: String) -> Double? {
        switch self[key] {
        case .double(let value)?: return value
        case .int(let value)?: return Double(value)
        default: return nil
        }
    }
}

extension E12Emotion {
    /// Position in `EmotionPresets.categories` (the weight-vector order). The two lists are the
    /// same eight names in the same order — `ManifestTests` pins that, so this cannot drift.
    var presetIndex: Int { EmotionPresets.categories.firstIndex(of: rawValue)! }
}
