import Foundation
import MLX
import MLXIndexTTS2
import MLXRandom
import MLXToolKit

/// IndexTTS2 on the canonical `tts` surface: zero-shot voice cloning from reference audio
/// with the two E12 control levers no other fleet TTS has natively — **emotion decoupled
/// from speaker identity** (8-category preset plane) and **explicit duration control**
/// (the length-regulator target length). Returns the canonical `Audio` (.wav, 22.05 kHz mono).
///
/// Engine-owned lifecycle (C13): the engine constructs from an `IndexTTS2Configuration`,
/// pages weights in with `load()` (auto-materializing the three declared sources under the
/// engine's models root when set), drives `run(_:)`, and reclaims with `unload()`.
///
/// Voice: `.referenceAudio` only (IndexTTS2 is a zero-shot cloner — it has no preset voices;
/// `.auto`/`.named` reject legibly via `unsupportedRequestFeature`). `referenceTranscript`
/// is not consumed (conditioning is transcript-free).
///
/// `metaData` keys (package-specific, C5 — the E12 param plane's first realization; a
/// canonical `TTSControls` proposal is filed in mlxengine-audio/AGENT_BRIDGE.md):
/// - `emotion` (string | array): preset name ("happy"), weighted list ("happy:0.8,calm:0.2"),
///   or an 8-number array in `EmotionPresets.categories` order
///   (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm).
/// - `emoAlpha` (double, default 0.6): emotion intensity — scales the preset weights;
///   the remainder (1 − Σw) stays on the reference audio's own emotion.
/// - `targetDuration` (double, seconds): native duration fit — pins the output length via
///   the length regulator (the dub cue-window lever). Wins over `speechRate`.
/// - `speechRate` (double, default 1.0): pace lever (>1 faster, <1 slower), same mechanism.
/// - `seed` (int): reproducible sampling. Clamped to 32 bits — large 64-bit seeds produce
///   Gumbel-noise patterns that never favor EOS → runaway generation (Qwen3 precedent).
@InferenceActor
public final class IndexTTS2Package: ModelPackage {
    public typealias Configuration = IndexTTS2Configuration

    /// Split footprints, in-app phys_footprint baseline (MLXEngineAudio INDEXTTS2_VALIDATE
    /// run, 2026-07-09): post-load floor 4.76 GB (fp16 chain: GPT 1.65 + S2Mel 0.20 +
    /// BigVGAN 0.21 + w2v-BERT 2.2 fp32 + codec 0.17 + CampPlus 0.03 GB + process overhead);
    /// run-phase peak 9.61 GB (~4.6 s utterance, emotion path) ⇒ transient ≈ 4.85 GB —
    /// CFM/BigVGAN-dominated, so quant tiers move residents, NOT the peak. int8/int4 deltas
    /// from the P7 quant lane (GPT backbone Linears only). Post-evict phys returns to 0.37 GB.
    nonisolated static let fp16ResidentBytes: UInt64 = 5_000_000_000
    nonisolated static let int8ResidentBytes: UInt64 = 4_350_000_000
    nonisolated static let int4ResidentBytes: UInt64 = 4_100_000_000
    nonisolated static let peakActivationBytes: UInt64 = 5_000_000_000

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: INDEX_MODEL_LICENSE — NonCommercial (commercial use needs bilibili written
            // authorization); admitted only under `.permissiveOrAcknowledged` (Anima pattern).
            // C8: the port code (this repo; donor solar2ain/mlx-indextts MIT) is Apache-2.0.
            license: LicenseDeclaration(weightLicense: .indexTTS2Model, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "vanch007/mlx-indextts2-standard-fp16",
                revision: "31118db400202a438e4d42bbce8e426298072d50", tier: 3),
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
                SpecialtyWeight(.emotionControl, strength: 1.0),
                SpecialtyWeight(.durationControl, strength: 1.0),
            ],
            surfaces: [
                TTSContract.descriptor(
                    name: "indextts2",
                    summary: "IndexTTS2 zero-shot voice-cloning TTS (.wav, 22.05 kHz) with "
                        + "native per-request emotion control (8-category preset plane, "
                        + "decoupled from the cloned speaker) and native duration control "
                        + "(targetDuration / speechRate) — the fit-to-cue lever for dubbing. "
                        + "Requires voice.referenceAudio (no preset voices).",
                    modes: [.expressive, .neutral]
                )
            ]
        )
    }

    private let configuration: Configuration
    private var generator: IndexTTS2Generator?
    // Reference-conditioning reuse (the Qwen3 E1 pattern): long-form/dub synthesis sends the
    // SAME reference for every line; preparing it re-runs w2v-BERT + RepCodec + CampPlus +
    // ref-mel + length regulator. Memoize keyed by the reference bytes. Safe to hold:
    // InferenceActor serializes run(), and Reference is read-only once built.
    private var cachedReference: (key: Int, reference: IndexTTS2Generator.Reference)?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard generator == nil else { return }

        // Auto-materialize missing sources into the engine store (dir-less configs only;
        // explicit directories never touch the network).
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        if !missing.isEmpty {
            guard let storeRoot else {
                throw IndexTTS2Error.missingWeights(
                    "no models root set and sources missing: \(missing.map(\.role).joined(separator: ", "))")
            }
            try await WeightMaterializer.materialize(missing, into: storeRoot)
        }
        try Task.checkCancellation()

        let resolved = configuration.resolved(storeRoot: storeRoot)
        guard let modelDir = resolved.modelDirectory,
              let w2vDir = resolved.w2vBertDirectory,
              let codecDir = resolved.semanticCodecDirectory else {
            throw IndexTTS2Error.missingWeights("unresolved weight directories (no store root)")
        }

        // Quant tier: configured, with the BudgetAware near-lossless drop — a tight stamped
        // budget downgrades fp16 → int8 (gpt_latent cos 0.99998) instead of failing to fit.
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

        // Heavy: pages ~4.5 GB across 7 components (CPU-stream loads inside).
        generator = try IndexTTS2Generator.load(
            modelDirectory: modelDir, w2vBertDirectory: w2vDir,
            semanticCodecDirectory: codecDir, quantBits: quantBits)
    }

    public func unload() async {
        generator = nil
        cachedReference = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

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
            let (mono, sourceRate) = try AudioSupport.decodeToMono(referenceClip)
            let samples16k = SincResampler.resample(
                audio: mono, from: sourceRate, to: IndexTTS2Generator.conditioningSampleRate)
            let samples22k = SincResampler.resample(
                audio: mono, from: sourceRate, to: IndexTTS2Generator.outputSampleRate)
            reference = try generator.prepareReference(
                samples16k: samples16k, samples22k: samples22k)
            cachedReference = (key, reference)
        }
        try Task.checkCancellation()

        // E12 metaData plane.
        let emotionWeights = try Self.parseEmotion(
            tts.metaData["emotion"],
            alpha: tts.metaData.doubleValue("emoAlpha") ?? 0.6)
        let targetDuration = tts.metaData.doubleValue("targetDuration")
        let speechRate = tts.metaData.doubleValue("speechRate")

        if let seed = tts.metaData.intValue("seed") {
            // Clamp to 32 bits — large 64-bit seeds yield Gumbel-noise patterns that never
            // favor EOS → runaway generation (verified in prior dub work; Qwen3 pattern).
            MLXRandom.seed(UInt64(bitPattern: Int64(seed)) & 0xFFFF_FFFF)
        }

        let samples = try generator.synthesize(
            text: tts.text,
            reference: reference,
            emotionWeights: emotionWeights,
            targetDurationSeconds: targetDuration,
            speechRate: speechRate,
            cancelCheck: { try Task.checkCancellation() })

        try Task.checkCancellation()
        let wav = AudioSupport.encodeWAV16(
            samples: samples, sampleRate: IndexTTS2Generator.outputSampleRate)
        return TTSResponse(audio: Audio(
            format: .wav, data: wav,
            sampleRate: IndexTTS2Generator.outputSampleRate, channels: 1))
    }

    // MARK: - E12 emotion parsing (generate_v2 `parse_emotion` + emo_alpha pre-scale)

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
                guard let index = EmotionPresets.categories.firstIndex(of: name) else {
                    throw PackageError.unsupportedRequestFeature(
                        "emotion '\(name)' — known: \(EmotionPresets.categories.joined(separator: ", "))")
                }
                let weight = pair.count == 2
                    ? Float(pair[1].trimmingCharacters(in: .whitespaces)) ?? 1.0
                    : 1.0
                weights[index] = max(0, min(1.2, weight))
            }
        case .array(let values):
            guard values.count == weights.count else {
                throw PackageError.unsupportedRequestFeature(
                    "emotion array — want \(weights.count) weights "
                    + "(\(EmotionPresets.categories.joined(separator: ", ")))")
            }
            for (index, v) in values.enumerated() {
                switch v {
                case .double(let d): weights[index] = max(0, min(1.2, Float(d)))
                case .int(let i): weights[index] = max(0, min(1.2, Float(i)))
                default:
                    throw PackageError.unsupportedRequestFeature("emotion array — numeric weights only")
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
