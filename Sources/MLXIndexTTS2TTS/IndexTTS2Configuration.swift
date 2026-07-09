import Foundation
import MLXToolKit

/// Init-time configuration for `IndexTTS2Package` (C9): where the IndexTTS2 weights live and
/// which quant tier the GPT backbone runs at. Per-request text/voice/emotion/duration ride the
/// canonical `TTSRequest`, not here.
///
/// Three weight sources back one loaded model (tier-3 pipeline):
/// - `repo` — the MLX-converted IndexTTS2 checkpoint (gpt / s2mel / bigvgan / vq2emb
///   safetensors). The small conversion artifacts that ship INSIDE that repo only as torch
///   pickles (feat1/feat2 emotion matrices, wav2vec2bert_stats, tokenizer.model) are baked
///   into the package resources instead (dumped by the oracle's tools/dump_stage2.py) — they
///   move into the weight repo at the own-conversion re-publish.
/// - `w2vBertRepo` — facebook/w2v-bert-2.0 front-end (ships fp32, 2.2 GB on disk; an fp16
///   halving rides the same re-publish).
/// - `semanticCodecRepo` — amphion/MaskGCT (semantic_codec/model.safetensors).
///
/// `quant` selects the GPT-backbone tier: `.fp16` (as-shipped), `.int8` / `.int4` quantize the
/// `gpt.h.*` Linears in-memory at load (donor scope, group 64) — the weight sources are
/// IDENTICAL across tiers, so quant never changes the materialization set.
public struct IndexTTS2Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The MLX-converted IndexTTS2 checkpoint repo.
    public var repo: String
    /// Pinned revision; nil = main.
    public var revision: String?
    /// w2v-BERT 2.0 front-end repo.
    public var w2vBertRepo: String
    /// MaskGCT semantic-codec repo.
    public var semanticCodecRepo: String
    /// GPT-backbone quant tier: fp16 (as-shipped) | int8 (near-lossless, cos 0.99998) |
    /// int4 (cos 0.9952). Applied in-memory at load; S2Mel/BigVGAN/conditioners stay fp16.
    public var quant: Quant
    /// Explicit checkpoint directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Explicit w2v-BERT directory.
    public var w2vBertDirectory: URL?
    /// Explicit MaskGCT directory (expects `semantic_codec/model.safetensors` inside).
    public var semanticCodecDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// Engine-stamped load headroom (`BudgetAware`): when tight, `load()` drops the GPT
    /// backbone fp16 → int8 (near-lossless) instead of failing admission.
    public var availableBudgetBytes: UInt64?

    public init(
        repo: String = "mlx-community/IndexTTS-2-fp16",
        revision: String? = nil,
        w2vBertRepo: String = "mlx-community/IndexTTS-2-fp16",
        semanticCodecRepo: String = "mlx-community/IndexTTS-2-fp16",
        quant: Quant = .fp16,
        modelDirectory: URL? = nil,
        w2vBertDirectory: URL? = nil,
        semanticCodecDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.w2vBertRepo = w2vBertRepo
        self.semanticCodecRepo = semanticCodecRepo
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.w2vBertDirectory = w2vBertDirectory
        self.semanticCodecDirectory = semanticCodecDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    // Environment-specific URLs + the engine-stamped budget are excluded from Codable.
    private enum CodingKeys: String, CodingKey {
        case repo, revision, w2vBertRepo, semanticCodecRepo, quant
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        w2vBertRepo = try c.decodeIfPresent(String.self, forKey: .w2vBertRepo)
            ?? "facebook/w2v-bert-2.0"
        semanticCodecRepo = try c.decodeIfPresent(String.self, forKey: .semanticCodecRepo)
            ?? "amphion/MaskGCT"
        quant = try c.decode(Quant.self, forKey: .quant)
    }
}

extension IndexTTS2Configuration: BudgetAware {}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension IndexTTS2Configuration: WeightSourcing {
    /// The checkpoint safetensors the Swift port loads (config/tokenizer/feat pickles are baked).
    static let mainFiles = [
        "gpt.safetensors", "s2mel.safetensors", "bigvgan.safetensors", "vq2emb.safetensors",
    ]
    static let w2vBertFile = "model.safetensors"
    static let semanticCodecFile = "semantic_codec/model.safetensors"

    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "main", repo: repo, revision: revision,
                         matching: Self.mainFiles),
            WeightSource(role: "w2v-bert", repo: w2vBertRepo,
                         matching: [Self.w2vBertFile]),
            WeightSource(role: "semantic-codec", repo: semanticCodecRepo,
                         matching: [Self.semanticCodecFile]),
        ]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func storeHas(_ repo: String, files: [String]) -> Bool {
            guard let dir = ModelStore(root: storeRoot).directory(for: repo) else { return false }
            return files.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
        }
        return weightSources.filter { source in
            switch source.role {
            case "main":
                if let dir = modelDirectory,
                   fm.fileExists(atPath: dir.appending(path: Self.mainFiles[0]).path) { return false }
                return !storeHas(source.repo, files: Self.mainFiles)
            case "w2v-bert":
                if let dir = w2vBertDirectory,
                   fm.fileExists(atPath: dir.appending(path: Self.w2vBertFile).path) { return false }
                return !storeHas(source.repo, files: [Self.w2vBertFile])
            default:  // semantic-codec
                if let dir = semanticCodecDirectory,
                   fm.fileExists(atPath: dir.appending(path: Self.semanticCodecFile).path) { return false }
                return !storeHas(source.repo, files: [Self.semanticCodecFile])
            }
        }
    }

    /// The configuration with nil directories resolved to the store layout — what `load()`
    /// uses AFTER materialization. Explicit directories always win.
    public func resolved(storeRoot: URL?) -> IndexTTS2Configuration {
        let store = ModelStore(root: storeRoot)
        var cfg = self
        if cfg.modelDirectory == nil { cfg.modelDirectory = store.directory(for: repo) }
        if cfg.w2vBertDirectory == nil { cfg.w2vBertDirectory = store.directory(for: w2vBertRepo) }
        if cfg.semanticCodecDirectory == nil {
            cfg.semanticCodecDirectory = store.directory(for: semanticCodecRepo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension IndexTTS2Configuration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved view so auto-materialize (nil-dir) configs prewarm the downloaded
        // layout on later cold launches; missing paths are skipped (best-effort prewarmer).
        let r = resolved(storeRoot: modelsRootDirectory)
        var paths: [URL] = []
        if let dir = r.modelDirectory {
            paths += Self.mainFiles.map { dir.appending(path: $0) }
        }
        if let dir = r.w2vBertDirectory { paths.append(dir.appending(path: Self.w2vBertFile)) }
        if let dir = r.semanticCodecDirectory {
            paths.append(dir.appending(path: Self.semanticCodecFile))
        }
        return paths
    }
}
