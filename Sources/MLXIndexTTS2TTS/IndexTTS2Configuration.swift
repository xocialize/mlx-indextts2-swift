import Foundation
import MLXToolKit

/// Init-time configuration for `IndexTTS2Package` (C9): where the IndexTTS-2.5 weights live
/// and which quant tier the GPT backbone runs at. Per-request text/voice/language/emotion/
/// duration ride the canonical `TTSRequest`, not here.
///
/// Two weight sources back one loaded model, both in the fleet-controlled
/// `mlx-community/IndexTTS-2.5-fp16` repo by default:
/// - `repo` / role `main` — the MLX-converted 2.5 checkpoint (gpt / codec / s2mel / bigvgan
///   safetensors + the tiktoken vocabulary). The small emotion matrices (feat1/feat2) and the
///   w2v-BERT feature statistics are baked into the package resources (identical bytes to the
///   2.0 release's, verified at the 2.5 update).
/// - `w2vBertRepo` / role `w2v-bert` — facebook/w2v-bert-2.0 (`model.safetensors`, fp32),
///   re-hosted in the same repo.
///
/// `quant` selects the GPT-backbone tier: `.fp16` (as-shipped), `.int8` / `.int4` quantize the
/// `gpt.h.*` Linears in-memory at load (donor scope, group 64) — the weight sources are
/// IDENTICAL across tiers, so quant never changes the materialization set.
public struct IndexTTS2Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The MLX-converted IndexTTS-2.5 checkpoint repo.
    public var repo: String
    /// Pinned revision; nil = main.
    public var revision: String?
    /// w2v-BERT 2.0 front-end repo.
    public var w2vBertRepo: String
    /// GPT-backbone quant tier: fp16 (as-shipped) | int8 (near-lossless) | int4.
    public var quant: Quant
    /// Explicit checkpoint directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Explicit w2v-BERT directory (expects `model.safetensors` inside).
    public var w2vBertDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// Engine-stamped load headroom (`BudgetAware`): when tight, `load()` drops the GPT
    /// backbone fp16 → int8 (near-lossless) instead of failing admission.
    public var availableBudgetBytes: UInt64?

    public static let defaultRepo = "mlx-community/IndexTTS-2.5-fp16"

    public init(
        repo: String = IndexTTS2Configuration.defaultRepo,
        revision: String? = nil,
        w2vBertRepo: String = IndexTTS2Configuration.defaultRepo,
        quant: Quant = .fp16,
        modelDirectory: URL? = nil,
        w2vBertDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.w2vBertRepo = w2vBertRepo
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.w2vBertDirectory = w2vBertDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    // Environment-specific URLs + the engine-stamped budget are excluded from Codable.
    private enum CodingKeys: String, CodingKey {
        case repo, revision, w2vBertRepo, quant
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        w2vBertRepo = try c.decodeIfPresent(String.self, forKey: .w2vBertRepo) ?? repo
        quant = try c.decode(Quant.self, forKey: .quant)
    }
}

extension IndexTTS2Configuration: BudgetAware {}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension IndexTTS2Configuration: WeightSourcing {
    /// Everything `load()` opens from the main checkpoint (the feat/stats pickles are baked).
    static let mainFiles = [
        "gpt.safetensors", "codec.safetensors", "s2mel.safetensors", "bigvgan.safetensors",
        "multilingual_zh_ja_yue_char_del.tiktoken",
    ]
    static let w2vBertFile = "model.safetensors"

    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "main", repo: repo, revision: revision, matching: Self.mainFiles),
            WeightSource(role: "w2v-bert", repo: w2vBertRepo, revision: revision, matching: [Self.w2vBertFile]),
        ]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func has(_ dir: URL?, _ files: [String]) -> Bool {
            guard let dir else { return false }
            return files.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
        }
        let store = ModelStore(root: storeRoot)
        return weightSources.filter { source in
            switch source.role {
            case "main":
                return !(has(modelDirectory, Self.mainFiles) || has(store.directory(for: repo), Self.mainFiles))
            default:  // w2v-bert
                return !(has(w2vBertDirectory, [Self.w2vBertFile])
                         || has(store.directory(for: w2vBertRepo), [Self.w2vBertFile]))
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
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension IndexTTS2Configuration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        let r = resolved(storeRoot: modelsRootDirectory)
        var paths: [URL] = []
        if let dir = r.modelDirectory {
            paths += Self.mainFiles.filter { $0.hasSuffix(".safetensors") }.map { dir.appending(path: $0) }
        }
        if let dir = r.w2vBertDirectory { paths.append(dir.appending(path: Self.w2vBertFile)) }
        return paths
    }
}
