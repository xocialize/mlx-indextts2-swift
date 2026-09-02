// MaterializationTests.swift — IndexTTS2 through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution — run per selectable quant tier (the declaration must be
// quant-invariant here: int8/int4 quantize in-memory at load, so sources never change).

import Foundation
import MLXServeConformance
import MLXToolKit
import MLXIndexTTS2
import XCTest
@testable import MLXIndexTTS2TTS

final class MaterializationTests: XCTestCase {

    /// Temp dirs holding probe files that make an explicit-dir config read as satisfied.
    private func satisfiedDirs() throws -> (main: URL, w2v: URL, cleanup: () -> Void) {
        let base = FileManager.default.temporaryDirectory
            .appending(path: "indextts2-mat-\(UUID().uuidString)")
        let main = base.appending(path: "main")
        let w2v = base.appending(path: "w2v")
        try FileManager.default.createDirectory(at: main, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: w2v, withIntermediateDirectories: true)
        for f in IndexTTS2Configuration.mainFiles {
            FileManager.default.createFile(atPath: main.appending(path: f).path, contents: Data([0]))
        }
        FileManager.default.createFile(
            atPath: w2v.appending(path: "model.safetensors").path, contents: Data([0]))
        return (main, w2v, { try? FileManager.default.removeItem(at: base) })
    }

    // MARK: - Engine MAT gate, per quant tier

    func testMATGatePerQuantTier() throws {
        let (main, w2v, cleanup) = try satisfiedDirs()
        defer { cleanup() }
        for quant in [Quant.fp16, .int8, .int4] {
            let report = MaterializationConformance.check(
                freshConfiguration: IndexTTS2Configuration(quant: quant),
                satisfiedConfiguration: IndexTTS2Configuration(
                    quant: quant, modelDirectory: main, w2vBertDirectory: w2v))
            XCTAssertTrue(report.passed, "quant \(quant): \(report.summary)")
        }
    }

    // MARK: - Source declaration shape

    func testDeclaresTwoSourcesQuantInvariant() {
        let fp16 = IndexTTS2Configuration().weightSources
        XCTAssertEqual(fp16.map(\.role), ["main", "w2v-bert"])
        XCTAssertEqual(fp16[0].repo, "mlx-community/IndexTTS-2.5-fp16")
        XCTAssertEqual(fp16[0].matching, IndexTTS2Configuration.mainFiles)
        XCTAssertTrue(IndexTTS2Configuration.mainFiles.contains(IndexTTS2Generator.tokenizerFile))
        XCTAssertEqual(fp16[1].repo, "mlx-community/IndexTTS-2.5-fp16")
        XCTAssertEqual(fp16[1].matching, ["model.safetensors"])
        // Quant tiers quantize in-memory at load — the materialization set never changes.
        let int4 = IndexTTS2Configuration(quant: .int4).weightSources
        XCTAssertEqual(fp16, int4)
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "indextts2-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = IndexTTS2Configuration()
        // Empty store: everything missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 2)
        // Populate the expected layout — paths from ModelStore so the fixture tracks the
        // engine's canonical models--org--name layout (contract 1.22.0) instead of a stale literal.
        let store = ModelStore(root: root)
        let mainDir = store.directory(for: cfg.repo)!
        let w2vDir = store.directory(for: cfg.w2vBertRepo)!
        try FileManager.default.createDirectory(at: mainDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: w2vDir, withIntermediateDirectories: true)
        for f in IndexTTS2Configuration.mainFiles {
            FileManager.default.createFile(atPath: mainDir.appending(path: f).path, contents: Data([0]))
        }
        FileManager.default.createFile(
            atPath: w2vDir.appending(path: "model.safetensors").path, contents: Data([0]))
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout; explicit dirs always win.
        let resolved = cfg.resolved(storeRoot: root)
        XCTAssertEqual(resolved.modelDirectory?.path, mainDir.path)
        XCTAssertEqual(resolved.w2vBertDirectory?.path, w2vDir.path)
        let explicit = IndexTTS2Configuration(modelDirectory: URL(fileURLWithPath: "/x"))
            .resolved(storeRoot: root)
        XCTAssertEqual(explicit.modelDirectory?.path, "/x")
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = IndexTTS2Configuration(modelsRootDirectory: root)
        let paths = cfg.prewarmPaths.map(\.path)
        // Expectations from ModelStore, not literals — the layout is the engine's to define.
        let store = ModelStore(root: root)
        XCTAssertTrue(paths.contains(
            store.directory(for: cfg.repo)!.appending(path: "gpt.safetensors").path))
        XCTAssertTrue(paths.contains(
            store.directory(for: cfg.w2vBertRepo)!.appending(path: "model.safetensors").path))
        XCTAssertFalse(paths.contains { $0.hasSuffix(".tiktoken") }, "prewarm is for weight files only")
    }

    func testCodableRoundTrip() throws {
        let cfg = IndexTTS2Configuration(revision: "abc123", quant: .int8)
        let decoded = try JSONDecoder().decode(IndexTTS2Configuration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.repo, cfg.repo)
        XCTAssertEqual(decoded.revision, "abc123")
        XCTAssertEqual(decoded.quant, .int8)
        XCTAssertNil(decoded.modelDirectory)   // environment-specific, never encoded
    }
}
