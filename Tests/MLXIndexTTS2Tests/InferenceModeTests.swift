// InferenceModeTests.swift — IndexTTS2 through the engine's INF gate (C14).
//
// The BatchNorm carrier is **CAMPPlus**, the speaker encoder whose 192-d embedding conditions the
// entire voice clone. It is BatchNorm-dense (`batchnorm`, `bn`, `bn1`, `bn2` across the TDNN /
// CAMDense / FSMN blocks). `MLXNN.Module.training` defaults to `true`, and in that state BatchNorm
// normalizes by the CURRENT batch's statistics and OVERWRITES the checkpoint's
// running_mean/running_var on every forward — so the speaker embedding would be computed from the
// reference clip's own statistics and would drift run to run. The other five components are
// LayerNorm/RMSNorm/GroupNorm and structurally unaffected.
//
// POSTURE OF RECORD: `.moduleGraph`.
//
// CHOKE POINT: `IndexTTS2Generator.loadComponent(_:url:sanitize:)` (2026-07-25). Inference mode
// used to be set only next to the CAMPPlus construction site — fix-by-repetition across four
// places. All six components load through the one function, so that is where it belongs.
//
// No download needed: these round-trip a component's OWN parameters through a temp safetensors
// (with an identity `sanitize`, since the keys are already module-shaped), which exercises the real
// loader including its 0-missing / 0-unused key contract.

import XCTest
import Foundation
import MLX
import MLXNN
import MLXServeConformance
import MLXServeConformanceNN
@testable import MLXIndexTTS2
@testable import MLXIndexTTS2TTS

extension IndexTTS2Package: InferenceModeInspectable {
    public func inferenceModeFlags() -> [InferenceModeConformance.ModuleTrainingFlag] {
        InferenceModeConformance.flags(of: inferenceModeGraphs)
    }
}

final class InferenceModeTests: XCTestCase {

    private func writeSyntheticCheckpoint(for module: Module) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("indextts2-inf-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: module.parameters().flattened()),
                     url: url)
        return url
    }

    /// INF-1's green assertion, offline. A fresh CAMPPlus is in training mode (the MLXNN default);
    /// going through the choke point puts it in inference mode.
    func testLoadComponentIsTheInferenceModeChokePoint() throws {
        let model = CAMPPlus()

        let before = InferenceModeConformance.check(
            flags: InferenceModeConformance.flags(of: model), posture: .moduleGraph)
        XCTAssertFalse(before.passed, "a fresh CAMPPlus must be in training mode (MLXNN default)")
        XCTAssertTrue(before.summary.contains("(BatchNorm)"),
                      "the failure should name the running-statistic layers:\n\(before.summary)")

        let url = try writeSyntheticCheckpoint(for: model)
        defer { try? FileManager.default.removeItem(at: url) }
        // Identity sanitize: the saved keys are already the module's own, so the checkpoint-key
        // remap CAMPPlus.sanitize performs would be wrong to apply twice.
        let loaded = try IndexTTS2Generator.loadComponent(model, url: url, sanitize: { $0 })

        let after = InferenceModeConformance.check(
            flags: InferenceModeConformance.flags(of: loaded), posture: .moduleGraph)
        XCTAssertTrue(after.passed,
                      "loadComponent must leave the component in inference mode:\n\(after.summary)")
    }

    /// The choke point is generic over `Module`, so it must put EVERY component in inference mode,
    /// not just the one that happens to carry BatchNorms today.
    func testChokePointAppliesToAnyComponent() throws {
        let model = EnhancedCodecDecoder()
        let url = try writeSyntheticCheckpoint(for: model)
        defer { try? FileManager.default.removeItem(at: url) }
        let loaded = try IndexTTS2Generator.loadComponent(model, url: url, sanitize: { $0 })
        XCTAssertTrue(InferenceModeConformance.flags(of: loaded).allSatisfy { !$0.training })
    }

    /// CAMPPlus's BatchNorms must be reachable by the walk — if a refactor re-nests them out of
    /// scope the gate would be watching nothing and still report green.
    func testCampPlusBatchNormsAreInScope() {
        let flags = InferenceModeConformance.flags(of: CAMPPlus())
        let sensitive = flags.filter { InferenceModeConformance.isTrainingSensitive(type: $0.type) }
        XCTAssertGreaterThan(sensitive.count, 10,
                             "expected the CAMPPlus BatchNorm stack in scope, saw \(sensitive.count)")
    }

    /// The seam reads real state: an unloaded package reports no modules, which INF-1 fails.
    func testUnloadedPackageFailsINF1() async {
        let pkg = IndexTTS2Package(configuration: IndexTTS2Configuration())
        let report = await InferenceModeConformance.check(pkg, posture: .moduleGraph)
        XCTAssertFalse(report.passed, "an unloaded package must not pass INF-1:\n\(report.summary)")
        XCTAssertTrue(report.summary.contains("no modules observed"), report.summary)
    }
}
