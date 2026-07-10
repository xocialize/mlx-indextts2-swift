// CancellationTests.swift — IndexTTS2 through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or weights); CAN-3 is the document of record for the checkpoint cadence: the
// GPT-backbone AR loop bails per generated mel token (`Task.isCancelled` break in
// UnifiedVoiceV2.generateMelCodes — core folded into this repo), and the throwing `cancelCheck`
// closure the wrapper passes to IndexTTS2Generator.synthesize checkpoints between every pipeline
// stage (per-segment AR, S2Mel length-regulate, CFM denoise, BigVGAN vocode) and immediately
// after the AR phase — so a cancel is never laundered into IndexTTS2Error.emptyGeneration.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXIndexTTS2TTS

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation (including the referenceAudio-only voice gate) or weights are touched,
        // so this is offline-safe.
        let package = IndexTTS2Package(configuration: IndexTTS2Configuration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: TTSRequest(text: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // tts is a long-run capability (and the 5 GB declared transient independently implies
        // long runs) — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: IndexTTS2Package.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: IndexTTS2Package.manifest,
            posture: .cadence([
                // The GPT AR driver checks Task.isCancelled once per generated mel token
                // (UnifiedVoiceV2+Generate.swift, generateMelCodes loop — up to 1500/segment).
                .init(phase: .generate, unit: .token),
                // The wrapper's throwing cancelCheck closure fires between pipeline stages:
                // per-segment before AR, post-AR (before the emptyGeneration guard), before
                // the CFM mel denoise, and before the BigVGAN vocode
                // (IndexTTS2Generator.synthesize) — i.e. once per segment chunk per stage.
                .init(phase: .decode, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
