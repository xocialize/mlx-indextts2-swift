// ManifestTests.swift — offline conformance checks on the Stage-2 contract surface:
// two-layer license gate behavior (C7/C8), requirements shape (C10), specialty declaration
// (C6), surface/capability derivation (C1/C2), and the E12 metaData emotion parser.

import Foundation
import MLXToolKit
import XCTest
@testable import MLXIndexTTS2TTS

final class ManifestTests: XCTestCase {

    let manifest = IndexTTS2Package.manifest

    func testLicenseGateTwoLayer() {
        // C7: NonCommercial weights REJECTED by the default product policy…
        XCTAssertFalse(LicensePolicy.permissiveOnly.evaluate(manifest.license).isAdmitted)
        if case .rejectedWeight(let license) = LicensePolicy.permissiveOnly.evaluate(manifest.license) {
            XCTAssertEqual(license, .indexTTS2Model)
        } else {
            XCTFail("expected the WEIGHT layer to be named (C8 legibility)")
        }
        // …and admitted only under the acknowledged eval policy.
        XCTAssertTrue(LicensePolicy.permissiveOrAcknowledged.evaluate(manifest.license).isAdmitted)
        // C8: the port code itself is permissive.
        XCTAssertTrue(manifest.license.portCodeLicense.isPermissive)
    }

    func testRequirementsAndSurfaces() {
        XCTAssertEqual(manifest.contractVersion, ContractVersion.current)   // C0
        XCTAssertEqual(manifest.capabilities, [.tts])                       // C1 (derived)
        XCTAssertEqual(Set(manifest.requirements.footprints.map(\.quant)),
                       [.fp16, .int8, .int4])                               // C10 per tier
        for footprint in manifest.requirements.footprints {
            XCTAssertGreaterThan(footprint.residentBytes, 0)
            XCTAssertGreaterThan(footprint.peakActivationBytes, 0)          // split declared
        }
        XCTAssertEqual(manifest.requirements.requiredBackends, [.metalGPU])
        // C6: zero-shot cloning selection axis + the two E12 control-plane specialties.
        XCTAssertEqual(Set(manifest.specialties.map(\.specialty)),
                       [.voiceClone, .emotionControl, .durationControl])
        // C11: descriptor is well-formed.
        let surface = manifest.surfaces[0]
        XCTAssertEqual(surface.capability, .tts)
        XCTAssertFalse(surface.summary.isEmpty)
        XCTAssertFalse(surface.parameters.isEmpty)
    }

    // MARK: - E12 emotion parser

    func testParseEmotionPresetName() throws {
        let weights = try IndexTTS2Package.parseEmotion(.string("happy"), alpha: 0.6)
        XCTAssertEqual(weights?[0], 0.6)                       // happy is category 0
        XCTAssertEqual(weights?.dropFirst().reduce(0, +), 0)
    }

    func testParseEmotionWeightedList() throws {
        let weights = try IndexTTS2Package.parseEmotion(.string("happy:0.8, calm:0.2"), alpha: 1.0)
        XCTAssertEqual(weights?[0] ?? 0, 0.8, accuracy: 1e-6)
        XCTAssertEqual(weights?[7] ?? 0, 0.2, accuracy: 1e-6)  // calm is category 7
    }

    func testParseEmotionVector() throws {
        let vector = MetaValue.array([.double(0.5), .int(0), .double(0), .double(0),
                                      .double(0), .double(0), .double(0), .double(0.5)])
        let weights = try IndexTTS2Package.parseEmotion(vector, alpha: 1.0)
        XCTAssertEqual(weights?[0] ?? 0, 0.5, accuracy: 1e-6)
        XCTAssertEqual(weights?[7] ?? 0, 0.5, accuracy: 1e-6)
    }

    func testParseEmotionRejectsLegibly() {
        XCTAssertNil(try? IndexTTS2Package.parseEmotion(.string("ecstatic"), alpha: 0.6))
        XCTAssertNil(try? IndexTTS2Package.parseEmotion(.array([.double(1)]), alpha: 0.6))
        XCTAssertNil(try? IndexTTS2Package.parseEmotion(.bool(true), alpha: 0.6))
        XCTAssertNoThrow(try IndexTTS2Package.parseEmotion(nil, alpha: 0.6))
    }
}
