// ManifestTests.swift — offline conformance checks on the Stage-2 contract surface:
// two-layer license declaration (C7/C8: 2.5 weights allowlisted), requirements shape (C10), specialty declaration
// (C6), surface/capability derivation (C1/C2), and the E12 metaData emotion parser.

import Foundation
import MLXToolKit
import XCTest
@testable import MLXIndexTTS2TTS

final class ManifestTests: XCTestCase {

    let manifest = IndexTTS2Package.manifest

    func testLicenseGateTwoLayer() {
        // C7: the IndexTTS-2.5 weights (bilibili Model Use License Agreement) are on the
        // engine's permissive allowlist — admitted under the DEFAULT product policy, no
        // acknowledgement flow (the 2.0 non-commercial tier is gone).
        XCTAssertEqual(manifest.license.weightLicense, .bilibiliModelUse)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(manifest.license).isAdmitted)
        XCTAssertTrue(manifest.license.weightLicense.isPermissive)
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

    // MARK: - E12 typed plane (contract 1.38.0, AB-A-0049 part 3)

    func testDeclaresTheE12ControlsItActuallyImplements() {
        let controls = manifest.surfaces[0].ttsControls
        XCTAssertEqual(controls?.emotionModes, [.categorical, .vector])
        XCTAssertEqual(controls?.supportsTargetDuration, true)
        // Declaration derives advertisement (engine 1.38.0): both knobs advertised, nothing else.
        XCTAssertTrue(manifest.surfaces[0].parameters.contains { $0.name == "emotion" })
        XCTAssertTrue(manifest.surfaces[0].parameters.contains { $0.name == "targetDuration" })
        XCTAssertTrue(manifest.surfaces[0].controlsMatchCapability)
    }

    func testSharedVocabularyIsThePresetOrder() {
        // Every canonical name sits in `EmotionPresets.categories` at its own position — the
        // weight-vector order — proven through `presetIndex` (which looks the name up there).
        XCTAssertEqual(E12Emotion.allCases.map(\.presetIndex), Array(0 ..< 8))
        XCTAssertEqual(E12Emotion.allCases.map(\.rawValue),
                       ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"])
        XCTAssertEqual(E12Emotion.resolve(" Fearful "), .afraid)
        XCTAssertEqual(E12Emotion.resolve("neutral"), .calm)
        XCTAssertEqual(E12Emotion.resolve("other"), .calm)
        XCTAssertEqual(E12Emotion.resolve("unknown"), .calm)
        XCTAssertNil(E12Emotion.resolve("ecstatic"))
    }

    func testTypedCategoricalLandsOnTheSamePresetAsTheMetaDataString() throws {
        let typed = try IndexTTS2Package.resolveEmotionWeights(
            typed: .categorical("happy"), meta: nil, alpha: 0.6)
        let meta = try IndexTTS2Package.parseEmotion(.string("happy"), alpha: 0.6)
        XCTAssertEqual(typed, meta)
        XCTAssertEqual(typed?[0], 0.6)
        // An alias from the emotion2vec set resolves on BOTH paths.
        XCTAssertEqual(try IndexTTS2Package.resolveEmotionWeights(typed: .categorical("fearful"), meta: nil, alpha: 1.0)?[3], 1.0)
        XCTAssertEqual(try IndexTTS2Package.parseEmotion(.string("neutral"), alpha: 1.0)?[7], 1.0)
    }

    func testTypedVectorMatchesTheMetaDataArrayForm() throws {
        let typed = try IndexTTS2Package.resolveEmotionWeights(
            typed: .vector([0.5, 0, 0, 0, 0, 0, 0, 0.5]), meta: nil, alpha: 1.0)
        let meta = try IndexTTS2Package.parseEmotion(
            .array([.double(0.5), .int(0), .double(0), .double(0), .double(0), .double(0), .double(0), .double(0.5)]),
            alpha: 1.0)
        XCTAssertEqual(typed, meta)
        XCTAssertNil(try? IndexTTS2Package.resolveEmotionWeights(typed: .vector([1]), meta: nil, alpha: 1.0))
    }

    func testTypedWinsOverMetaDataAndUndeclaredModesAreRefused() throws {
        let both = try IndexTTS2Package.resolveEmotionWeights(
            typed: .categorical("sad"), meta: .string("happy"), alpha: 1.0)
        XCTAssertEqual(both?[2], 1.0)   // sad
        XCTAssertEqual(both?[0], 0.0)   // not happy
        XCTAssertNil(try? IndexTTS2Package.resolveEmotionWeights(
            typed: .textDescription("sound tired"), meta: nil, alpha: 1.0))
        XCTAssertNil(try? IndexTTS2Package.resolveEmotionWeights(
            typed: .referenceAudio(Audio(data: Data())), meta: nil, alpha: 1.0))
        XCTAssertNil(try? IndexTTS2Package.resolveEmotionWeights(
            typed: .categorical("ecstatic"), meta: nil, alpha: 1.0))
        // No typed value: the metaData path, unchanged.
        XCTAssertNil(try IndexTTS2Package.resolveEmotionWeights(typed: nil, meta: nil, alpha: 1.0))
    }
}
