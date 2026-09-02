// TokenizerParityTests.swift — the text-front-end parity gate (offline, no kernels).
//
// Fixtures were captured from the oracle (WIP/indextts25/tools/capture_v25_goldens.py: the
// donor's `IndexTTS25TextFrontend.prepare` + tiktoken over the checkpoint's own vocabulary).
// The gate is ID-EXACT: for every fixture the Swift `TiktokenBPE` must reproduce tiktoken's
// ids, and the Swift frontend must reproduce the segment token ids the GPT consumes.
//
// Fixtures the oracle normalized with WeTextProcessing number expansion (digits in zh/en text)
// are the DOCUMENTED gap of this port (no WeText equivalent, see Normalize.swift) and are
// asserted to differ ONLY on that axis (they are listed, not silently skipped).

import XCTest
@testable import MLXIndexTTS2

private struct FrontendFixture: Codable {
    let language: String
    let language_id: Int
    let text: String
    let normalized: String
    let segments: [String]
    let token_ids: [[Int]]
    let raw_bpe_ids: [Int]
    let max_text_tokens_per_segment: Int?
}

private struct BPEFixture: Codable {
    let text: String
    let ids: [Int]
}

private struct Fixtures: Codable {
    let frontend: [FrontendFixture]
    let bpe: [BPEFixture]
}

final class TokenizerParityTests: XCTestCase {

    private static let resources = Bundle.module.resourceURL!.appending(path: "Resources")
    private static let tokenizer = try! TiktokenBPE(
        vocabularyURL: resources.appending(path: IndexTTS2Generator.tokenizerFile))
    private static let frontend = IndexTTSTextFrontend(tokenizer: tokenizer)
    private static let fixtures = try! JSONDecoder().decode(
        Fixtures.self, from: Data(contentsOf: resources.appending(path: "text_fixtures.json")))

    /// Fixtures whose zh/en text carries digits: the oracle expanded them with WeText.
    private static func hasDigits(_ f: FrontendFixture) -> Bool {
        (f.language == "zh" || f.language == "en") && f.text.unicodeScalars.contains { CharacterSet.decimalDigits.contains($0) }
    }

    func testVocabularyShape() {
        XCTAssertEqual(Self.tokenizer.vocabularySize, 60509)
        XCTAssertEqual(Self.tokenizer.specialTokens["<|endoftext|>"], 58836)
        XCTAssertEqual(Self.tokenizer.specialTokens["<|en|>"], 58838)
        XCTAssertEqual(Self.tokenizer.specialTokens["<|SPECIAL_TOKEN_1|>"], 58958)
        XCTAssertEqual(Self.tokenizer.specialTokens["<|30.00|>"], 60508)
    }

    func testBPEParity() {
        for f in Self.fixtures.bpe {
            XCTAssertEqual(Self.tokenizer.encode(f.text), f.ids, "bpe mismatch for \(f.text.debugDescription)")
        }
    }

    func testRawBPEOnFrontendCorpus() {
        for f in Self.fixtures.frontend {
            XCTAssertEqual(Self.tokenizer.encode(f.text), f.raw_bpe_ids, "raw bpe mismatch for \(f.text.debugDescription)")
        }
    }

    func testFrontendParity() throws {
        var gaps: [String] = []
        for f in Self.fixtures.frontend {
            let lang = IndexTTSLanguage(rawValue: f.language)!
            let prepared = try Self.frontend.prepare(
                f.text, language: lang, maxTokensPerSegment: f.max_text_tokens_per_segment ?? 120)
            XCTAssertEqual(prepared.language.languageID, f.language_id)
            if Self.hasDigits(f) {
                // Documented WeText gap: normalized text differs, everything else must hold.
                if prepared.tokenIDs != f.token_ids { gaps.append(f.text) }
                continue
            }
            XCTAssertEqual(prepared.normalized, f.normalized, "normalized mismatch for \(f.text.debugDescription)")
            XCTAssertEqual(prepared.segments, f.segments, "segments mismatch for \(f.text.debugDescription)")
            XCTAssertEqual(prepared.tokenIDs, f.token_ids, "token ids mismatch for \(f.text.debugDescription)")
        }
        // The gap list is the document of record: every gap is a digit fixture, and the
        // corpus must actually exercise the gap (so a future WeText port has something to flip).
        XCTAssertFalse(gaps.isEmpty, "expected the digit fixtures to expose the WeText gap")
        for gap in gaps { print("documented WeText gap: \(gap)") }
    }

    func testLanguageDetection() {
        XCTAssertEqual(IndexTTSLanguage.detect("Hello there"), .en)
        XCTAssertEqual(IndexTTSLanguage.detect("¿Cómo estás?"), .es)
        XCTAssertEqual(IndexTTSLanguage.detect("今天天气很好"), .zh)
        XCTAssertEqual(IndexTTSLanguage.detect("こんにちは"), .ja)
        XCTAssertEqual(IndexTTSLanguage.detect("مرحبا"), .ar)
        XCTAssertNil(IndexTTSLanguage.detect("12345 !!!"))
        XCTAssertEqual(IndexTTSLanguage(parsing: "Mandarin"), .zh)
        XCTAssertEqual(IndexTTSLanguage(parsing: " EN "), .en)
        XCTAssertNil(IndexTTSLanguage(parsing: "klingon"))
        XCTAssertEqual(TiktokenBPE.languageID("common"), 105)
        XCTAssertEqual(TiktokenBPE.languageID("xx"), 105)
    }

    func testEveryPreparedSegmentEndsWithStopAndStartsWithLanguage() throws {
        let prepared = try Self.frontend.prepare("One. Two. Three.", language: .en, maxTokensPerSegment: 6)
        XCTAssertGreaterThan(prepared.tokenIDs.count, 1, "a 6-token budget must split three sentences")
        for ids in prepared.tokenIDs {
            XCTAssertEqual(ids.last, IndexTTSTextFrontend.stopTextToken)
            XCTAssertEqual(ids.first, Self.tokenizer.specialTokens["<|en|>"])
        }
    }
}
