// TokenizerParityTests.swift — P1 parity gate.
//
// Fixtures were dumped from the oracle (tools/dump_tokenizer.py against solar2ain
// TextTokenizer + the shipped tokenizer.model). The gate is BIT-EXACT: for every fixture,
// the Swift normalizer must reproduce `normalized` and the Swift tokenizer must reproduce
// `ids` and `pieces` exactly. Fixture 0 is the Stage-0 golden sentence, so passing here
// includes the goldens' `text_tokens` (core_gpt_latent__in0).

import XCTest
@testable import MLXIndexTTS2

private struct Fixture: Codable {
    let text: String
    let normalized: String
    let ids: [Int]
    let pieces: [String]
}

final class TokenizerParityTests: XCTestCase {

    private static let resources = Bundle.module.resourceURL!.appending(path: "Resources")

    private static let tokenizer = try! IndexTTSTextTokenizer(
        vocabURL: resources.appending(path: "tokenizer_vocab.json"))

    private static let fixtures = try! JSONDecoder().decode(
        [Fixture].self,
        from: Data(contentsOf: resources.appending(path: "tokenizer_fixtures.json")))

    func testNormalizerParity() {
        for f in Self.fixtures {
            XCTAssertEqual(Self.tokenizer.normalize(f.text), f.normalized,
                           "normalize mismatch for \(f.text.debugDescription)")
        }
    }

    func testEncodeParity() {
        for f in Self.fixtures {
            XCTAssertEqual(Self.tokenizer.encode(f.text), f.ids,
                           "ids mismatch for \(f.text.debugDescription)")
        }
    }

    func testPiecesParity() {
        for f in Self.fixtures {
            XCTAssertEqual(Self.tokenizer.tokenize(f.text), f.pieces,
                           "pieces mismatch for \(f.text.debugDescription)")
        }
    }

    func testGoldenSentenceMatchesStage0() {
        // The Stage-0 goldens' text_tokens (core_gpt_latent__in0.npy).
        let golden = [10204, 10934, 11744, 10201, 10421, 10395, 10393, 11244, 10208,
                      10311, 10204, 10201, 10694, 10634, 10344, 11329, 10216]
        XCTAssertEqual(
            Self.tokenizer.encode("The quick brown fox jumps over the lazy dog."), golden)
    }

    func testEmptyAndWhitespace() {
        XCTAssertEqual(Self.tokenizer.encode(""), [])
        XCTAssertEqual(Self.tokenizer.encode("   "), [])
        XCTAssertEqual(Self.tokenizer.normalize(""), "")
    }

    func testSplitSegmentsBasic() {
        // Sanity (behavioral, not oracle-gated yet): sentences split at terminal punctuation
        // and short segments merge under the cap.
        let tokens = Self.tokenizer.tokenize("One. Two. Three.")
        let segments = Self.tokenizer.splitSegments(tokens, maxTokensPerSegment: 4)
        XCTAssertFalse(segments.isEmpty)
        XCTAssertEqual(segments.flatMap { $0 }, tokens, "segmentation must not drop tokens")
    }
}
