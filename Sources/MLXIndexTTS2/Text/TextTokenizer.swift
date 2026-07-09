// TextTokenizer.swift — IndexTTS2 text front-end (P1).
//
// Pipeline mirrors mlx_indextts/tokenizer.py `TextTokenizer`:
//   normalize (IndexTTSTextNormalizer) → tokenize_by_cjk_char (CJK spacing + UPPERCASE)
//   → SentencePiece Unigram encode.
// Plus `splitSegments` — the recursive punctuation/comma/hyphen long-text splitter.

import Foundation

/// Sentence-boundary tokens used by `splitSegments` (PUNCTUATION_MARKS_TOKENS).
private let punctuationMarkTokens: Set<String> = [
    ".", "!", "?", "\u{2581}.", "\u{2581}?", "...", "\u{2581}...",
]

public final class IndexTTSTextTokenizer: @unchecked Sendable {

    public let sp: SentencePieceUnigram
    public let normalizer: IndexTTSTextNormalizer

    public init(vocabURL: URL, normalizer: IndexTTSTextNormalizer = IndexTTSTextNormalizer()) throws {
        self.sp = try SentencePieceUnigram(vocabURL: vocabURL)
        self.normalizer = normalizer
    }

    public func normalize(_ text: String) -> String { normalizer.normalize(text) }

    public func tokenize(_ text: String, normalize: Bool = true) -> [String] {
        var t = normalize ? normalizer.normalize(text) : text
        t = tokenizeByCJKChar(t)
        return sp.encodeAsPieces(t)
    }

    public func encode(_ text: String, normalize: Bool = true) -> [Int] {
        var t = normalize ? normalizer.normalize(text) : text
        t = tokenizeByCJKChar(t)
        return sp.encode(t)
    }

    // MARK: - Long-text segmentation (port of _split_segments_by_token)

    public func splitSegments(_ tokens: [String], maxTokensPerSegment: Int = 120) -> [[String]] {
        Self.splitSegments(tokens, splitTokens: punctuationMarkTokens,
                           maxTokensPerSegment: maxTokensPerSegment)
    }

    static func splitSegments(
        _ tokens: [String], splitTokens: Set<String>, maxTokensPerSegment: Int
    ) -> [[String]] {
        guard !tokens.isEmpty else { return [] }

        var segments: [[String]] = []
        var current: [String] = []

        var i = 0
        while i < tokens.count {
            let token = tokens[i]
            current.append(token)

            let splitsOnComma = splitTokens.contains(",") || splitTokens.contains("\u{2581},")
            let splitsOnHyphen = splitTokens.contains("-")

            if !splitsOnComma && (current.contains(",") || current.contains("\u{2581},")) {
                segments.append(contentsOf: splitSegments(
                    current, splitTokens: [",", "\u{2581},"],
                    maxTokensPerSegment: maxTokensPerSegment))
                current = []
            } else if !splitsOnHyphen && current.contains("-") {
                segments.append(contentsOf: splitSegments(
                    current, splitTokens: ["-"], maxTokensPerSegment: maxTokensPerSegment))
                current = []
            } else if current.count <= maxTokensPerSegment {
                if splitTokens.contains(token) && current.count > 2 {
                    // Don't split just before a quote token.
                    if i < tokens.count - 1 {
                        let next = tokens[i + 1]
                        if next == "'" || next == "\u{2581}'" {
                            current.append(next)
                            i += 1
                        }
                    }
                    segments.append(current)
                    current = []
                }
            } else {
                // Force split at maxTokensPerSegment.
                var j = 0
                while j < current.count {
                    let end = min(j + maxTokensPerSegment, current.count)
                    segments.append(Array(current[j..<end]))
                    j = end
                }
                current = []
            }
            i += 1
        }
        if !current.isEmpty { segments.append(current) }

        // Merge short adjacent segments.
        var merged: [[String]] = []
        for segment in segments where !segment.isEmpty {
            if merged.isEmpty {
                merged.append(segment)
            } else if merged[merged.count - 1].count + segment.count <= maxTokensPerSegment {
                // (The Python's two merge branches — ≤max/2 and ≤max — both just append; the
                // effective condition is combined length ≤ max.)
                merged[merged.count - 1] += segment
            } else {
                merged.append(segment)
            }
        }
        return merged
    }
}
