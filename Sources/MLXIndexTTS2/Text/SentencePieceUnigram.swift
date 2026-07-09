// SentencePieceUnigram.swift — minimal SentencePiece Unigram encoder for IndexTTS2.
//
// The shipped `tokenizer.model` is an SP Unigram model (12k pieces, nmt_nfkc,
// add_dummy_prefix, remove_extra_whitespaces, no byte_fallback). Rather than pull a protobuf
// dependency, the conversion dumps (piece, score, type) to JSON (tools/dump_tokenizer.py in
// the oracle workspace) and this implements the Unigram Viterbi directly:
//   normalize (NFKC + NMT cleanup + collapse + dummy prefix + space→▁) → trie-matched
//   Viterbi over scalars (max sum of piece scores) → merge adjacent unknowns into one <unk>.
// Unknown score = minScore − 10.0 (SP's kUnkPenalty). CONTROL pieces (<s>, </s>) never match;
// USER_DEFINED pieces participate with their stored score. Parity gated bit-exact against the
// oracle's EncodeAsIds/EncodeAsPieces over the fixture corpus.

import Foundation

public struct SentencePieceUnigram: @unchecked Sendable {

    public struct Piece: Codable {
        public let p: String   // piece text
        public let s: Double   // log score
        public let t: Int      // proto type: 1 normal, 2 unknown, 3 control, 4 user-defined, 6 byte
    }
    private struct VocabFile: Codable {
        let unk_id: Int
        let vocab: [Piece]
    }

    private final class TrieNode {
        var children: [UInt32: TrieNode] = [:]
        var terminal: (id: Int, score: Double)? = nil
    }

    private let root = TrieNode()
    private let pieces: [Piece]
    public let unkID: Int
    private let unkScore: Double

    public init(vocabURL: URL) throws {
        let file = try JSONDecoder().decode(VocabFile.self, from: Data(contentsOf: vocabURL))
        pieces = file.vocab
        unkID = file.unk_id
        var minScore = 0.0
        for (id, piece) in file.vocab.enumerated() {
            // Trie holds only pieces that may match text: normal / user-defined / byte.
            guard piece.t == 1 || piece.t == 4 || piece.t == 6 else { continue }
            if piece.t == 1 { minScore = min(minScore, piece.s) }
            var node = root
            for scalar in piece.p.unicodeScalars {
                if node.children[scalar.value] == nil { node.children[scalar.value] = TrieNode() }
                node = node.children[scalar.value]!
            }
            node.terminal = (id, piece.s)
        }
        unkScore = minScore - 10.0  // sentencepiece kUnkPenalty
    }

    public func idToPiece(_ id: Int) -> String { pieces[id].p }

    // MARK: - SP-side normalization (nmt_nfkc + options)

    /// NFKC + NMT cleanup + extra-whitespace removal + dummy prefix + space→▁.
    func spNormalize(_ text: String) -> [Unicode.Scalar] {
        // 1. NFKC (covers …→..., NBSP→space, fullwidth→ASCII, etc.)
        let nfkc = text.precomposedStringWithCompatibilityMapping
        // 2. NMT cleanup: line separators → space; drop controls + zero-width/format chars.
        var mapped = [Unicode.Scalar]()
        for s in nfkc.unicodeScalars {
            switch s.value {
            case 0x09, 0x0A, 0x0D, 0x85, 0x2028, 0x2029:
                mapped.append(" ")
            case 0x00...0x1F, 0x7F, 0x80...0x9F,           // C0/C1 controls
                 0x200B...0x200F, 0x202A...0x202E, 0x2060, 0xFEFF:  // zero-width/format
                continue
            default:
                mapped.append(s)
            }
        }
        // 3. remove_extra_whitespaces: collapse runs of spaces, trim.
        var collapsed = [Unicode.Scalar]()
        var inSpace = false
        for s in mapped {
            if s == " " {
                if !inSpace { collapsed.append(" ") }
                inSpace = true
            } else {
                collapsed.append(s)
                inSpace = false
            }
        }
        while collapsed.first == " " { collapsed.removeFirst() }
        while collapsed.last == " " { collapsed.removeLast() }
        guard !collapsed.isEmpty else { return [] }
        // 4. add_dummy_prefix + space→▁ (U+2581)
        var out: [Unicode.Scalar] = ["\u{2581}"]
        for s in collapsed {
            out.append(s == " " ? "\u{2581}" : s)
        }
        return out
    }

    // MARK: - Viterbi encode

    public func encode(_ text: String) -> [Int] { encodeWithPieces(text).map(\.id) }

    public func encodeAsPieces(_ text: String) -> [String] { encodeWithPieces(text).map(\.surface) }

    public func encodeWithPieces(_ text: String) -> [(id: Int, surface: String)] {
        let chars = spNormalize(text)
        guard !chars.isEmpty else { return [] }
        let n = chars.count

        // best[i] = best score covering chars[0..<i]; back[i] = (start, pieceID or nil=unk)
        var best = [Double](repeating: -.infinity, count: n + 1)
        var back = [(start: Int, id: Int?)](repeating: (0, nil), count: n + 1)
        best[0] = 0

        for i in 0..<n {
            guard best[i] > -.infinity else { continue }
            // unk covers one scalar
            if best[i] + unkScore > best[i + 1] {
                best[i + 1] = best[i] + unkScore
                back[i + 1] = (i, nil)
            }
            // trie matches from i
            var node = root
            var j = i
            while j < n, let next = node.children[chars[j].value] {
                node = next
                j += 1
                if let term = node.terminal, best[i] + term.score > best[j] {
                    best[j] = best[i] + term.score
                    back[j] = (i, term.id)
                }
            }
        }

        // Backtrace
        var spans: [(start: Int, end: Int, id: Int?)] = []
        var pos = n
        while pos > 0 {
            let (start, id) = back[pos]
            spans.append((start, pos, id))
            pos = start
        }
        spans.reverse()

        // Merge adjacent unknowns into one <unk> (surface = covered text), like sentencepiece.
        var out: [(id: Int, surface: String)] = []
        var i = 0
        while i < spans.count {
            let span = spans[i]
            if span.id == nil {
                var end = span.end
                var k = i + 1
                while k < spans.count, spans[k].id == nil {
                    end = spans[k].end
                    k += 1
                }
                let surface = String(String.UnicodeScalarView(chars[span.start..<end]))
                out.append((unkID, surface))
                i = k
            } else {
                let surface = String(String.UnicodeScalarView(chars[span.start..<span.end]))
                out.append((span.id!, surface))
                i += 1
            }
        }
        return out
    }
}
