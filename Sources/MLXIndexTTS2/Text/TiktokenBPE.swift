// TiktokenBPE.swift — the IndexTTS-2.5 text tokenizer: a tiktoken byte-level BPE over the
// checkpoint's `multilingual_zh_ja_yue_char_del.tiktoken` (58 836 mergeable ranks) plus the
// Whisper-style special-token table (1673 specials → n_vocab 60 509), reproduced from
// `indextts/utils/tokenizer.py get_encoding`.
//
// Algorithm = tiktoken's: split on special tokens (allowed_special="all"), pre-tokenize each
// span with the GPT-2-style pattern (ICU `\p{L}` / `\p{N}` / `\s` — same classes tiktoken's
// Rust regex resolves), then byte-pair-merge each piece by lowest rank first (`byte_pair_merge`).
// Gated id-exact against tiktoken on a corpus (Tests/…/text_fixtures.json `bpe`).
//
// The vocabulary is read from the weight directory (it ships with the checkpoint) — nothing
// is baked into the package.

import Foundation

public final class TiktokenBPE: @unchecked Sendable {

    public enum LoadError: Error, CustomStringConvertible {
        case unreadable(String)
        case malformedLine(Int)
        public var description: String {
            switch self {
            case .unreadable(let p): return "tiktoken vocabulary unreadable: \(p)"
            case .malformedLine(let n): return "tiktoken vocabulary: malformed line \(n)"
            }
        }
    }

    /// Byte sequence → rank (merge priority AND token id for mergeable tokens).
    private let ranks: [[UInt8]: Int]
    /// Special-token string → id, and the alternation regex that finds them.
    public let specialTokens: [String: Int]
    private let specialRegex: NSRegularExpression
    private let pattern: NSRegularExpression
    public let vocabularySize: Int

    /// Whisper's multilingual language table — the FIRST 99 entries become `<|xx|>` specials
    /// (upstream `list(LANGUAGES.keys())[:num_languages]`); the whole table (106 entries) is
    /// the `lang_to_token` id space used by the GPT's `lang_embedding`.
    public static let languageCodes: [String] = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
        "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
        "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
        "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
        "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
        "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
        "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
        "common",
    ]

    /// `lang_to_token`: index into the full language table (unknown → "common").
    public static func languageID(_ code: String) -> Int {
        let lowered = code.lowercased()
        return languageCodes.firstIndex(of: lowered) ?? languageCodes.firstIndex(of: "common")!
    }

    /// The special-token strings in registration order (ids follow the mergeable ranks).
    static let specialTokenStrings: [String] = {
        var s = ["<|endoftext|>", "<|startoftranscript|>"]
        s += languageCodes.prefix(99).map { "<|\($0)|>" }
        s += ["ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter",
              "Applause", "/Applause"].map { "<|\($0)|>" }
        s += ["HAPPY", "SAD", "ANGRY", "NEUTRAL"].map { "<|\($0)|>" }
        s += ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
              "<|nospeech|>", "<|notimestamps|>"]
        s += (1...30).map { "<|SPECIAL_TOKEN_\($0)|>" }
        s += ["TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H"].map { "<|\($0)|>" }
        s += (1...13).map { String(format: "<|TTS/SP%02d|>", $0) }
        s += (0...1500).map { String(format: "<|%.2f|>", Double($0) * 0.02) }
        return s
    }()

    /// tiktoken `pat_str` for this encoding (GPT-2 family).
    static let patternString =
        #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

    public init(vocabularyURL: URL) throws {
        guard let text = try? String(contentsOf: vocabularyURL, encoding: .utf8) else {
            throw LoadError.unreadable(vocabularyURL.path)
        }
        var ranks: [[UInt8]: Int] = [:]
        var lineNumber = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            lineNumber += 1
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            guard parts.count == 2, let rank = Int(parts[1]) else { throw LoadError.malformedLine(lineNumber) }
            let b64 = String(parts[0])
            // One vocabulary entry is the EMPTY byte string, spelled "=" (Python's b64decode
            // accepts it; Foundation's does not). It can never match a piece, but it holds a rank.
            if b64.allSatisfy({ $0 == "=" }) { ranks[[]] = rank; continue }
            guard let data = Data(base64Encoded: b64) else { throw LoadError.malformedLine(lineNumber) }
            ranks[[UInt8](data)] = rank
        }
        self.ranks = ranks
        var specials: [String: Int] = [:]
        var next = ranks.count
        for token in Self.specialTokenStrings {
            specials[token] = next
            next += 1
        }
        self.specialTokens = specials
        self.vocabularySize = next
        let escaped = Self.specialTokenStrings.map { NSRegularExpression.escapedPattern(for: $0) }
        self.specialRegex = try NSRegularExpression(pattern: escaped.joined(separator: "|"))
        self.pattern = try NSRegularExpression(pattern: Self.patternString)
    }

    // MARK: - Encoding

    /// `Encoding.encode(text, allowed_special="all")`.
    public func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        let ns = text as NSString
        var cursor = 0
        for match in specialRegex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            if match.range.location > cursor {
                ids += encodeOrdinary(ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor)))
            }
            ids.append(specialTokens[ns.substring(with: match.range)]!)
            cursor = match.range.location + match.range.length
        }
        if cursor < ns.length { ids += encodeOrdinary(ns.substring(from: cursor)) }
        return ids
    }

    /// Token count of `text` (the frontend's segmentation budget probe).
    public func tokenCount(_ text: String) -> Int { encode(text).count }

    /// `encode_ordinary`: pre-tokenize, then BPE each piece.
    public func encodeOrdinary(_ text: String) -> [Int] {
        var ids: [Int] = []
        let ns = text as NSString
        for match in pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let piece = [UInt8](ns.substring(with: match.range).utf8)
            if let whole = ranks[piece] {
                ids.append(whole)
            } else {
                ids += bytePairMerge(piece)
            }
        }
        return ids
    }

    /// tiktoken `byte_pair_merge`: repeatedly merge the adjacent pair with the lowest rank.
    private func bytePairMerge(_ piece: [UInt8]) -> [Int] {
        // parts[i] = (start offset, rank of the pair starting at i), sentinel at the end.
        var parts: [(start: Int, rank: Int)] = []
        parts.reserveCapacity(piece.count + 1)
        for i in 0 ..< piece.count {
            let rank = i + 1 < piece.count ? (ranks[Array(piece[i ... i + 1])] ?? Int.max) : Int.max
            parts.append((i, rank))
        }
        parts.append((piece.count, Int.max))

        func pairRank(_ index: Int) -> Int {
            guard index + 2 < parts.count else { return Int.max }
            return ranks[Array(piece[parts[index].start ..< parts[index + 2].start])] ?? Int.max
        }

        while parts.count > 2 {
            var best = Int.max
            var bestIndex = -1
            for i in 0 ..< parts.count - 1 where parts[i].rank < best {
                best = parts[i].rank
                bestIndex = i
            }
            if bestIndex < 0 { break }
            // Merge bestIndex with its successor; refresh the affected ranks.
            parts.remove(at: bestIndex + 1)
            parts[bestIndex].rank = pairRank(bestIndex)
            if bestIndex > 0 { parts[bestIndex - 1].rank = pairRank(bestIndex - 1) }
        }
        var out: [Int] = []
        out.reserveCapacity(parts.count - 1)
        for i in 0 ..< parts.count - 1 {
            out.append(ranks[Array(piece[parts[i].start ..< parts[i + 1].start])]!)
        }
        return out
    }
}
