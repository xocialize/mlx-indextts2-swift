// TextFrontendV25.swift — the IndexTTS-2.5 text pipeline (`infer_v2_5.py infer` text block +
// `split_text_by_tokens` + `apply_pronunciation_annotations`), operation-for-operation:
//
//   language (explicit or detected) → CHAR_REP_MAP clean → language normalization
//   (zh/en: TextNormalizer; es: NeMo TN — not portable, passthrough like the upstream fallback;
//   ja/ar: none) → case rule (zh/en/ja lower, es UPPER) → `<word|pron>` markup → `<|xx|>`
//   uppercase → token-budget segmentation (annotation spans atomic) → per segment:
//   tiktoken(`<|lang|> ` + segment) + stop id 1.
//
// Deliberate, documented deviations (the parity target is the oracle as run):
// - Japanese word spacing: upstream runs MeCab (fugashi + unidic-lite) at g2p_ratio 0 — it
//   converts no kanji, it only re-joins the morphemes with single spaces (punctuation included).
//   MeCab is not portable; this port segments with Apple's NaturalLanguage word tokenizer
//   instead, which reproduces MeCab's boundaries on the fixture corpus but is not guaranteed
//   identical on every sentence. Special-token spans are protected from segmentation (upstream
//   would shred `<|…|>` markup here; that is an upstream defect, not a behavior to mirror).
// - Spanish NeMo text normalization (numbers/dates) — upstream passes raw text through when
//   NeMo is absent, and so does this port.
// - zh/en WeTextProcessing number expansion (digits → words) is not ported; digit-bearing
//   fixtures are the documented gap (see Normalize.swift header).

import Foundation
import NaturalLanguage

/// The five languages the released 2.5 checkpoint is trained on, with the model's own ids
/// (`lang_to_token` over the 106-entry Whisper table).
public enum IndexTTSLanguage: String, CaseIterable, Sendable {
    case zh, en, ja, es, ar

    public var languageID: Int { TiktokenBPE.languageID(rawValue) }

    /// Accepts codes and common names (`"english"`, `"mandarin"`, `"zh-cn"`, …).
    public init?(parsing value: String) {
        let key = value.trimmingCharacters(in: .whitespaces).lowercased()
        let aliases: [String: IndexTTSLanguage] = [
            "chinese": .zh, "mandarin": .zh, "cn": .zh, "zh-cn": .zh, "zhen": .zh,
            "english": .en, "japanese": .ja, "spanish": .es, "castilian": .es, "arabic": .ar,
        ]
        if let direct = IndexTTSLanguage(rawValue: key) { self = direct }
        else if let alias = aliases[key] { self = alias }
        else { return nil }
    }

    /// Script-based detection (the donor's `resolve_v25_language`): Arabic → ar, kana → ja,
    /// Han → zh, Spanish diacritics → es, other Latin → en. Nil when nothing is recognizable.
    public static func detect(_ text: String) -> IndexTTSLanguage? {
        var arabic = false, kana = false, han = false, spanish = false, latin = false
        for s in text.unicodeScalars {
            switch s.value {
            case 0x0600...0x06FF, 0x0750...0x077F, 0x08A0...0x08FF: arabic = true
            case 0x3040...0x309F, 0x30A0...0x30FF: kana = true
            case 0x3400...0x4DBF, 0x4E00...0x9FFF: han = true
            case 0x41...0x5A, 0x61...0x7A: latin = true
            default:
                if "áéíóúüñ¿¡ÁÉÍÓÚÜÑ".unicodeScalars.contains(s) { spanish = true }
            }
        }
        if arabic { return .ar }
        if kana { return .ja }
        if han { return .zh }
        if spanish { return .es }
        if latin { return .en }
        return nil
    }
}

/// One prepared utterance: the resolved language and the per-segment token ids
/// (each already `<|lang|> `-prefixed and stop-padded, exactly what the GPT consumes).
public struct PreparedText: Sendable {
    public let language: IndexTTSLanguage
    public let normalized: String
    public let segments: [String]
    public let tokenIDs: [[Int]]
}

public final class IndexTTSTextFrontend: @unchecked Sendable {

    public let tokenizer: TiktokenBPE
    public let normalizer: IndexTTSTextNormalizer
    /// `text_pos_embedding` rows (602): the GPT's text position budget.
    public let textPositionCapacity: Int
    public static let stopTextToken = 1

    public init(tokenizer: TiktokenBPE, normalizer: IndexTTSTextNormalizer = IndexTTSTextNormalizer(),
                textPositionCapacity: Int = 602) {
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.textPositionCapacity = textPositionCapacity
    }

    private static let pronunciation = try! NSRegularExpression(pattern: #"<([^|>\n]+)\|([^>\n]+)>"#)
    private static let protected = try! NSRegularExpression(pattern: #"<\|SPECIAL_TOKEN_(\d+)\|>.*?<\|SPECIAL_TOKEN_\1\|>"#)
    private static let specialToken = try! NSRegularExpression(pattern: #"<\|([^|]+)\|>"#)
    private static let hanRE = try! NSRegularExpression(pattern: "[一-鿿]")   // 一-鿿
    private static let punctuationSplit = try! NSRegularExpression(pattern: #"(?<=[，。！？、；：,\.!?;:\n])"#)

    /// Full pipeline. `language` nil = detect from the text (Latin script defaults to English,
    /// which is ambiguous for Spanish — pass it explicitly).
    public func prepare(_ text: String, language: IndexTTSLanguage? = nil,
                        maxTokensPerSegment: Int = 120, normalize: Bool = true) throws -> PreparedText {
        guard let lang = language ?? IndexTTSLanguage.detect(text) else {
            throw FrontendError.undetectableLanguage
        }
        var t = normalizer.applyBaseCharRepMap(text)
        if normalize, lang == .zh || lang == .en { t = normalizer.normalize(t) }
        switch lang {
        case .zh, .en, .ja: t = t.lowercased()
        case .es: t = t.uppercased()
        case .ar: break
        }
        t = Self.applyPronunciationAnnotations(t)
        if lang == .ja { t = Self.spaceJapanese(t) }
        t = Self.uppercaseSpecialTokens(t)
        let prefix = "<|\(lang.rawValue)|> "
        let segments = splitByTokens(t, maxTokens: maxTokensPerSegment, languagePrefix: prefix)
        let ids = segments.map { tokenizer.encode(prefix + $0) + [Self.stopTextToken] }
        return PreparedText(language: lang, normalized: t, segments: segments, tokenIDs: ids)
    }

    public enum FrontendError: Error, CustomStringConvertible {
        case undetectableLanguage
        public var description: String {
            "cannot detect the text language — pass metaData.language (zh | en | ja | es | ar)"
        }
    }

    // MARK: - Steps

    /// `<word|pron>` → `<|SPECIAL_TOKEN_1|>PRON<|SPECIAL_TOKEN_1|>` (Han word → token 2);
    /// a kana pronunciation is inserted bare, space-padded.
    static func applyPronunciationAnnotations(_ text: String) -> String {
        let ns = text as NSString
        var out = ""
        var last = 0
        for m in pronunciation.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            let word = ns.substring(with: m.range(at: 1))
            let pron = ns.substring(with: m.range(at: 2)).uppercased()
            if isKana(pron) {
                out += " \(pron) "
            } else {
                let hasHan = hanRE.firstMatch(in: word, range: NSRange(location: 0, length: (word as NSString).length)) != nil
                let marker = hasHan ? "SPECIAL_TOKEN_2" : "SPECIAL_TOKEN_1"
                out += "<|\(marker)|>\(pron)<|\(marker)|>"
            }
            last = m.range.location + m.range.length
        }
        out += ns.substring(from: last)
        return out
    }

    /// MeCab-style word spacing for Japanese (`JapaneseG2PProcessor.process` at ratio 0): every
    /// non-space span is re-joined from its word tokens with single spaces, punctuation as its
    /// own token; existing spaces are preserved; `<|…|>` spans pass through untouched.
    static func spaceJapanese(_ text: String) -> String {
        let ns = text as NSString
        var out = ""
        var last = 0
        func emitSpan(_ span: String) {
            // Split on runs of spaces, segment each non-space run, keep the space runs verbatim.
            var buffer = ""
            var spaces = ""
            func flush() {
                if !buffer.isEmpty { out += segmentJapaneseWords(buffer); buffer = "" }
                out += spaces; spaces = ""
            }
            for ch in span {
                if ch == " " { if !buffer.isEmpty { flush() }; spaces.append(ch) }
                else { if !spaces.isEmpty { flush() }; buffer.append(ch) }
            }
            flush()
        }
        for m in specialToken.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            emitSpan(ns.substring(with: NSRange(location: last, length: m.range.location - last)))
            out += ns.substring(with: m.range)
            last = m.range.location + m.range.length
        }
        emitSpan(ns.substring(from: last))
        return out
    }

    private static func segmentJapaneseWords(_ text: String) -> String {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.setLanguage(.japanese)
        tokenizer.string = text
        var tokens: [String] = []
        var cursor = text.startIndex
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            for ch in text[cursor..<range.lowerBound] where !ch.isWhitespace { tokens.append(String(ch)) }
            tokens.append(String(text[range]))
            cursor = range.upperBound
            return true
        }
        for ch in text[cursor...] where !ch.isWhitespace { tokens.append(String(ch)) }
        return tokens.joined(separator: " ")
    }

    static func isKana(_ s: String) -> Bool {
        !s.isEmpty && s.unicodeScalars.allSatisfy { (0x3040...0x309F).contains($0.value) || (0x30A0...0x30FF).contains($0.value) }
    }

    /// `<|xx|>` → `<|XX|>` (the lowercase rule above would otherwise break the specials).
    static func uppercaseSpecialTokens(_ text: String) -> String {
        let ns = text as NSString
        var out = ""
        var last = 0
        for m in specialToken.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            out += "<|\(ns.substring(with: m.range(at: 1)).uppercased())|>"
            last = m.range.location + m.range.length
        }
        out += ns.substring(from: last)
        return out
    }

    /// `split_text_by_tokens`: budget = min(maxTokens, capacity − 2) − len(prefix); annotation
    /// spans are atomic; sentence punctuation splits first, then characters; chunks pack greedily.
    func splitByTokens(_ text: String, maxTokens: Int, languagePrefix: String) -> [String] {
        let budget = max(1, min(maxTokens, textPositionCapacity - 2) - tokenizer.tokenCount(languagePrefix))
        if tokenizer.tokenCount(text) <= budget { return [text] }

        var chunks: [String] = []
        for (piece, atomic) in atomicPieces(text) {
            if atomic { chunks.append(piece); continue }
            for part in Self.splitAfterPunctuation(piece) where !part.isEmpty {
                if tokenizer.tokenCount(part) <= budget { chunks.append(part); continue }
                var current = ""
                for ch in part {
                    if !current.isEmpty, tokenizer.tokenCount(current + String(ch)) > budget {
                        chunks.append(current)
                        current = String(ch)
                    } else {
                        current += String(ch)
                    }
                }
                if !current.isEmpty { chunks.append(current) }
            }
        }
        var segments: [String] = []
        var current = ""
        for chunk in chunks {
            if !current.isEmpty, tokenizer.tokenCount(current + chunk) > budget {
                segments.append(current)
                current = chunk
            } else {
                current += chunk
            }
        }
        if !current.isEmpty { segments.append(current) }
        return segments.isEmpty ? [text] : segments
    }

    private func atomicPieces(_ text: String) -> [(String, Bool)] {
        let ns = text as NSString
        var pieces: [(String, Bool)] = []
        var pos = 0
        for m in Self.protected.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            if m.range.location > pos {
                pieces.append((ns.substring(with: NSRange(location: pos, length: m.range.location - pos)), false))
            }
            pieces.append((ns.substring(with: m.range), true))
            pos = m.range.location + m.range.length
        }
        if pos < ns.length { pieces.append((ns.substring(from: pos), false)) }
        return pieces
    }

    /// `re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', piece)` — split AFTER each punctuation mark.
    static func splitAfterPunctuation(_ text: String) -> [String] {
        let ns = text as NSString
        var parts: [String] = []
        var last = 0
        for m in punctuationSplit.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let cut = m.range.location
            if cut > last || cut == last { parts.append(ns.substring(with: NSRange(location: last, length: cut - last))) }
            last = cut
        }
        parts.append(ns.substring(from: last))
        return parts
    }
}
