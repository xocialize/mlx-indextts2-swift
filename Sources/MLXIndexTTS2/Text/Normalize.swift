// Normalize.swift — text normalization for IndexTTS2.
//
// Bit-faithful port of solar2ain mlx_indextts/normalize.py (P1). Parity target is the
// ORACLE AS RUN, gated by Tests/Resources/tokenizer_fixtures.json:
// - WeTextProcessing (number/date expansion) was a no-op in the oracle install, so this port
//   deliberately has no equivalent — digits flow through and tokenize to <unk>, exactly like
//   the goldens. (Known upstream quality gap; pre-normalize numbers upstream if needed.)
// - CHAR_REP_MAP is reproduced EXACTLY as effective in Python — including the source bug where
//   the smart-quote entries collapsed into one garbage multi-char key (so “ ” are NOT mapped).
//   Order matters: replacement is a single left-to-right pass, first-matching key in map order.

import Foundation

// MARK: - Emoji / CJK character classes (ranges mirror normalize.py)

private let emojiRanges: [(UInt32, UInt32)] = [
    (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF), (0x1F1E0, 0x1F1FF),
    (0x2600, 0x26FF), (0x2700, 0x27BF), (0x1F900, 0x1F9FF), (0x1FA00, 0x1FAFF),
    (0xFE00, 0xFE0F), (0x200D, 0x200D),
]

private let cjkRanges: [(UInt32, UInt32)] = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
]

/// Replace emoji scalars with a space (mirrors `remove_emoji`).
public func removeEmoji(_ text: String) -> String {
    String(String.UnicodeScalarView(text.unicodeScalars.map { s in
        emojiRanges.contains(where: { $0.0 <= s.value && s.value <= $0.1 }) ? " " : s
    }))
}

public func isCJKScalar(_ s: Unicode.Scalar) -> Bool {
    cjkRanges.contains { $0.0 <= s.value && s.value <= $0.1 }
}

/// Add spaces around CJK characters, collapse whitespace, strip, uppercase
/// (mirrors `tokenize_by_cjk_char`, `do_upper_case=True`).
public func tokenizeByCJKChar(_ text: String, doUpperCase: Bool = true) -> String {
    var out = String.UnicodeScalarView()
    for s in text.unicodeScalars {
        if isCJKScalar(s) {
            out.append(" ")
            out.append(s)
            out.append(" ")
        } else {
            out.append(s)
        }
    }
    // \s+ -> " " then strip (Python re \s over str ≈ Unicode whitespace)
    var collapsed = ""
    var inWS = false
    for ch in String(out) {
        if ch.isWhitespace {
            if !inWS { collapsed.append(" ") }
            inWS = true
        } else {
            collapsed.append(ch)
            inWS = false
        }
    }
    let stripped = collapsed.trimmingCharacters(in: .whitespaces)
    return doUpperCase ? stripped.uppercased() : stripped
}

// MARK: - TextNormalizer

/// Port of `TextNormalizer` (en/zh branch, protections, char-rep map). The zh/en text
/// normalizers (WeTextProcessing) are intentionally absent — see header note.
public final class IndexTTSTextNormalizer: @unchecked Sendable {

    /// Effective CHAR_REP_MAP, dumped from the running Python (order = dict order; includes the
    /// collapsed-smart-quote garbage key from the upstream source bug, position-faithful).
    static let charRepMap: [(String, String)] = [
        ("\u{FF1A}", ","), ("\u{FF1B}", ","), (";", ","), ("\u{FF0C}", ","),
        ("\u{3002}", "."), ("\u{FF01}", "!"), ("\u{FF1F}", "?"), ("\n", " "),
        ("\u{00B7}", "-"), ("\u{3001}", ","),
        ("...", "\u{2026}"), (",,,", "\u{2026}"), ("\u{FF0C}\u{FF0C}\u{FF0C}", "\u{2026}"),
        ("\u{2026}\u{2026}", "\u{2026}"),
        (": \u{0022}'\u{0022},\n        ", "'"),  // upstream source bug, kept verbatim
        ("\u{0022}", "'"), ("'", "'"),
        ("\u{FF08}", "'"), ("\u{FF09}", "'"), ("(", "'"), (")", "'"),
        ("\u{300A}", "'"), ("\u{300B}", "'"), ("\u{3010}", "'"), ("\u{3011}", "'"),
        ("[", "'"), ("]", "'"),
        ("\u{2014}", "-"), ("\u{FF5E}", "-"), ("~", "-"),
        ("\u{300C}", "'"), ("\u{300D}", "'"), (":", ","),
    ]
    /// ZH map = {"$": "."} FIRST, then charRepMap (mirrors `{"$": ".", **CHAR_REP_MAP}`).
    static let zhCharRepMap: [(String, String)] = [("$", ".")] + charRepMap

    private let contraction = try! NSRegularExpression(
        pattern: #"(what|where|who|which|how|t?here|it|s?he|that|this)'s"#,
        options: [.caseInsensitive])
    private let techTerm = try! NSRegularExpression(
        pattern: #"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"#)
    private let restoreHyphen = try! NSRegularExpression(pattern: #"\s*<H>\s*"#)
    private let pinyinTone = try! NSRegularExpression(
        pattern: #"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"#,
        options: [.caseInsensitive])
    private let namePattern = try! NSRegularExpression(
        pattern: #"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"#)
    private let hasChineseRe = try! NSRegularExpression(pattern: #"[\u4e00-\u9fff]"#)
    private let hasAlphaRe = try! NSRegularExpression(pattern: #"[a-zA-Z]"#)
    private let jqxFix = try! NSRegularExpression(
        pattern: #"([jqx])[uü](n|e|an)*(\d)"#, options: [.caseInsensitive])

    public init() {}

    // MARK: helpers

    private func fullRange(_ s: String) -> NSRange { NSRange(s.startIndex..., in: s) }

    private func matches(_ re: NSRegularExpression, _ s: String) -> Bool {
        re.firstMatch(in: s, range: fullRange(s)) != nil
    }

    /// All whole-match strings for a regex.
    private func findAll(_ re: NSRegularExpression, _ s: String) -> [String] {
        re.matches(in: s, range: fullRange(s)).compactMap {
            Range($0.range, in: s).map { String(s[$0]) }
        }
    }

    func useChinese(_ text: String) -> Bool {
        let hasChinese = matches(hasChineseRe, text)
        let hasAlpha = matches(hasAlphaRe, text)
        if hasChinese || !hasAlpha { return true }
        return matches(pinyinTone, text)
    }

    // MARK: protections (mirror _save/_restore_*)

    private func saveTechTerms(_ text: String) -> (String, [String]?) {
        let found = findAll(techTerm, text)
        guard !found.isEmpty else { return (text, nil) }
        // sorted(set, key=len, reverse=True); Python set order is arbitrary but replacement is
        // whole-term string replace, so length-desc is the only ordering that matters.
        let terms = Array(Set(found)).sorted { $0.count > $1.count }
        var t = text
        for term in terms {
            t = t.replacingOccurrences(of: term, with: term.replacingOccurrences(of: "-", with: "<H>"))
        }
        return (t, terms)
    }

    private func restoreTechTerms(_ text: String, _ terms: [String]?) -> String {
        guard terms != nil else { return text }
        return restoreHyphen.stringByReplacingMatches(
            in: text, range: fullRange(text), withTemplate: "-")
    }

    private func savePinyinTones(_ text: String) -> (String, [String]?) {
        let ms = pinyinTone.matches(in: text, range: fullRange(text))
        guard !ms.isEmpty else { return (text, nil) }
        var set = Set<String>()
        var list: [String] = []
        for m in ms {
            guard let r = Range(m.range, in: text) else { continue }
            let s = String(text[r])
            if set.insert(s).inserted { list.append(s) }
        }
        var t = text
        for (i, py) in list.enumerated() {
            let placeholder = "<pinyin_\(Character(UnicodeScalar(UInt8(97 + i))))>"
            t = t.replacingOccurrences(of: py, with: placeholder)
        }
        return (t, list)
    }

    private func restorePinyinTones(_ text: String, _ list: [String]?) -> String {
        guard let list else { return text }
        var t = text
        for (i, var py) in list.enumerated() {
            let placeholder = "<pinyin_\(Character(UnicodeScalar(UInt8(97 + i))))>"
            if let first = py.first, "jqx".contains(first.lowercased()) {
                py = jqxFix.stringByReplacingMatches(
                    in: py, range: fullRange(py), withTemplate: "$1v$2$3")
            }
            t = t.replacingOccurrences(of: placeholder, with: py.uppercased())
        }
        return t
    }

    private func saveNames(_ text: String) -> (String, [String]?) {
        let found = findAll(namePattern, text)
        guard !found.isEmpty else { return (text, nil) }
        var set = Set<String>()
        var list: [String] = []
        for f in found where set.insert(f).inserted { list.append(f) }
        var t = text
        for (i, name) in list.enumerated() {
            t = t.replacingOccurrences(
                of: name, with: "<n_\(Character(UnicodeScalar(UInt8(97 + i))))>")
        }
        return (t, list)
    }

    private func restoreNames(_ text: String, _ names: [String]?) -> String {
        guard let names else { return text }
        var t = text
        for (i, name) in names.enumerated() {
            t = t.replacingOccurrences(
                of: "<n_\(Character(UnicodeScalar(UInt8(97 + i))))>", with: name)
        }
        return t
    }

    /// Single left-to-right pass; at each position try keys in map order, first match wins
    /// (replicates Python `re.sub` over an alternation joined in dict order).
    private func applyCharRepMap(_ text: String, _ map: [(String, String)]) -> String {
        var out = ""
        var idx = text.startIndex
        outer: while idx < text.endIndex {
            for (key, value) in map {
                if let end = text.index(idx, offsetBy: key.count, limitedBy: text.endIndex),
                   text[idx..<end] == key {
                    out += value
                    idx = end
                    continue outer
                }
            }
            out.append(text[idx])
            idx = text.index(after: idx)
        }
        return out
    }

    // MARK: normalize

    public func normalize(_ input: String) -> String {
        guard !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return "" }
        var text = removeEmoji(input)
        text = contraction.stringByReplacingMatches(
            in: text, range: fullRange(text), withTemplate: "$1 is")

        if useChinese(text) {
            // rstrip() (Python str.rstrip strips trailing unicode whitespace)
            while let last = text.unicodeScalars.last,
                  CharacterSet.whitespacesAndNewlines.contains(last) {
                text.unicodeScalars.removeLast()
            }
            let (t1, tech) = saveTechTerms(text)
            let (t2, pinyin) = savePinyinTones(t1)
            let (t3, names) = saveNames(t2)
            // zh_normalizer: absent (oracle-as-run parity) — identity
            var t = restoreNames(t3, names)
            t = restorePinyinTones(t, pinyin)
            t = restoreTechTerms(t, tech)
            return applyCharRepMap(t, Self.zhCharRepMap)
        } else {
            let (t1, tech) = saveTechTerms(text)
            // en_normalizer: absent (oracle-as-run parity) — identity
            let t = restoreTechTerms(t1, tech)
            return applyCharRepMap(t, Self.charRepMap)
        }
    }
}
