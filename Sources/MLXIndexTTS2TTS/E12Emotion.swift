// E12Emotion.swift — the shared E12 categorical emotion vocabulary (AB-A-0049 part 3).
//
// The contract carries the plane (`TTSEmotion.categorical` is an OPEN String, contract 1.38.0);
// the audio packages own the vocabulary. This file is IDENTICAL in mlx-indextts2-swift,
// mlx-qwen3-tts-swift and mlx-voxcpm2-tts-swift — the source of truth is
// mlxengine-audio/Docs/ENHANCEMENTS.md § E12 "Shared vocabulary", and a change here is a fleet
// change, not a package change.

import Foundation

/// The eight IndexTTS-2 emotion categories every E12 adopter in the fleet speaks (this is also
/// ML[X] Audio Studio's `DubEmotion`), plus the emotion2vec 9-way annotation labels folded onto
/// them — the 9→8 map from E12's design note: `fearful` → `afraid`, `neutral` / `other` /
/// `unknown` → `calm`. A consumer that wants NO steering sends no `emotion` at all.
public enum E12Emotion: String, CaseIterable, Sendable {
    case happy, angry, sad, afraid, disgusted, melancholic, surprised, calm

    /// Labels accepted besides the canonical names (case-insensitive).
    public static let aliases: [String: E12Emotion] = [
        "fearful": .afraid, "fear": .afraid,
        "neutral": .calm, "other": .calm, "unknown": .calm,
    ]

    /// Resolve an open categorical label — trimmed, case-insensitive, canonical or alias.
    /// `nil` = outside the vocabulary; the caller refuses legibly rather than guessing.
    public static func resolve(_ label: String) -> E12Emotion? {
        let key = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return E12Emotion(rawValue: key) ?? aliases[key]
    }

    /// For refusal messages: the canonical names, then the aliases.
    public static var knownLabels: String {
        allCases.map(\.rawValue).joined(separator: ", ")
            + " (aliases: " + aliases.keys.sorted().joined(separator: ", ") + ")"
    }
}
