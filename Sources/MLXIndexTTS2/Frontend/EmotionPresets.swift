// EmotionPresets.swift — the preset-emotion path (generate_v2 `_compute_emotion_vector`).
//
// The checkpoint ships two small matrices (feat1.pt spk_matrix 73×192, feat2.pt emo_matrix
// 73×1280) holding 73 per-speaker emotion vectors grouped into 8 categories
// (EMO_NUM = [3,17,2,8,4,5,10,24]). The preset path picks, per category, the speaker whose
// spk_matrix row is most cosine-similar to the reference CampPlus style, gathers the matching
// emo_matrix rows, and weight-sums them: emovec_mat = Σ w_i · emo_vectors[i]  (1, 1280).
// The caller then blends: emo_vec = emovec_mat + (1 − Σw)·base_emovec (Σw ≥ 1 ⇒ emovec_mat).
//
// Both matrices are BAKED resources dumped from the checkpoint's feat1/feat2.pt
// (tools/dump_stage2.py; torch pickles — unreadable from Swift). They are CHECKPOINT DATA:
// covered by the IndexTTS2 weight license (INDEX_MODEL_LICENSE, NonCommercial), not this
// repo's Apache port-code license; they move into the weight repo at the own-conversion
// re-publish. Gated bitwise-classed against `frontend_emovec_mat` in `indextts2-gate emovec`
// (the oracle recompute from these dumps is bitwise-identical to the Stage-0 golden).

import Foundation
import MLX

/// The 8-category preset-emotion vector head.
public enum EmotionPresets {
    /// IndexTTS 2.0 category order (weight-vector order).
    public static let categories = [
        "happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm",
    ]
    /// Vectors per category in the 73-row matrices.
    static let emoNum = [3, 17, 2, 8, 4, 5, 10, 24]

    private static func loadNPY(_ name: String) -> MLXArray {
        guard let url = Bundle.module.url(forResource: name, withExtension: "npy",
                                          subdirectory: "Resources"),
              let array = try? NPY.load(url)
        else { fatalError("missing baked resource \(name).npy") }
        return array
    }

    /// (73, 1280) — per-speaker emotion vectors (feat2.pt).
    static let emoMatrix = loadNPY("emo_matrix")
    /// (73, 192) — per-speaker style anchors (feat1.pt).
    static let spkMatrix = loadNPY("spk_matrix")

    /// Weighted preset-emotion vector: `weights` in category order (already emo_alpha-scaled),
    /// `style` = the reference CampPlus embedding (1, 192). Returns (1, 1280).
    public static func emovecMat(weights: [Float], style: MLXArray) -> MLXArray {
        precondition(weights.count == categories.count, "want 8 category weights")
        let styleVec = style.reshaped(-1).asType(.float32)               // (192,)
        let styleNorm = sqrt(sum(styleVec * styleVec))

        var offset = 0
        var rows: [MLXArray] = []
        for n in emoNum {
            let spkCat = spkMatrix[offset ..< (offset + n)]              // (n, 192)
            let sims = matmul(spkCat, styleVec.reshaped(192, 1)).reshaped(-1)
                / (sqrt(sum(spkCat * spkCat, axis: 1)) * styleNorm)      // cosine per row
            let idx = argMax(sims).item(Int.self)
            rows.append(emoMatrix[offset + idx])                         // (1280,)
            offset += n
        }
        let emoVectors = stacked(rows)                                    // (8, 1280)
        let w = MLXArray(weights).reshaped(8, 1)
        return (w * emoVectors).sum(axis: 0)[.newAxis, 0...]              // (1, 1280)
    }

    /// The full blend used by generation: preset weights → emo_vec.
    /// `emovec_mat + (1 − Σw)·base` when Σw < 1, else `emovec_mat` (generate_v2 exact).
    public static func blend(weights: [Float], style: MLXArray, baseEmovec: MLXArray) -> MLXArray {
        let mat = emovecMat(weights: weights, style: style)
        let weightSum = weights.reduce(0, +)
        return weightSum >= 1.0 ? mat : mat + (1.0 - weightSum) * baseEmovec
    }
}
