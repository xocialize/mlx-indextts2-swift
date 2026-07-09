// Vq2Emb.swift — mel codes → semantic embeddings (generation-time vq2emb).
//
// Isomorphic to donor generate_v2 `_init_vq2emb`/`_vq2emb_forward`: raw codebook lookup
// (8192×8) + 1×1 out-projection, weights from vq2emb.safetensors (the checkpoint ships this
// tiny pre-fused extract of the MaskGCT quantizer so generation never needs the full codec).

import Foundation
import MLX
import MLXNN

public final class Vq2Emb: Module {
    @ModuleInfo(key: "codebook") var codebook: Embedding
    @ParameterInfo(key: "out_project_w") var outProjectW: MLXArray  // (1024, 8)
    @ParameterInfo(key: "out_project_b") var outProjectB: MLXArray  // (1024,)

    public init(codebookSize: Int = 8192, codebookDim: Int = 8, outDim: Int = 1024) {
        self._codebook.wrappedValue = Embedding(
            embeddingCount: codebookSize, dimensions: codebookDim)
        self._outProjectW.wrappedValue = MLXArray.zeros([outDim, codebookDim])
        self._outProjectB.wrappedValue = MLXArray.zeros([outDim])
    }

    /// codes: (B, T) int → (B, 1024, T) NCL (matches the donor's Conv1d-output convention).
    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let emb = codebook(codes)  // (B, T, 8)
        let out = matmul(emb, outProjectW.transposed()) + outProjectB  // (B, T, 1024)
        return out.transposed(0, 2, 1)
    }

    /// vq2emb.safetensors: codebook.weight (8192,8), out_project.weight (1024,8,1) → 2D,
    /// out_project.bias (1024,).
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            switch k {
            case "out_project.weight": out["out_project_w"] = v.squeezed(axis: 2)
            case "out_project.bias": out["out_project_b"] = v
            default: out[k] = v
            }
        }
        return out
    }
}
