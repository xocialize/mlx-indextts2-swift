// Perceiver.swift — Perceiver Resampler for IndexTTS2 conditioning.
//
// Isomorphic port of mlx_indextts/models/perceiver.py (solar2ain, MIT). Latent queries
// cross-attend to [latents; context]; 2 layers of (attention, GEGLU feed-forward); final
// RMSNorm (eps=1e-8, weight renamed from `gamma` by the converter).
//
// The checkpoint stores torch ModuleList-of-pairs paths `layers.N.0.*` (attention) and
// `layers.N.1.*` (feed-forward); `UnifiedVoiceV2.sanitize` remaps those to
// `layers.N.attn.*` / `layers.N.ff.*` (numeric module keys collide with array-index
// unflattening — same fix as CampPlus shortcut).

import Foundation
import MLX
import MLXNN

/// Feed-forward with GEGLU: w_1 → (x · GELU(gate)) → w_2. inner = ⌊dim·mult·2/3⌋.
public final class PerceiverFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    public init(dim: Int, mult: Int) {
        let innerDim = dim * mult * 2 / 3
        self._w1.wrappedValue = Linear(dim, innerDim * 2)
        self._w2.wrappedValue = Linear(innerDim, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(w1(x), parts: 2, axis: -1)
        return w2(parts[0] * gelu(parts[1]))
    }
}

/// Cross-attention where latents attend to [latents; context]. All projections bias-free.
public final class PerceiverAttention: Module {
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    public init(dim: Int, numHeads: Int = 8, headDim: Int = 64) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        let innerDim = numHeads * headDim
        self._linearQ.wrappedValue = Linear(dim, innerDim, bias: false)
        self._linearK.wrappedValue = Linear(dim, innerDim, bias: false)
        self._linearV.wrappedValue = Linear(dim, innerDim, bias: false)
        self._linearOut.wrappedValue = Linear(innerDim, dim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let (batchSize, nLatents, _) = (x.dim(0), x.dim(1), x.dim(2))

        // latents attend to both themselves and the context
        let kvInput = concatenated([x, context], axis: 1)

        var q = linearQ(x).reshaped(batchSize, nLatents, numHeads, headDim)
        var k = linearK(kvInput).reshaped(batchSize, -1, numHeads, headDim)
        var v = linearV(kvInput).reshaped(batchSize, -1, numHeads, headDim)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let attn = softmax(scores, axis: -1)
        var out = matmul(attn, v)
        out = out.transposed(0, 2, 1, 3).reshaped(batchSize, nLatents, -1)
        return linearOut(out)
    }
}

/// One (attention, feed-forward) pair (torch ModuleList indices 0/1 → keys attn/ff).
public final class PerceiverLayer: Module {
    @ModuleInfo(key: "attn") var attn: PerceiverAttention
    @ModuleInfo(key: "ff") var ff: PerceiverFeedForward

    public init(dim: Int, numHeads: Int, headDim: Int, ffMult: Int) {
        self._attn.wrappedValue = PerceiverAttention(dim: dim, numHeads: numHeads, headDim: headDim)
        self._ff.wrappedValue = PerceiverFeedForward(dim: dim, mult: ffMult)
    }
}

public final class PerceiverResampler: Module {
    @ModuleInfo(key: "proj_context") var projContext: Linear?
    @ParameterInfo(key: "latents") var latents: MLXArray
    @ModuleInfo(key: "layers") var layers: [PerceiverLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(dim: Int, nDimContext: Int? = nil, nLatents: Int = 32, nHeads: Int = 8,
                nHeadDim: Int = 64, nFFMult: Int = 4, nLayers: Int = 2) {
        let contextDim = nDimContext ?? dim
        if contextDim != dim {
            self._projContext.wrappedValue = Linear(contextDim, dim)
        }
        self._latents.wrappedValue = MLXArray.zeros([nLatents, dim])
        self._layers.wrappedValue = (0 ..< nLayers).map { _ in
            PerceiverLayer(dim: dim, numHeads: nHeads, headDim: nHeadDim, ffMult: nFFMult)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-8)
    }

    /// context: (B, seqLen, contextDim) → latent features (B, nLatents, dim).
    public func callAsFunction(_ context: MLXArray) -> MLXArray {
        var context = context
        if let projContext {
            context = projContext(context)
        }

        var lat = broadcast(
            latents[.newAxis, 0..., 0...],
            to: [context.dim(0), latents.dim(0), latents.dim(1)])

        for layer in layers {
            lat = lat + layer.attn(lat, context: context)
            lat = lat + layer.ff(lat)
        }
        return norm(lat)
    }
}
