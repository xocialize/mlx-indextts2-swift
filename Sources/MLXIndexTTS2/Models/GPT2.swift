// GPT2.swift — GPT-2 backbone for IndexTTS2.
//
// Isomorphic port of mlx_indextts/models/gpt2.py (same classes, same call order, same
// naming so parameter key paths match gpt.safetensors: gpt.h.N.{ln_1,attn.c_attn,attn.c_proj,
// ln_2,mlp.c_fc,mlp.c_proj}, gpt.ln_f). Deviations from upstream: none.

import Foundation
import MLX
import MLXNN

/// GPT-2 style multi-head attention (combined QKV, HF-style bias).
public final class GPT2Attention: Module {
    public let dim: Int
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    public init(dim: Int, numHeads: Int) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._cAttn.wrappedValue = Linear(dim, 3 * dim)
        self._cProj.wrappedValue = Linear(dim, dim)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (batchSize, _, _) = (x.dim(0), x.dim(1), x.dim(2))

        let qkv = cAttn(x)
        let parts = split(qkv, parts: 3, axis: -1)
        var q = parts[0], k = parts[1], v = parts[2]

        if let (kCache, vCache) = cache {
            k = concatenated([kCache, k], axis: 1)
            v = concatenated([vCache, v], axis: 1)
        }
        let newCache = (k, v)

        q = q.reshaped(batchSize, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batchSize, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batchSize, -1, numHeads, headDim).transposed(0, 2, 1, 3)

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let mask { scores = scores + mask }
        let attn = softmax(scores, axis: -1)

        var out = matmul(attn, v)
        out = out.transposed(0, 2, 1, 3).reshaped(batchSize, -1, dim)
        return (cProj(out), newCache)
    }
}

/// GPT-2 style MLP (GELU tanh approximation, like upstream `nn.gelu_approx`).
public final class GPT2MLP: Module {
    @ModuleInfo(key: "c_fc") var cFC: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    public init(dim: Int, hiddenDim: Int? = nil) {
        let hidden = hiddenDim ?? dim * 4
        self._cFC.wrappedValue = Linear(dim, hidden)
        self._cProj.wrappedValue = Linear(hidden, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        cProj(geluApproximate(cFC(x)))
    }
}

/// GPT-2 transformer block.
public final class GPT2Block: Module {
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "attn") var attn: GPT2Attention
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: GPT2MLP

    public init(dim: Int, numHeads: Int) {
        self._ln1.wrappedValue = LayerNorm(dimensions: dim)
        self._attn.wrappedValue = GPT2Attention(dim: dim, numHeads: numHeads)
        self._ln2.wrappedValue = LayerNorm(dimensions: dim)
        self._mlp.wrappedValue = GPT2MLP(dim: dim)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        var h = x
        let (attnOut, newCache) = attn(ln1(h), mask: mask, cache: cache)
        h = h + attnOut
        h = h + mlp(ln2(h))
        return (h, newCache)
    }
}

/// GPT-2 backbone (no embeddings / heads — UnifiedVoice owns those).
public final class GPT2Model: Module {
    public let dim: Int
    public let numHeads: Int
    public let numLayers: Int

    @ModuleInfo(key: "h") var h: [GPT2Block]
    @ModuleInfo(key: "ln_f") var lnF: LayerNorm

    public init(dim: Int, numHeads: Int, numLayers: Int) {
        self.dim = dim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self._h.wrappedValue = (0..<numLayers).map { _ in GPT2Block(dim: dim, numHeads: numHeads) }
        self._lnF.wrappedValue = LayerNorm(dimensions: dim)
    }

    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        mask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var x = inputsEmbeds
        let queryLen = x.dim(1)
        var newCache: [(MLXArray, MLXArray)] = []

        let keyLen: Int
        if let cache, !cache.isEmpty {
            keyLen = cache[0].0.dim(1) + queryLen
        } else {
            keyLen = queryLen
        }

        let effectiveMask = mask ?? Self.createCausalMask(queryLen: queryLen, keyLen: keyLen)

        for (i, block) in h.enumerated() {
            let layerCache = cache?[i]
            let (out, updated) = block(x, mask: effectiveMask, cache: layerCache)
            x = out
            newCache.append(updated)
        }
        return (lnF(x), newCache)
    }

    /// Additive causal mask (1,1,Q,K); disallowed positions are -inf.
    public static func createCausalMask(queryLen: Int, keyLen: Int) -> MLXArray {
        let mask: MLXArray
        if queryLen == keyLen {
            mask = triu(MLXArray.ones([queryLen, keyLen]), k: 1)
        } else {
            mask = MLXArray.zeros([queryLen, keyLen])
        }
        let additive = MLX.where(mask .> 0, MLXArray(-Float.infinity), MLXArray(Float(0)))
        return additive[.newAxis, .newAxis, 0..., 0...]
    }
}
