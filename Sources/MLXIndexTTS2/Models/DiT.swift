// DiT.swift — Diffusion Transformer estimator for the S2Mel CFM.
//
// Isomorphic port of donor `mlx_indextts/models/s2mel/dit.py` (gpt_fast-style transformer
// with adaptive RMS layer norm, U-ViT skips, long skip, wavenet final layer).
//
// Numerics carried verbatim from the donor:
// - RoPE is PAIRED/interleaved layout (reshape (..., D/2, 2)), NOT the half-split — hand-rolled,
//   plain (non-Module) class exactly like the donor (freqs_cis is computed, not a checkpoint key).
// - Attention is hand-rolled softmax((qkᵀ)·scale + mask) — the donor does not use fast SDPA.
// - t_embedder.freqs IS a checkpoint tensor (fp16) that overwrote the donor's computed fp32
//   buffer at load — declared as a parameter here so it loads the same values.
// - FinalLayer's non-affine LayerNorm uses eps=1e-6 (donor hardcodes it).
// - Resolved config (checkpoint config.yaml): hidden 512, heads 8, depth 13, mel 80,
//   content 512, style 192, wavenet 512×8 k5 d1, long_skip + uvit_skip, style_condition,
//   NOT time/style-as-token. Donor keeps block_size=16384 (config says 8192; positions ≤ 621,
//   numerically irrelevant — donor wins).

import Foundation
import MLX
import MLXNN

/// Transformer config (donor ModelArgs, post-init resolved).
public struct DiTModelArgs {
    public var blockSize = 16384
    public var vocabSize = 1024
    public var nLayer = 13
    public var nHead = 8
    public var dim = 512
    public var intermediateSize: Int
    public var nLocalHeads: Int
    public var headDim: Int
    public var ropeBase: Float = 10000
    public var normEps: Float = 1e-5
    public var uvitSkipConnection = true
    public var timeAsToken = false

    public init(nLayer: Int = 13, nHead: Int = 8, dim: Int = 512,
                uvitSkipConnection: Bool = true, timeAsToken: Bool = false) {
        self.nLayer = nLayer
        self.nHead = nHead
        self.dim = dim
        self.uvitSkipConnection = uvitSkipConnection
        self.timeAsToken = timeAsToken
        self.nLocalHeads = nHead
        // donor __post_init__: 2/3 · 4·dim rounded up to a multiple of 256
        let hiddenDim = 4 * dim
        let nHidden = Int(2 * hiddenDim / 3)
        self.intermediateSize = ((nHidden + 255) / 256) * 256
        self.headDim = dim / nHead
    }
}

/// Sinusoidal timestep embedder + MLP (donor TimestepEmbedder).
public final class TimestepEmbedder: Module {
    public let hiddenSize: Int
    public let frequencyEmbeddingSize: Int
    let scale: Float = 1000

    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ParameterInfo(key: "freqs") var freqs: MLXArray  // checkpoint tensor (128,)

    public init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.hiddenSize = hiddenSize
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self._linear1.wrappedValue = Linear(frequencyEmbeddingSize, hiddenSize)
        self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize)
        // placeholder; overwritten by the checkpoint (donor computes fp32, ckpt carries fp16)
        self._freqs.wrappedValue = MLXArray.zeros([frequencyEmbeddingSize / 2])
    }

    func timestepEmbedding(_ t: MLXArray) -> MLXArray {
        let args = scale * t[0..., .newAxis] * freqs[.newAxis, 0...]  // (B, half)
        return concatenated([cos(args), sin(args)], axis: -1)
        // frequency_embedding_size is even for this checkpoint; no zero-pad branch
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        var tEmb = linear1(timestepEmbedding(t))
        tEmb = silu(tEmb)
        return linear2(tEmb)
    }
}

/// SiLU + Linear conditioning MLP; the Linear lives at checkpoint key
/// `adaLN_modulation.layers.1` (torch Sequential) — remapped to `linear` in sanitize.
public final class AdaLNModulation: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    public init(hiddenSize: Int) {
        self._linear.wrappedValue = Linear(hiddenSize, 2 * hiddenSize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(silu(x))
    }
}

/// Final layer with adaptive (non-affine, eps=1e-6) layer norm (donor FinalLayer).
public final class DiTFinalLayer: Module {
    public let hiddenSize: Int

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: AdaLNModulation

    public init(hiddenSize: Int, patchSize: Int, outChannels: Int) {
        self.hiddenSize = hiddenSize
        self._linear.wrappedValue = Linear(hiddenSize, patchSize * patchSize * outChannels)
        self._adaLNModulation.wrappedValue = AdaLNModulation(hiddenSize: hiddenSize)
    }

    func layerNorm(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        return (x - mean) / sqrt(variance + 1e-6)
    }

    public func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(c)  // (B, 2*hidden)
        let shift = modulation[0..., ..<hiddenSize]
        let scale = modulation[0..., hiddenSize...]
        var x = layerNorm(x)
        x = x * (1 + scale[0..., .newAxis, 0...]) + shift[0..., .newAxis, 0...]
        return linear(x)
    }
}

/// RMSNorm hand-rolled to match donor bit-order (sqrt then divide).
public final class DiTRMSNorm: Module {
    public let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    public init(dims: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dims])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rms = sqrt(mean(x * x, axis: -1, keepDims: true) + eps)
        return (x / rms) * weight
    }
}

/// Adaptive layer norm: weight·RMSNorm(x) + bias, weight/bias projected from conditioning.
public final class AdaptiveLayerNorm: Module {
    public let dModel: Int

    @ModuleInfo(key: "project_layer") var projectLayer: Linear
    @ModuleInfo(key: "norm") var norm: DiTRMSNorm

    public init(dModel: Int, eps: Float = 1e-5) {
        self.dModel = dModel
        self._projectLayer.wrappedValue = Linear(dModel, 2 * dModel)
        self._norm.wrappedValue = DiTRMSNorm(dims: dModel, eps: eps)
    }

    public func callAsFunction(_ x: MLXArray, embedding: MLXArray?) -> MLXArray {
        guard var embedding else { return norm(x) }
        if embedding.ndim == 2 { embedding = embedding[0..., .newAxis, 0...] }
        let proj = projectLayer(embedding)  // (B, 1, 2*d)
        let weight = proj[.ellipsis, ..<dModel]
        let bias = proj[.ellipsis, dModel...]
        return weight * norm(x) + bias
    }
}

/// Rotary embedding table. Plain class, NOT a Module — freqs_cis is computed, never loaded
/// (mirrors the donor, and keeps it out of the parameter contract).
public final class RotaryPositionEmbedding {
    public let freqsCis: MLXArray  // (max_seq, dim/2, 2) [cos, sin]

    public init(dim: Int, maxSeqLen: Int = 16384, base: Float = 10000) {
        let nElem = dim
        let exponents = MLXArray(stride(from: 0, to: nElem, by: 2).map { Float($0) }) / Float(nElem)
        let freqs = 1.0 / pow(MLXArray(base), exponents)  // (dim/2,)
        let t = MLXArray(0 ..< maxSeqLen).asType(.float32)
        let outerFreqs = t[0..., .newAxis] * freqs[.newAxis, 0...]  // (max_seq, dim/2)
        self.freqsCis = stacked([cos(outerFreqs), sin(outerFreqs)], axis: -1)
    }
}

/// Paired/interleaved rotary application (donor apply_rotary_emb).
/// x: (B, S, H, D), freqsCis: (S, D/2, 2).
func applyRotaryEmb(_ x: MLXArray, _ freqsCis: MLXArray) -> MLXArray {
    let xShape = x.shape
    let xr = x.reshaped(xShape[0], xShape[1], xShape[2], -1, 2)

    let cosF = freqsCis[.ellipsis, 0][.newAxis, 0..., .newAxis, 0...]  // (1, S, 1, D/2)
    let sinF = freqsCis[.ellipsis, 1][.newAxis, 0..., .newAxis, 0...]

    let xReal = xr[.ellipsis, 0]
    let xImag = xr[.ellipsis, 1]
    let outReal = xReal * cosF - xImag * sinF
    let outImag = xImag * cosF + xReal * sinF

    return stacked([outReal, outImag], axis: -1).reshaped(xShape)
}

/// Multi-head attention with rotary embeddings (donor Attention; hand-rolled, no fast SDPA).
public final class DiTAttention: Module {
    public let nHead: Int
    public let nLocalHeads: Int
    public let headDim: Int

    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    public init(_ config: DiTModelArgs) {
        self.nHead = config.nHead
        self.nLocalHeads = config.nLocalHeads
        self.headDim = config.headDim
        let totalHeadDim = (config.nHead + 2 * config.nLocalHeads) * config.headDim
        self._wqkv.wrappedValue = Linear(config.dim, totalHeadDim, bias: false)
        self._wo.wrappedValue = Linear(config.headDim * config.nHead, config.dim, bias: false)
    }

    public func callAsFunction(
        _ x: MLXArray, freqsCis: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let (bsz, seqlen) = (x.dim(0), x.dim(1))
        let kvSize = nLocalHeads * headDim

        let qkv = wqkv(x)
        var q = qkv[0..., 0..., ..<kvSize]
        var k = qkv[0..., 0..., kvSize ..< (2 * kvSize)]
        var v = qkv[0..., 0..., (2 * kvSize)...]

        q = q.reshaped(bsz, seqlen, nHead, headDim)
        k = k.reshaped(bsz, seqlen, nLocalHeads, headDim)
        v = v.reshaped(bsz, seqlen, nLocalHeads, headDim)

        q = applyRotaryEmb(q, freqsCis)
        k = applyRotaryEmb(k, freqsCis)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        if nHead != nLocalHeads {
            k = repeated(k, count: nHead / nLocalHeads, axis: 1)
            v = repeated(v, count: nHead / nLocalHeads, axis: 1)
        }

        let scale = 1.0 / Float(Double(headDim).squareRoot())
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let mask { scores = scores + mask }

        let attn = softmax(scores, axis: -1)
        var y = matmul(attn, v)
        y = y.transposed(0, 2, 1, 3).reshaped(bsz, seqlen, -1)
        return wo(y)
    }
}

/// SwiGLU feed-forward (donor FeedForward).
public final class DiTFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    public init(_ config: DiTModelArgs) {
        self._w1.wrappedValue = Linear(config.dim, config.intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(config.intermediateSize, config.dim, bias: false)
        self._w3.wrappedValue = Linear(config.dim, config.intermediateSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

/// Transformer block with adaptive layer norm + optional U-ViT skip-in (donor TransformerBlock).
public final class DiTTransformerBlock: Module {
    public let uvitSkipConnection: Bool
    public let timeAsToken: Bool

    @ModuleInfo(key: "attention") var attention: DiTAttention
    @ModuleInfo(key: "feed_forward") var feedForward: DiTFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: AdaptiveLayerNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: AdaptiveLayerNorm
    @ModuleInfo(key: "skip_in_linear") var skipInLinear: Linear?

    public init(_ config: DiTModelArgs) {
        self.uvitSkipConnection = config.uvitSkipConnection
        self.timeAsToken = config.timeAsToken
        self._attention.wrappedValue = DiTAttention(config)
        self._feedForward.wrappedValue = DiTFeedForward(config)
        self._attentionNorm.wrappedValue = AdaptiveLayerNorm(dModel: config.dim, eps: config.normEps)
        self._ffnNorm.wrappedValue = AdaptiveLayerNorm(dModel: config.dim, eps: config.normEps)
        self._skipInLinear.wrappedValue =
            config.uvitSkipConnection ? Linear(config.dim * 2, config.dim) : nil
    }

    public func callAsFunction(
        _ x: MLXArray, c: MLXArray, freqsCis: MLXArray, mask: MLXArray?,
        skipInX: MLXArray?
    ) -> MLXArray {
        let cUse: MLXArray? = timeAsToken ? nil : c

        var x = x
        if uvitSkipConnection, let skipInX, let skipInLinear {
            x = skipInLinear(concatenated([x, skipInX], axis: -1))
        }

        let h = x + attention(attentionNorm(x, embedding: cUse), freqsCis: freqsCis, mask: mask)
        return h + feedForward(ffnNorm(h, embedding: cUse))
    }
}

/// Transformer backbone (donor Transformer).
public final class DiTTransformer: Module {
    public let config: DiTModelArgs
    public let rope: RotaryPositionEmbedding
    let layersEmitSkip: [Int]
    let layersReceiveSkip: [Int]

    @ModuleInfo(key: "layers") var layers: [DiTTransformerBlock]
    @ModuleInfo(key: "norm") var norm: AdaptiveLayerNorm

    public init(_ config: DiTModelArgs) {
        self.config = config
        self._layers.wrappedValue = (0 ..< config.nLayer).map { _ in DiTTransformerBlock(config) }
        self._norm.wrappedValue = AdaptiveLayerNorm(dModel: config.dim, eps: config.normEps)
        self.rope = RotaryPositionEmbedding(
            dim: config.headDim, maxSeqLen: config.blockSize, base: config.ropeBase)
        if config.uvitSkipConnection {
            self.layersEmitSkip = (0 ..< config.nLayer).filter { $0 < config.nLayer / 2 }
            self.layersReceiveSkip = (0 ..< config.nLayer).filter { $0 > config.nLayer / 2 }
        } else {
            self.layersEmitSkip = []
            self.layersReceiveSkip = []
        }
    }

    public func callAsFunction(
        _ x: MLXArray, c: MLXArray, inputPos: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let freqsCis = rope.freqsCis[inputPos]

        var x = x
        var skipInXList: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            var skipInX: MLXArray? = nil
            if config.uvitSkipConnection && layersReceiveSkip.contains(i) {
                skipInX = skipInXList.removeLast()
            }
            x = layer(x, c: c, freqsCis: freqsCis, mask: mask, skipInX: skipInX)
            if config.uvitSkipConnection && layersEmitSkip.contains(i) {
                skipInXList.append(x)
            }
        }
        return norm(x, embedding: c)
    }
}

func ditSequenceMask(_ lengths: MLXArray, maxLength: Int) -> MLXArray {
    let seqRange = MLXArray(0 ..< maxLength)[.newAxis, 0...]
    return (seqRange .< lengths[0..., .newAxis]).asType(.float32)
}

/// Diffusion Transformer estimator (donor DiT, wavenet final-layer variant).
public final class DiT: Module {
    public let inChannels: Int
    public let hiddenDim: Int
    public let timeAsToken: Bool
    public let styleAsToken: Bool
    public let transformerStyleCondition: Bool
    public let longSkipConnection: Bool

    @ModuleInfo(key: "transformer") var transformer: DiTTransformer
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "cond_projection") var condProjection: Linear
    // present in the checkpoint but unused at inference — declared for the key contract
    @ModuleInfo(key: "cond_embedder") var condEmbedder: Embedding
    @ModuleInfo(key: "content_mask_embedder") var contentMaskEmbedder: Embedding
    @ModuleInfo(key: "t_embedder") var tEmbedder: TimestepEmbedder
    @ModuleInfo(key: "cond_x_merge_linear") var condXMergeLinear: Linear
    @ModuleInfo(key: "skip_linear") var skipLinear: Linear
    @ModuleInfo(key: "t_embedder2") var tEmbedder2: TimestepEmbedder
    @ModuleInfo(key: "conv1") var conv1: Linear
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "wavenet") var wavenet: WN
    @ModuleInfo(key: "final_layer") var finalLayer: DiTFinalLayer
    @ModuleInfo(key: "res_projection") var resProjection: Linear

    public init(
        hiddenDim: Int = 512, numHeads: Int = 8, depth: Int = 13, inChannels: Int = 80,
        contentDim: Int = 512, styleDim: Int = 192,
        longSkipConnection: Bool = true, uvitSkipConnection: Bool = true,
        timeAsToken: Bool = false, styleAsToken: Bool = false, styleCondition: Bool = true,
        wavenetHiddenDim: Int = 512, wavenetNumLayers: Int = 8,
        wavenetKernelSize: Int = 5, wavenetDilationRate: Int = 1
    ) {
        self.inChannels = inChannels
        self.hiddenDim = hiddenDim
        self.timeAsToken = timeAsToken
        self.styleAsToken = styleAsToken
        self.transformerStyleCondition = styleCondition
        self.longSkipConnection = longSkipConnection

        let config = DiTModelArgs(
            nLayer: depth, nHead: numHeads, dim: hiddenDim,
            uvitSkipConnection: uvitSkipConnection, timeAsToken: timeAsToken)
        self._transformer.wrappedValue = DiTTransformer(config)

        self._xEmbedder.wrappedValue = Linear(inChannels, hiddenDim)
        self._condProjection.wrappedValue = Linear(contentDim, hiddenDim)
        self._condEmbedder.wrappedValue = Embedding(embeddingCount: 1024, dimensions: hiddenDim)
        self._contentMaskEmbedder.wrappedValue = Embedding(embeddingCount: 1, dimensions: hiddenDim)
        self._tEmbedder.wrappedValue = TimestepEmbedder(hiddenSize: hiddenDim)

        var mergeDim = hiddenDim + inChannels * 2
        if styleCondition && !styleAsToken { mergeDim += styleDim }
        self._condXMergeLinear.wrappedValue = Linear(mergeDim, hiddenDim)

        self._skipLinear.wrappedValue = Linear(hiddenDim + inChannels, hiddenDim)

        self._tEmbedder2.wrappedValue = TimestepEmbedder(hiddenSize: wavenetHiddenDim)
        self._conv1.wrappedValue = Linear(hiddenDim, wavenetHiddenDim)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: wavenetHiddenDim, outputChannels: inChannels, kernelSize: 1)
        self._wavenet.wrappedValue = WN(
            hiddenChannels: wavenetHiddenDim, kernelSize: wavenetKernelSize,
            dilationRate: wavenetDilationRate, nLayers: wavenetNumLayers,
            ginChannels: wavenetHiddenDim)
        self._finalLayer.wrappedValue = DiTFinalLayer(
            hiddenSize: wavenetHiddenDim, patchSize: 1, outChannels: wavenetHiddenDim)
        self._resProjection.wrappedValue = Linear(hiddenDim, wavenetHiddenDim)
    }

    /// x, promptX: (B, in_channels, T) NCL; t: (B,); style: (B, style_dim);
    /// cond: (B, T, content_dim). Returns predicted flow (B, in_channels, T) NCL.
    public func callAsFunction(
        _ x: MLXArray, promptX: MLXArray, xLens: MLXArray, t: MLXArray,
        style: MLXArray, cond: MLXArray
    ) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(2))

        let t1 = tEmbedder(t)  // (B, hidden)
        let condProj = condProjection(cond)  // (B, T, hidden)

        let xT = x.transposed(0, 2, 1)  // (B, T, in)
        let promptXT = promptX.transposed(0, 2, 1)

        var xIn = concatenated([xT, promptXT, condProj], axis: -1)
        if transformerStyleCondition && !styleAsToken {
            let styleExpanded = repeated(style[0..., .newAxis, 0...], count: T, axis: 1)
            xIn = concatenated([xIn, styleExpanded], axis: -1)
        }
        // (mask_content CFG branch unused: the CFM zeroes cond/style/prompt instead)
        xIn = condXMergeLinear(xIn)
        // (style_as_token / time_as_token: false for this checkpoint)

        let seqLen = xIn.dim(1)
        let inputPos = MLXArray(0 ..< seqLen)

        var xMask = ditSequenceMask(xLens, maxLength: seqLen)  // (b, seq)
        xMask = xMask[0..., .newAxis, .newAxis, 0...]
        xMask = xMask * MLXArray.ones([1, 1, seqLen, 1])
        let attnMask = which(xMask .> 0, MLXArray(Float(0)), MLXArray(-Float.infinity))

        var xRes = transformer(xIn, c: t1[0..., .newAxis, 0...], inputPos: inputPos, mask: attnMask)

        if longSkipConnection {
            xRes = skipLinear(concatenated([xRes, xT], axis: -1))
        }

        // wavenet final layer
        var xOut = conv1(xRes)               // (B, T, wn_hidden) NLC
        xOut = xOut.transposed(0, 2, 1)      // NLC -> NCL
        let t2 = tEmbedder2(t)
        let xMaskWN = MLXArray.ones([B, 1, T])
        var wnOut = wavenet(xOut, xMask: xMaskWN, g: t2[0..., 0..., .newAxis])
        wnOut = wnOut.transposed(0, 2, 1)    // NCL -> NLC
        xOut = wnOut + resProjection(xRes)   // long residual
        xOut = finalLayer(xOut, t1)          // NLC
        // donor: NLC -> NCL -> NLC (net no-op) then conv2 in NLC
        xOut = conv2(xOut)                   // (B, T, in_channels)
        return xOut.transposed(0, 2, 1)      // NLC -> NCL
    }
}
