// Conformer.swift — Conformer encoder for IndexTTS2 conditioning.
//
// Isomorphic port of mlx_indextts/models/conformer.py (solar2ain, MIT). Key facts carried
// from the donor: RelPositionalEncoding multiplies x by √dim and returns pe UNADDED;
// rel_shift is NOT used (commented out upstream); solar2ain's ConvolutionModule ignores
// the padding mask; NO macaron FFNs in this variant (ff_scale = 1.0, no
// feed_forward_macaron weights in gpt.safetensors); scores = (matrix_ac + matrix_bd)·scale.
//
// Masking note: the donor builds an all-true mask when every sequence is full-length
// (batch=1 in this pipeline) — `mx.where(all-true, scores, -inf)` is the identity, so this
// port passes mask=nil in that case and applies `where` only when a real mask exists.

import Foundation
import MLX
import MLXNN

/// Resolved conditioner configs (checkpoint config.yaml values, not dataclass defaults).
public struct ConformerConfig: Sendable {
    public var inputSize = 1024
    public var outputSize = 512
    public var linearUnits = 2048
    public var attentionHeads = 8
    public var numBlocks = 6
    public var cnnModuleKernel = 15

    public init(inputSize: Int = 1024, outputSize: Int = 512, linearUnits: Int = 2048,
                attentionHeads: Int = 8, numBlocks: Int = 6, cnnModuleKernel: Int = 15) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.linearUnits = linearUnits
        self.attentionHeads = attentionHeads
        self.numBlocks = numBlocks
        self.cnnModuleKernel = cnnModuleKernel
    }
}

public final class PositionwiseFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    public init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim)
        self._w2.wrappedValue = Linear(hiddenDim, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)))
    }
}

/// pointwise → GLU → depthwise (SAME pad) → LN → SiLU → pointwise.
/// (solar2ain's version does not apply the padding mask.)
public final class ConformerConvolutionModule: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    public init(channels: Int, kernelSize: Int = 15) {
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: 2 * channels, kernelSize: 1)
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
            padding: (kernelSize - 1) / 2, groups: channels)
        self._norm.wrappedValue = LayerNorm(dimensions: channels)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = pointwiseConv1(x)
        let parts = split(x, parts: 2, axis: -1)
        x = parts[0] * sigmoid(parts[1])
        x = depthwiseConv(x)
        x = silu(norm(x))
        return pointwiseConv2(x)
    }
}

/// Sinusoidal relative positional encoding. Returns (x·√dim, pe) — pe is NOT added to x.
public final class RelPositionalEncoding: Module {
    /// Non-Module holder so the precomputed table is not reflected as a parameter.
    private final class PETable {
        let pe: MLXArray
        init(_ pe: MLXArray) { self.pe = pe }
    }

    public let dim: Int
    public let xscale: Float
    private let table: PETable

    public init(dim: Int, maxLen: Int = 5000) {
        self.dim = dim
        self.xscale = Float(dim).squareRoot()
        let position = MLXArray(0 ..< Int32(maxLen)).asType(.float32).reshaped(maxLen, 1)
        let divTerm = exp(
            MLXArray(stride(from: Int32(0), to: Int32(dim), by: 2)).asType(.float32)
                * (-log(10000.0) / Float(dim)))
        let args = position * divTerm                       // (maxLen, dim/2)
        // even indices = sin, odd = cos → interleave pairs
        let pe = stacked([sin(args), cos(args)], axis: -1).reshaped(maxLen, dim)
        self.table = PETable(pe)
    }

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = x.dim(1)
        let pe = table.pe[..<seqLen][.newAxis, 0..., 0...]
        return (x * xscale, pe)
    }
}

/// Multi-head attention with relative position encoding (wenet-style; rel_shift unused).
public final class RelPositionMultiHeadAttention: Module {
    public let numHeads: Int
    public let dim: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    public init(numHeads: Int, dim: Int) {
        self.numHeads = numHeads
        self.dim = dim
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._linearQ.wrappedValue = Linear(dim, dim)
        self._linearK.wrappedValue = Linear(dim, dim)
        self._linearV.wrappedValue = Linear(dim, dim)
        self._linearOut.wrappedValue = Linear(dim, dim)
        self._linearPos.wrappedValue = Linear(dim, dim, bias: false)
        self._posBiasU.wrappedValue = MLXArray.zeros([numHeads, headDim])
        self._posBiasV.wrappedValue = MLXArray.zeros([numHeads, headDim])
    }

    public func callAsFunction(
        _ query: MLXArray, _ key: MLXArray, _ value: MLXArray,
        mask: MLXArray?, posEmb: MLXArray
    ) -> MLXArray {
        let (b, seqLen, _) = (query.dim(0), query.dim(1), query.dim(2))

        let q = linearQ(query).reshaped(b, seqLen, numHeads, headDim)
        let k = linearK(key).reshaped(b, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = linearV(value).reshaped(b, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        let nBatchPos = posEmb.dim(0)
        let p = linearPos(posEmb).reshaped(nBatchPos, -1, numHeads, headDim).transposed(0, 2, 1, 3)

        // position bias added BEFORE the transpose (PyTorch order)
        let qWithBiasU = (q + posBiasU[.newAxis, .newAxis, 0..., 0...]).transposed(0, 2, 1, 3)
        let qWithBiasV = (q + posBiasV[.newAxis, .newAxis, 0..., 0...]).transposed(0, 2, 1, 3)

        let matrixAC = matmul(qWithBiasU, k.transposed(0, 1, 3, 2))
        let matrixBD = matmul(qWithBiasV, p.transposed(0, 1, 3, 2))
        // NOTE: rel_shift is NOT used (commented out upstream)
        var scores = (matrixAC + matrixBD) * scale

        if var mask {
            if mask.ndim == 3 { mask = mask[0..., .newAxis, 0..., 0...] }
            scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
        }

        let attn = softmax(scores, axis: -1)
        var out = matmul(attn, v)
        out = out.transposed(0, 2, 1, 3).reshaped(b, seqLen, dim)
        return linearOut(out)
    }
}

/// Conv2d subsampling to 1/2 length: Conv2d(1→odim, k3, s2) + ReLU → flatten (c,f) → Linear.
public final class Conv2dSubsampling2: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "out") var out: Linear
    public let posEnc: RelPositionalEncoding

    public init(inputDim: Int, outputDim: Int, posEnc: RelPositionalEncoding) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: outputDim, kernelSize: 3, stride: 2)
        self._out.wrappedValue = Linear(outputDim * ((inputDim - 1) / 2), outputDim)
        self.posEnc = posEnc
    }

    /// x: (B, T, F) → (subsampled (B, T', odim), posEmb (1, T', odim)).
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var x = x.expandedDimensions(axis: -1)   // NHWC: (B, T, F, 1)
        x = relu(conv(x))                        // (B, T', F', odim)
        let (b, t, f, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // PyTorch flattens (c, f); NHWC has (f, c) → transpose first
        x = x.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
        x = out(x)
        return posEnc(x)
    }
}

public final class ConformerEncoderLayer: Module {
    @ModuleInfo(key: "norm_ff") var normFF: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: PositionwiseFeedForward
    @ModuleInfo(key: "norm_mha") var normMHA: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionMultiHeadAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv_module") var convModule: ConformerConvolutionModule
    @ModuleInfo(key: "norm_final") var normFinal: LayerNorm

    public init(dim: Int, numHeads: Int, ffDim: Int, kernelSize: Int = 15) {
        self._normFF.wrappedValue = LayerNorm(dimensions: dim)
        self._feedForward.wrappedValue = PositionwiseFeedForward(dim: dim, hiddenDim: ffDim)
        self._normMHA.wrappedValue = LayerNorm(dimensions: dim)
        self._selfAttn.wrappedValue = RelPositionMultiHeadAttention(numHeads: numHeads, dim: dim)
        self._normConv.wrappedValue = LayerNorm(dimensions: dim)
        self._convModule.wrappedValue = ConformerConvolutionModule(channels: dim, kernelSize: kernelSize)
        self._normFinal.wrappedValue = LayerNorm(dimensions: dim)
    }

    /// normalize_before=true, no macaron (ff_scale = 1.0), use_cnn_module=true.
    public func callAsFunction(_ x: MLXArray, mask: MLXArray?, posEmb: MLXArray) -> MLXArray {
        // Self-attention
        var residual = x
        var x = normMHA(x)
        x = residual + selfAttn(x, x, x, mask: mask, posEmb: posEmb)

        // Convolution module
        residual = x
        x = residual + convModule(normConv(x))

        // Feed-forward
        residual = x
        x = residual + feedForward(normFF(x))

        // Final norm for conformer
        return normFinal(x)
    }
}

public final class ConformerEncoder: Module {
    public let config: ConformerConfig

    @ModuleInfo(key: "embed") var embed: Conv2dSubsampling2
    @ModuleInfo(key: "encoders") var encoders: [ConformerEncoderLayer]
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm

    public init(_ config: ConformerConfig) {
        self.config = config
        self._embed.wrappedValue = Conv2dSubsampling2(
            inputDim: config.inputSize, outputDim: config.outputSize,
            posEnc: RelPositionalEncoding(dim: config.outputSize))
        self._encoders.wrappedValue = (0 ..< config.numBlocks).map { _ in
            ConformerEncoderLayer(
                dim: config.outputSize, numHeads: config.attentionHeads,
                ffDim: config.linearUnits, kernelSize: config.cnnModuleKernel)
        }
        self._afterNorm.wrappedValue = LayerNorm(dimensions: config.outputSize)
    }

    /// x: (B, T, inputSize) NLC. Full-length sequences (batch=1 pipeline) → all-true mask
    /// → mask=nil (identity `where`), matching the donor numerically.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (subsampled, posEmb) = embed(x)
        var h = subsampled
        for layer in encoders {
            h = layer(h, mask: nil, posEmb: posEmb)
        }
        return afterNorm(h)
    }
}
