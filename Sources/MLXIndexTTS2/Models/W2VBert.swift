// W2VBert.swift — facebook/w2v-bert-2.0 Conformer encoder (SeamlessM4T-v2).
//
// Isomorphic port of our verified MLX-Python donor `_indextts2-oracle/w2vbert_mlx/w2vbert.py`
// (itself isomorphic to HF modeling_wav2vec2_bert.py). Same classes, same call order, same
// key paths as model.safetensors after `W2VBertModel.sanitize`.
//
// Config facts for this checkpoint (from config.json, not defaults):
//   position_embeddings_type = "relative_key"  (Shaw-style, per-layer distance_embedding
//   (73, 64) = left(64)+right(8)+1 positions × head_size; asymmetric clamp)
//   add_adapter = false, use_intermediate_ffn_before_adapter = false, hidden_act = swish
//
// The IndexTTS2 semantic tap is hidden_states[17] (mid-stack; hiddenStates is a 25-element
// ladder, index 0 = the masked pre-encoder input), then per-channel z-norm with the
// wav2vec2bert_stats mean/std.

import Foundation
import MLX
import MLXNN

public struct Wav2Vec2BertConfig {
    public var hiddenSize = 1024
    public var numAttentionHeads = 16
    public var numHiddenLayers = 24
    public var intermediateSize = 4096
    public var featureProjectionInputDim = 160
    public var layerNormEps: Float = 1e-5
    public var convDepthwiseKernelSize = 31
    public var leftMaxPositionEmbeddings = 64
    public var rightMaxPositionEmbeddings = 8

    public init() {}
}

public final class Wav2Vec2BertFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "projection") var projection: Linear

    public init(_ config: Wav2Vec2BertConfig) {
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.featureProjectionInputDim, eps: config.layerNormEps)
        self._projection.wrappedValue = Linear(config.featureProjectionInputDim, config.hiddenSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        projection(layerNorm(hiddenStates))
    }
}

public final class Wav2Vec2BertFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    public init(_ config: Wav2Vec2BertConfig) {
        self._intermediateDense.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        self._outputDense.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        outputDense(silu(intermediateDense(hiddenStates)))
    }
}

/// pointwise_conv1 → GLU → (causal left pad) depthwise_conv → LN → Swish → pointwise_conv2.
/// Channels-last (B, T, C); the 1×1 pointwise convs are bias-free (O, I) matmuls.
public final class Wav2Vec2BertConvolutionModule: Module {
    public let hiddenSize: Int
    public let kernelSize: Int

    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ParameterInfo(key: "pointwise_conv1_w") var pointwiseConv1W: MLXArray
    @ParameterInfo(key: "depthwise_conv_w") var depthwiseConvW: MLXArray
    @ModuleInfo(key: "depthwise_layer_norm") var depthwiseLayerNorm: LayerNorm
    @ParameterInfo(key: "pointwise_conv2_w") var pointwiseConv2W: MLXArray

    public init(_ config: Wav2Vec2BertConfig) {
        self.hiddenSize = config.hiddenSize
        self.kernelSize = config.convDepthwiseKernelSize
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._pointwiseConv1W.wrappedValue = MLXArray.zeros([2 * config.hiddenSize, config.hiddenSize])
        // MLX Conv1d weight layout is (out_ch, kernel, in_ch/groups); groups == hiddenSize.
        self._depthwiseConvW.wrappedValue = MLXArray.zeros([config.hiddenSize, kernelSize, 1])
        self._depthwiseLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._pointwiseConv2W.wrappedValue = MLXArray.zeros([config.hiddenSize, config.hiddenSize])
    }

    public func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var hiddenStates = layerNorm(hiddenStates)

        // Zero padded positions before the depthwise conv leaks them.
        if let attentionMask {
            hiddenStates = hiddenStates * attentionMask.asType(.float32).expandedDimensions(axis: -1)
        }

        // pointwise_conv1 (1x1): (B,T,C) @ (C,2C)ᵀ → (B,T,2C)
        hiddenStates = matmul(hiddenStates, pointwiseConv1W.transposed())

        // GLU over the channel axis
        let parts = split(hiddenStates, parts: 2, axis: -1)
        hiddenStates = parts[0] * sigmoid(parts[1])

        // causal left pad along time by kernel-1, then depthwise conv
        let pad = kernelSize - 1
        hiddenStates = padded(hiddenStates, widths: [IntOrPair((0, 0)), IntOrPair((pad, 0)), IntOrPair((0, 0))])
        hiddenStates = conv1d(hiddenStates, depthwiseConvW, stride: 1, padding: 0, groups: hiddenSize)

        hiddenStates = silu(depthwiseLayerNorm(hiddenStates))

        // pointwise_conv2 (1x1)
        return matmul(hiddenStates, pointwiseConv2W.transposed())
    }
}

/// Shaw-style `relative_key` self-attention: per-layer distance_embedding (73, 64),
/// distance = clip(j−i, −left(64), +right(8)) + left, rel added to scores / √head_size.
public final class Wav2Vec2BertSelfAttention: Module {
    public let headSize: Int
    public let numHeads: Int
    public let leftMaxPositionEmbeddings: Int
    public let rightMaxPositionEmbeddings: Int

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "distance_embedding") var distanceEmbedding: Embedding

    public init(_ config: Wav2Vec2BertConfig) {
        let hiddenSize = config.hiddenSize
        self.headSize = hiddenSize / config.numAttentionHeads
        self.numHeads = config.numAttentionHeads
        self.leftMaxPositionEmbeddings = config.leftMaxPositionEmbeddings
        self.rightMaxPositionEmbeddings = config.rightMaxPositionEmbeddings

        self._linearQ.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._linearK.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._linearV.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._linearOut.wrappedValue = Linear(hiddenSize, hiddenSize)
        let numPositions = config.leftMaxPositionEmbeddings + config.rightMaxPositionEmbeddings + 1
        self._distanceEmbedding.wrappedValue = Embedding(
            embeddingCount: numPositions, dimensions: headSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let B = hiddenStates.dim(0)

        var query = linearQ(hiddenStates).reshaped(B, -1, numHeads, headSize)
        var key = linearK(hiddenStates).reshaped(B, -1, numHeads, headSize)
        var value = linearV(hiddenStates).reshaped(B, -1, numHeads, headSize)

        // (B, H, T, d)
        query = query.transposed(0, 2, 1, 3)
        key = key.transposed(0, 2, 1, 3)
        value = value.transposed(0, 2, 1, 3)

        var scores = matmul(query, key.transposed(0, 1, 3, 2)) / sqrt(Float(headSize))

        // relative_key positional scores
        let ql = query.dim(2), kl = key.dim(2)
        let posL = MLXArray(0 ..< Int32(ql)).reshaped(ql, 1)
        let posR = MLXArray(0 ..< Int32(kl)).reshaped(1, kl)
        let distance = clip(
            posR - posL,
            min: Int32(-leftMaxPositionEmbeddings), max: Int32(rightMaxPositionEmbeddings))
        let idx = distance + Int32(leftMaxPositionEmbeddings)     // (ql, kl)
        let positionalEmbedding = distanceEmbedding(idx)          // (ql, kl, d)

        // einsum("bhld,lrd->bhlr", query, positional_embedding)
        let rel = einsum("bhld,lrd->bhlr", query, positionalEmbedding)
        scores = scores + rel / sqrt(Float(headSize))

        if let attentionMask { scores = scores + attentionMask }

        let probs = softmax(scores, axis: -1)
        var out = matmul(probs, value)                            // (B,H,T,d)
        out = out.transposed(0, 2, 1, 3).reshaped(B, -1, numHeads * headSize)
        return linearOut(out)
    }
}

/// Conformer layer: 0.5·FFN1 + res → SelfAttn + res → Conv + res → 0.5·FFN2 + res → final LN.
public final class Wav2Vec2BertEncoderLayer: Module {
    @ModuleInfo(key: "ffn1_layer_norm") var ffn1LayerNorm: LayerNorm
    @ModuleInfo(key: "ffn1") var ffn1: Wav2Vec2BertFeedForward
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Wav2Vec2BertSelfAttention
    @ModuleInfo(key: "conv_module") var convModule: Wav2Vec2BertConvolutionModule
    @ModuleInfo(key: "ffn2_layer_norm") var ffn2LayerNorm: LayerNorm
    @ModuleInfo(key: "ffn2") var ffn2: Wav2Vec2BertFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    public init(_ config: Wav2Vec2BertConfig) {
        self._ffn1LayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._ffn1.wrappedValue = Wav2Vec2BertFeedForward(config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._selfAttn.wrappedValue = Wav2Vec2BertSelfAttention(config)
        self._convModule.wrappedValue = Wav2Vec2BertConvolutionModule(config)
        self._ffn2LayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._ffn2.wrappedValue = Wav2Vec2BertFeedForward(config)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray, attentionMask: MLXArray?, convAttentionMask: MLXArray?
    ) -> MLXArray {
        // 1. FFN1 (half-step macaron)
        var residual = hiddenStates
        var hiddenStates = ffn1(ffn1LayerNorm(hiddenStates)) * 0.5 + residual
        residual = hiddenStates

        // 2. Self-Attention
        hiddenStates = selfAttn(selfAttnLayerNorm(hiddenStates), attentionMask: attentionMask)
        hiddenStates = hiddenStates + residual

        // 3. Convolution
        residual = hiddenStates
        hiddenStates = residual + convModule(hiddenStates, attentionMask: convAttentionMask)

        // 4. FFN2 (half-step macaron) + final LN
        residual = hiddenStates
        hiddenStates = ffn2(ffn2LayerNorm(hiddenStates)) * 0.5 + residual
        return finalLayerNorm(hiddenStates)
    }
}

public final class Wav2Vec2BertEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [Wav2Vec2BertEncoderLayer]

    public init(_ config: Wav2Vec2BertConfig) {
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            Wav2Vec2BertEncoderLayer(config)
        }
    }

    /// Returns (lastHiddenState, allHiddenStates); allHiddenStates has numLayers+1 entries,
    /// index 0 = the (mask-zeroed) pre-encoder input.
    public func callAsFunction(
        _ hiddenStates: MLXArray, attentionMask: MLXArray?
    ) -> (MLXArray, [MLXArray]) {
        var hiddenStates = hiddenStates
        var allHiddenStates: [MLXArray] = []

        let convAttentionMask = attentionMask  // (B, T), 1 = keep, 0 = pad
        var extMask: MLXArray? = nil
        if let attentionMask {
            let maskF = attentionMask.asType(.float32)
            // zero out padded tokens (affects hidden_states[0])
            hiddenStates = hiddenStates * maskF.expandedDimensions(axis: -1)
            // extended additive mask: (B,1,T,T), 0 for keep, large-neg for pad
            let neg: Float = -3.4028235e38  // torch.finfo(float32).min
            let T = attentionMask.dim(-1)
            let m = (1.0 - maskF)[0..., .newAxis, .newAxis, 0...] * neg  // (B,1,1,T)
            extMask = broadcast(m, to: [attentionMask.dim(0), 1, T, T])
        }

        for layer in layers {
            allHiddenStates.append(hiddenStates)
            hiddenStates = layer(
                hiddenStates, attentionMask: extMask, convAttentionMask: convAttentionMask)
        }
        allHiddenStates.append(hiddenStates)
        return (hiddenStates, allHiddenStates)
    }
}

public final class Wav2Vec2BertModel: Module {
    public let config: Wav2Vec2BertConfig

    @ModuleInfo(key: "feature_projection") var featureProjection: Wav2Vec2BertFeatureProjection
    @ModuleInfo(key: "encoder") var encoder: Wav2Vec2BertEncoder

    public init(_ config: Wav2Vec2BertConfig = Wav2Vec2BertConfig()) {
        self.config = config
        self._featureProjection.wrappedValue = Wav2Vec2BertFeatureProjection(config)
        self._encoder.wrappedValue = Wav2Vec2BertEncoder(config)
        // adapter / intermediate_ffn are absent in this config.
    }

    public func callAsFunction(
        inputFeatures: MLXArray, attentionMask: MLXArray?
    ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]) {
        let hiddenStates = featureProjection(inputFeatures)
        // _mask_hidden_states is a no-op at inference (apply_spec_augment=false)
        let (lastHiddenState, allHiddenStates) = encoder(hiddenStates, attentionMask: attentionMask)
        return (lastHiddenState, allHiddenStates)
    }

    /// The IndexTTS2 semantic embedding: per-channel z-norm of hidden_states[17].
    public static func semanticTap(
        _ hiddenStates: [MLXArray], mean: MLXArray, std: MLXArray
    ) -> MLXArray {
        (hiddenStates[17] - mean) / std
    }

    /// Remap HF safetensors keys/layouts → this module's parameter tree
    /// (mirrors the Python donor's `sanitize`):
    /// - drop unused params (masked_spec_embed, adapter, intermediate_ffn)
    /// - pointwise convs (O,I,1) → (O,I) matrices stored as *_w
    /// - depthwise conv (C,1,K) torch → (C,K,1) MLX
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k == "masked_spec_embed" { continue }
            if k.hasPrefix("adapter.") || k.hasPrefix("intermediate_ffn.") { continue }
            if k.contains("conv_module.pointwise_conv1.weight") {
                out[k.replacingOccurrences(of: "pointwise_conv1.weight", with: "pointwise_conv1_w")] =
                    v.reshaped(v.dim(0), v.dim(1))
                continue
            }
            if k.contains("conv_module.pointwise_conv2.weight") {
                out[k.replacingOccurrences(of: "pointwise_conv2.weight", with: "pointwise_conv2_w")] =
                    v.reshaped(v.dim(0), v.dim(1))
                continue
            }
            if k.contains("conv_module.depthwise_conv.weight") {
                // torch (C, 1, K) → MLX (C, K, 1)
                out[k.replacingOccurrences(of: "depthwise_conv.weight", with: "depthwise_conv_w")] =
                    v.transposed(0, 2, 1)
                continue
            }
            out[k] = v
        }
        return out
    }
}
