// EnhancedCodec.swift — the IndexTTS-2.5 semantic codec, DECODE path only
// (`indextts/codec/models.py EnhancedCodec.decode`): one factorized VQ codebook (8192×8,
// 1×1 out-projection to 1024) → Vocos/ConvNeXt backbone (12 blocks, dim 384, ff 2048) →
// Linear(384→1024) → nearest ×2 upsample → `up` Conv1d(k3). Output (B, 2T, 1024) is the
// S2Mel content stream (25 Hz codes → 50 Hz features; ×1.72 → mel frames).
//
// The checkpoint (codec.safetensors, donor layout, conv weights already (out, k, in)) also
// carries the ENCODER stack (`encoder.*`, `down.*`, `quantizer…in_project.*`) — inference never
// runs it (2.5 conditions on raw w2v-BERT features, not on re-quantized codes), so `sanitize`
// drops those keys explicitly and the rest is held to the 0-missing / 0-unused contract.
// Replaces the 2.0 RepCodec + vq2emb + gpt_layer path.

import Foundation
import MLX
import MLXNN

/// 1-D ConvNeXt block (vocos.py `ConvNeXtBlock`, no AdaLN): depthwise k7 → LN → pw1 → GELU
/// (exact, erf) → pw2 → layer-scale γ, residual. NLC in / NLC out.
public final class ConvNeXtBlock1D: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    public init(dim: Int, intermediateDim: Int, layerScaleInitValue: Float) {
        self._dwconv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7,
                                           padding: 3, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, intermediateDim)
        self._pwconv2.wrappedValue = Linear(intermediateDim, dim)
        self._gamma.wrappedValue = MLXArray.full([dim], values: MLXArray(layerScaleInitValue))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = dwconv(x)
        h = norm(h)
        h = pwconv2(gelu(pwconv1(h)))
        return x + gamma * h
    }
}

/// `VocosBackbone`: embed Conv1d(k7) → LN → ConvNeXt×N → final LN. NLC.
public final class VocosBackbone: Module {
    @ModuleInfo(key: "embed") var embed: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "convnext") var convnext: [ConvNeXtBlock1D]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    public init(inputChannels: Int, dim: Int, intermediateDim: Int, numLayers: Int) {
        self._embed.wrappedValue = Conv1d(inputChannels: inputChannels, outputChannels: dim,
                                          kernelSize: 7, padding: 3)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._convnext.wrappedValue = (0 ..< numLayers).map { _ in
            ConvNeXtBlock1D(dim: dim, intermediateDim: intermediateDim,
                            layerScaleInitValue: 1.0 / Float(numLayers))
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm(embed(x))
        for block in convnext { h = block(h) }
        return finalLayerNorm(h)
    }
}

/// The decode half of the codec (see header).
public final class EnhancedCodecDecoder: Module {
    @ModuleInfo(key: "codebook") var codebook: Embedding
    @ModuleInfo(key: "out_project") var outProject: Linear
    @ModuleInfo(key: "backbone") var backbone: VocosBackbone
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "up") var up: Conv1d

    public let downsampleScale: Int

    public init(codebookSize: Int = 8192, hiddenSize: Int = 1024, codebookDim: Int = 8,
                vocosDim: Int = 384, vocosIntermediateDim: Int = 2048, vocosNumLayers: Int = 12,
                downsampleScale: Int = 2) {
        self.downsampleScale = downsampleScale
        self._codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
        self._outProject.wrappedValue = Linear(codebookDim, hiddenSize)
        self._backbone.wrappedValue = VocosBackbone(inputChannels: hiddenSize, dim: vocosDim,
                                                   intermediateDim: vocosIntermediateDim,
                                                   numLayers: vocosNumLayers)
        self._proj.wrappedValue = Linear(vocosDim, hiddenSize)
        self._up.wrappedValue = Conv1d(inputChannels: hiddenSize, outputChannels: hiddenSize,
                                       kernelSize: 3, padding: 1)
    }

    /// `quantizer.vq2emb`: codes (B, T) → (B, T, 1024).
    public func vq2emb(_ codes: MLXArray) -> MLXArray {
        outProject(codebook(codes.asType(.int32)))
    }

    /// `EnhancedCodec.decode`: codes (B, T) → content features (B, T·scale, 1024).
    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        var x = proj(backbone(vq2emb(codes)))
        if downsampleScale > 1 {
            x = repeated(x, count: downsampleScale, axis: 1)   // F.interpolate nearest ×2
            x = up(x)
        }
        return x
    }

    /// codec.safetensors → this module's tree. Drops the unused encoder half; remaps the
    /// donor's list-indexed decoder (`decoder.0` backbone, `decoder.1` linear) and the
    /// quantizer's 1×1 conv out-projection ((1024, 1, 8) → Linear (1024, 8)).
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.hasPrefix("encoder.") || k.hasPrefix("down.") || k.contains(".in_project.") { continue }
            if k.hasPrefix("decoder.0.") {
                out["backbone." + k.dropFirst("decoder.0.".count)] = v
            } else if k.hasPrefix("decoder.1.") {
                out["proj." + k.dropFirst("decoder.1.".count)] = v
            } else if k == "quantizer.quantizers.0.codebook.weight" {
                out["codebook.weight"] = v
            } else if k == "quantizer.quantizers.0.out_project.weight" {
                out["out_project.weight"] = v.squeezed(axis: 1)
            } else if k == "quantizer.quantizers.0.out_project.bias" {
                out["out_project.bias"] = v
            } else {
                out[k] = v   // up.weight / up.bias
            }
        }
        return out
    }
}
