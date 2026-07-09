// RepCodec.swift — MaskGCT semantic codec (RepCodec + FactorizedVectorQuantize).
//
// Isomorphic port of our verified MLX-Python donor `_indextts2-oracle/maskgct_mlx/repcodec.py`
// (itself isomorphic to Amphion repcodec_model.py / amphion_codec quantizers / kmeans vocos.py).
// Reproduces `semantic_codec.quantize(spk_cond_emb)` → (codes, S_ref).
//
// Resolved config (from amphion/MaskGCT semantic_codec/model.safetensors, NOT the config knob —
// num_quantizers=1, the downstream "3" is a red herring): codebook_size=8192, codebook_dim=8,
// hidden=1024, vocos_dim=384/inter=2048/layers=12, downsample_scale=1 (no down/up convs).
//
// Numerics traps carried from the donor: in/out projections are weight_norm'd 1×1 Conv1d —
// fused g·v/‖v‖ → (O,I) matmul in `sanitize`; the VQ distance L2-normalizes BOTH encodings and
// codebook, but `decodeCode` looks up the RAW codebook; ALL LayerNorms are eps=1e-6;
// straight-through is identity at inference; decoder is loaded but unused by quantize().

import Foundation
import MLX
import MLXNN

/// 1D ConvNeXt block, channels-last (B, T, C).
public final class ConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    public init(dim: Int, intermediateDim: Int) {
        self._dwconv.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, intermediateDim)
        self._pwconv2.wrappedValue = Linear(intermediateDim, dim)
        self._gamma.wrappedValue = MLXArray.zeros([dim])  // layer scale (loaded from ckpt)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var x = dwconv(x)            // (B, T, C) depthwise
        x = norm(x)                  // LayerNorm over channels (last axis)
        x = pwconv1(x)
        x = gelu(x)
        x = pwconv2(x)
        x = gamma * x
        return residual + x
    }
}

public final class VocosBackbone: Module {
    @ModuleInfo(key: "embed") var embed: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "convnext") var convnext: [ConvNeXtBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    public init(inputChannels: Int, dim: Int, intermediateDim: Int, numLayers: Int) {
        self._embed.wrappedValue = Conv1d(
            inputChannels: inputChannels, outputChannels: dim, kernelSize: 7, padding: 3)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._convnext.wrappedValue = (0 ..< numLayers).map { _ in
            ConvNeXtBlock(dim: dim, intermediateDim: intermediateDim)
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
    }

    // Ladder access for the parity gate.
    public var embedLayer: Conv1d { embed }
    public var normLayer: LayerNorm { norm }
    public var convnextBlocks: [ConvNeXtBlock] { convnext }
    public var finalLayerNormLayer: LayerNorm { finalLayerNorm }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, T, C_in) channels-last (torch takes (B, C_in, T))
        var x = embed(x)             // (B, T, dim)
        x = norm(x)
        for block in convnext {
            x = block(x)
        }
        return finalLayerNorm(x)
    }
}

/// torch F.normalize default: p=2, eps=1e-12, over the last axis here.
private func l2Normalized(_ x: MLXArray, eps: Float = 1e-12) -> MLXArray {
    x / maximum(sqrt(sum(x * x, axis: -1, keepDims: true)), eps)
}

/// FVQ. in/out projections are weight_norm 1×1 Conv1d, materialized in `sanitize` into
/// plain (O,I) matmuls. Codebook lookup = argmin L2 distance in the L2-normalized code space.
public final class FactorizedVectorQuantize: Module {
    public let codebookSize: Int
    public let codebookDim: Int

    @ParameterInfo(key: "in_project_w") var inProjectW: MLXArray
    @ParameterInfo(key: "in_project_b") var inProjectB: MLXArray
    @ParameterInfo(key: "out_project_w") var outProjectW: MLXArray
    @ParameterInfo(key: "out_project_b") var outProjectB: MLXArray
    @ModuleInfo(key: "codebook") var codebook: Embedding

    public init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self._inProjectW.wrappedValue = MLXArray.zeros([codebookDim, inputDim])
        self._inProjectB.wrappedValue = MLXArray.zeros([codebookDim])
        self._outProjectW.wrappedValue = MLXArray.zeros([inputDim, codebookDim])
        self._outProjectB.wrappedValue = MLXArray.zeros([inputDim])
        self._codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
    }

    public func inProject(_ x: MLXArray) -> MLXArray {  // (B,T,in) → (B,T,code)
        matmul(x, inProjectW.transposed()) + inProjectB
    }

    public func outProject(_ x: MLXArray) -> MLXArray {  // (B,T,code) → (B,T,in)
        matmul(x, outProjectW.transposed()) + outProjectB
    }

    public func decodeLatents(_ zE: MLXArray) -> (zQ: MLXArray, indices: MLXArray) {
        let (B, T, D) = (zE.dim(0), zE.dim(1), zE.dim(2))
        var encodings = zE.reshaped(B * T, D)
        var codebookW = codebook.weight  // (K, D)

        // L2-normalize BOTH sides for the distance metric only.
        encodings = l2Normalized(encodings)
        codebookW = l2Normalized(codebookW)

        // dist = |e|² − 2·e·cᵀ + |c|²   (rows: B*T, cols: K)
        let dist = sum(encodings * encodings, axis: 1, keepDims: true)
            - 2 * matmul(encodings, codebookW.transposed())
            + sum(codebookW * codebookW, axis: 1, keepDims: true).transposed()
        let indices = argMax(-dist, axis: 1)         // (B*T,)
        let zQ = decodeCode(indices).reshaped(B, T, D)
        return (zQ, indices.reshaped(B, T))
    }

    /// RAW codebook lookup — no normalization (only the distance computation normalizes).
    public func decodeCode(_ indices: MLXArray) -> MLXArray {
        codebook.weight[indices]
    }

    public func callAsFunction(_ z: MLXArray) -> (zQ: MLXArray, indices: MLXArray) {
        let zE = inProject(z)
        let (zQ, indices) = decodeLatents(zE)
        // inference straight-through is identity: z_q = z_e + (z_q - z_e)
        return (outProject(zQ), indices)
    }
}

public final class ResidualVQ: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [FactorizedVectorQuantize]

    public init(inputDim: Int, numQuantizers: Int, codebookSize: Int, codebookDim: Int) {
        self._quantizers.wrappedValue = (0 ..< numQuantizers).map { _ in
            FactorizedVectorQuantize(
                inputDim: inputDim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    public var firstQuantizer: FactorizedVectorQuantize { quantizers[0] }

    public func callAsFunction(_ z: MLXArray) -> (quantizedOut: MLXArray, allIndices: MLXArray) {
        // Inference-only residual loop.
        var quantizedOut = MLXArray.zeros(z.shape)
        var residual = z
        var allIndices: [MLXArray] = []
        for quantizer in quantizers {
            let (zQi, indicesI) = quantizer(residual)
            quantizedOut = quantizedOut + zQi
            residual = residual - zQi
            allIndices.append(indicesI)
        }
        return (quantizedOut, stacked(allIndices, axis: 0))  // (N, B, T)
    }
}

public final class RepCodec: Module {
    public let hiddenSize: Int

    @ModuleInfo(key: "encoder_backbone") var encoderBackbone: VocosBackbone
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    // decoder present in the checkpoint but unused by quantize(); kept for weight loading
    @ModuleInfo(key: "decoder_backbone") var decoderBackbone: VocosBackbone
    @ModuleInfo(key: "decoder_proj") var decoderProj: Linear
    @ModuleInfo(key: "quantizer") var quantizer: ResidualVQ

    public init(
        codebookSize: Int = 8192,
        hiddenSize: Int = 1024,
        codebookDim: Int = 8,
        vocosDim: Int = 384,
        vocosIntermediateDim: Int = 2048,
        vocosNumLayers: Int = 12,
        numQuantizers: Int = 1
    ) {
        self.hiddenSize = hiddenSize
        self._encoderBackbone.wrappedValue = VocosBackbone(
            inputChannels: hiddenSize, dim: vocosDim,
            intermediateDim: vocosIntermediateDim, numLayers: vocosNumLayers)
        self._encoderProj.wrappedValue = Linear(vocosDim, hiddenSize)
        self._decoderBackbone.wrappedValue = VocosBackbone(
            inputChannels: hiddenSize, dim: vocosDim,
            intermediateDim: vocosIntermediateDim, numLayers: vocosNumLayers)
        self._decoderProj.wrappedValue = Linear(vocosDim, hiddenSize)
        self._quantizer.wrappedValue = ResidualVQ(
            inputDim: hiddenSize, numQuantizers: numQuantizers,
            codebookSize: codebookSize, codebookDim: codebookDim)
    }

    // Ladder access for the parity gate.
    public var encoderBackboneModule: VocosBackbone { encoderBackbone }
    public var encoderProjLayer: Linear { encoderProj }
    public var quantizerModule: ResidualVQ { quantizer }

    public func encoder(_ x: MLXArray) -> MLXArray {
        // x: (B, T, hidden) channels-last. torch: encoder(x.transpose(1,2)).transpose(1,2)
        encoderProj(encoderBackbone(x))
    }

    /// x: (B, T, hidden) → (codes (B,T) int, S_ref (B,T,hidden)).
    public func quantize(_ x: MLXArray) -> (codes: MLXArray, sRef: MLXArray) {
        // (no downsample_scale>1 branch for this checkpoint)
        let z = encoder(x)
        let (quantizedOut, allIndices) = quantizer(z)
        // channels-last quantizedOut already == torch's quantized_out.transpose(1,2);
        // allIndices (N=1,B,T) → [0] → (B,T).
        return (allIndices[0], quantizedOut)
    }

    /// Remap amphion/MaskGCT semantic_codec torch keys → this module's parameter tree
    /// (mirrors the Python donor's `sanitize`):
    /// - encoder.0.* → encoder_backbone.*, encoder.1.* → encoder_proj.* (decoder likewise)
    /// - weight_norm'd 1×1 conv projections fused: w = g·v/‖v‖ over (I,K) axes → (O,I)
    /// - Conv1d (O,I,K) → (O,K,I); depthwise (C,1,K) → (C,K,1)
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var weightNormParts: [String: MLXArray] = [:]

        for (k, v) in weights {
            if k.contains(".in_project.weight_g") || k.contains(".in_project.weight_v")
                || k.contains(".out_project.weight_g") || k.contains(".out_project.weight_v") {
                weightNormParts[k] = v
                continue
            }

            var nk = k
            if nk.hasPrefix("encoder.0.") {
                nk = "encoder_backbone." + nk.dropFirst("encoder.0.".count)
            } else if nk.hasPrefix("encoder.1.") {
                nk = "encoder_proj." + nk.dropFirst("encoder.1.".count)
            } else if nk.hasPrefix("decoder.0.") {
                nk = "decoder_backbone." + nk.dropFirst("decoder.0.".count)
            } else if nk.hasPrefix("decoder.1.") {
                nk = "decoder_proj." + nk.dropFirst("decoder.1.".count)
            }

            if nk.hasSuffix("in_project.bias") {
                out[nk.replacingOccurrences(of: "in_project.bias", with: "in_project_b")] = v
                continue
            }
            if nk.hasSuffix("out_project.bias") {
                out[nk.replacingOccurrences(of: "out_project.bias", with: "out_project_b")] = v
                continue
            }

            if nk.hasSuffix("embed.weight") && v.ndim == 3 {
                out[nk] = v.transposed(0, 2, 1)   // (O, I, K) → (O, K, I)
                continue
            }
            if nk.hasSuffix("dwconv.weight") && v.ndim == 3 {
                out[nk] = v.transposed(0, 2, 1)   // depthwise (C, 1, K) → (C, K, 1)
                continue
            }

            out[nk] = v
        }

        // fuse weight_norm projections
        for base in ["quantizer.quantizers.0.in_project", "quantizer.quantizers.0.out_project"] {
            guard let g = weightNormParts[base + ".weight_g"],
                  let v = weightNormParts[base + ".weight_v"] else { continue }
            // v: (O, I, 1), g: (O, 1, 1); norm over all axes except out-channel (torch dim=0)
            let norm = sqrt(sum(v * v, axes: [1, 2], keepDims: true))  // (O,1,1)
            let w = (g * v / norm).squeezed(axis: 2)                    // (O, I)
            out[base.replacingOccurrences(of: "in_project", with: "in_project_w")
                    .replacingOccurrences(of: "out_project", with: "out_project_w")] = w
        }

        return out
    }
}
