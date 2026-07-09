// LengthRegulator.swift — InterpolateRegulator for S2Mel (the DURATION-CONTROL surface).
//
// Isomorphic port of donor `mlx_indextts/models/s2mel/length_regulator.py`.
// Semantic content is nearest-neighbor-interpolated to the target mel length; the target
// length (generate_v2: `int(code_len * 1.72)`) is the E12 duration-control lever — callers
// choose `ylens` to stretch/compress speech, so it stays an explicit parameter here.
//
// Checkpoint keys are a heterogeneous torch Sequential (`model.0/1/3/4/...`: conv, groupnorm,
// mish(no keys), ×4, then a final 1×1 conv at model.12) — the classic numeric-Sequential
// pitfall; S2Mel.sanitize remaps them to convs.N / norms.N / out_proj.

import Foundation
import MLX
import MLXNN

/// GroupNorm over NCL input (donor hand-rolled version, num_groups=1 for this checkpoint).
public final class LRGroupNorm: Module {
    public let numGroups: Int
    public let eps: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    public init(numGroups: Int, numChannels: Int, eps: Float = 1e-5) {
        self.numGroups = max(numGroups, 1)
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([numChannels])
        self._bias.wrappedValue = MLXArray.zeros([numChannels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, channels, length) = (x.dim(0), x.dim(1), x.dim(2))
        var x = x.reshaped(batch, numGroups, channels / numGroups, length)
        let mean = x.mean(axes: [2, 3], keepDims: true)
        let variance = x.variance(axes: [2, 3], keepDims: true)
        x = (x - mean) / sqrt(variance + eps)
        x = x.reshaped(batch, channels, length)
        return x * weight[.newAxis, 0..., .newAxis] + bias[.newAxis, 0..., .newAxis]
    }
}

/// Mish, spelled exactly as the donor: x · tanh(log(1 + eˣ)).
func donorMish(_ x: MLXArray) -> MLXArray {
    x * tanh(log(1 + exp(x)))
}

/// Nearest-neighbor interpolation along L of an NCL tensor (donor _interpolate_nearest).
func interpolateNearest(_ x: MLXArray, targetLength: Int) -> MLXArray {
    let length = x.dim(2)
    let scale = Float(length) / Float(targetLength)
    var indices = (MLXArray(0 ..< targetLength).asType(.float32) * scale).asType(.int32)
    indices = clip(indices, min: 0, max: length - 1)
    return take(x, indices, axis: 2)
}

func lrSequenceMask(_ lengths: MLXArray, maxLength: Int) -> MLXArray {
    let seqRange = MLXArray(0 ..< maxLength)[.newAxis, 0...]
    return (seqRange .< lengths[0..., .newAxis]).asType(.float32)
}

/// Length regulator using nearest interpolation + conv-norm-act stack (donor InterpolateRegulator).
public final class InterpolateRegulator: Module {
    public let channels: Int
    public let isDiscrete: Bool
    public let interpolate: Bool

    @ModuleInfo(key: "convs") var convs: [Conv1d]
    @ModuleInfo(key: "norms") var norms: [LRGroupNorm]
    @ModuleInfo(key: "out_proj") var outProj: Conv1d
    // discrete-input path, present in the checkpoint but unused (is_discrete=false)
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "content_in_proj") var contentInProj: Linear
    @ParameterInfo(key: "mask_token") var maskToken: MLXArray  // unused at inference

    public init(
        channels: Int = 512, samplingRatios: [Int] = [1, 1, 1, 1], isDiscrete: Bool = false,
        inChannels: Int = 1024, codebookSize: Int = 2048, groups: Int = 1
    ) {
        self.channels = channels
        self.isDiscrete = isDiscrete
        self.interpolate = !samplingRatios.isEmpty

        self._convs.wrappedValue = samplingRatios.map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        }
        self._norms.wrappedValue = samplingRatios.map { _ in
            LRGroupNorm(numGroups: groups, numChannels: channels)
        }
        self._outProj.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self._embedding.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: channels)
        self._contentInProj.wrappedValue = Linear(inChannels, channels)
        self._maskToken.wrappedValue = MLXArray.zeros([1, channels])
    }

    /// x: (B, T, in_channels) continuous (this checkpoint), ylens: (B,) target lengths.
    /// Returns (B, max(ylens), channels).
    public func callAsFunction(_ x: MLXArray, ylens: MLXArray) -> MLXArray {
        var x = isDiscrete ? embedding(x) : contentInProj(x)  // (B, T, channels) NLC

        let maxLen = Int(ylens.max().item(Int32.self))

        x = x.transposed(0, 2, 1)  // NLC -> NCL
        if interpolate {
            x = interpolateNearest(x, targetLength: maxLen)
        }

        for i in 0 ..< convs.count {
            x = x.transposed(0, 2, 1)  // NCL -> NLC (MLX Conv1d is NLC)
            x = convs[i](x)
            x = x.transposed(0, 2, 1)  // NLC -> NCL
            x = norms[i](x)
            x = donorMish(x)
        }
        x = x.transposed(0, 2, 1)
        x = outProj(x)
        x = x.transposed(0, 2, 1)

        let out = x.transposed(0, 2, 1)  // NCL -> NLC
        let mask = lrSequenceMask(ylens, maxLength: maxLen)[0..., 0..., .newAxis]
        return out * mask
    }
}
