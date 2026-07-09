// WaveNet.swift — WaveNet-style final layer for the S2Mel DiT.
//
// Isomorphic port of donor `mlx_indextts/models/s2mel/wavenet.py` (WN + SConv1d).
// NCL convention throughout, matching the donor.
//
// Donor quirk carried verbatim: SConv1d does its own SYMMETRIC REFLECT padding in the
// forward (the donor deviates from torch's zero-padded Conv1d here — but the donor produced
// the goldens, so the donor wins). This checkpoint uses dilation_rate=1 everywhere, so the
// donor's manual kernel-dilation path never fires; Swift passes dilation to Conv1d natively.

import Foundation
import MLX
import MLXNN

/// Reflect-pad along an axis via explicit index gather (exact match to np.pad reflect).
func reflectPadded1d(_ x: MLXArray, padLeft: Int, padRight: Int, axis: Int) -> MLXArray {
    if padLeft == 0 && padRight == 0 { return x }
    let L = x.dim(axis)
    var idx: [Int32] = []
    idx.reserveCapacity(padLeft + L + padRight)
    for i in 0 ..< padLeft { idx.append(Int32(padLeft - i)) }
    for i in 0 ..< L { idx.append(Int32(i)) }
    for i in 0 ..< padRight { idx.append(Int32(L - 2 - i)) }
    return take(x, MLXArray(idx), axis: axis)
}

/// Conv1d with symmetric reflect padding, NCL in/out (donor SConv1d).
public final class SConv1d: Module {
    public let kernelSize: Int
    public let dilation: Int
    public let stride: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, dilation: Int = 1
    ) {
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.stride = stride
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize,
            stride: stride, padding: 0, dilation: dilation)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, L) NCL
        var x = x.transposed(0, 2, 1)  // NCL -> NLC

        let effectiveKernel = (kernelSize - 1) * dilation + 1
        let totalPadding = effectiveKernel - 1
        let padLeft = totalPadding / 2
        let padRight = totalPadding - padLeft
        x = reflectPadded1d(x, padLeft: padLeft, padRight: padRight, axis: 1)

        return conv(x).transposed(0, 2, 1)  // NLC -> NCL
    }
}

/// Gated activation: tanh(a) * sigmoid(b) over channel halves (NCL).
func fusedAddTanhSigmoidMultiply(
    _ inputA: MLXArray, _ inputB: MLXArray, nChannels: Int
) -> MLXArray {
    let inAct = inputA + inputB
    let tAct = tanh(inAct[0..., ..<nChannels, 0...])
    let sAct = sigmoid(inAct[0..., nChannels..., 0...])
    return tAct * sAct
}

/// WaveNet module with dilated convolutions and gated activations (donor WN).
/// Inference-only: dropout is a no-op.
public final class WN: Module {
    public let hiddenChannels: Int
    public let nLayers: Int

    @ModuleInfo(key: "cond_layer") var condLayer: SConv1d
    @ModuleInfo(key: "in_layers") var inLayers: [SConv1d]
    @ModuleInfo(key: "res_skip_layers") var resSkipLayers: [SConv1d]

    public init(
        hiddenChannels: Int = 512, kernelSize: Int = 5, dilationRate: Int = 1,
        nLayers: Int = 8, ginChannels: Int = 512
    ) {
        precondition(kernelSize % 2 == 1, "kernel_size must be odd")
        self.hiddenChannels = hiddenChannels
        self.nLayers = nLayers

        self._condLayer.wrappedValue = SConv1d(
            inChannels: ginChannels, outChannels: 2 * hiddenChannels * nLayers, kernelSize: 1)

        var inLayers: [SConv1d] = []
        var resSkipLayers: [SConv1d] = []
        for i in 0 ..< nLayers {
            var dilation = 1
            for _ in 0 ..< i { dilation *= dilationRate }
            inLayers.append(SConv1d(
                inChannels: hiddenChannels, outChannels: 2 * hiddenChannels,
                kernelSize: kernelSize, dilation: dilation))
            let resSkipChannels = i < nLayers - 1 ? 2 * hiddenChannels : hiddenChannels
            resSkipLayers.append(SConv1d(
                inChannels: hiddenChannels, outChannels: resSkipChannels, kernelSize: 1))
        }
        self._inLayers.wrappedValue = inLayers
        self._resSkipLayers.wrappedValue = resSkipLayers
    }

    /// x: (B, C, L) NCL, xMask: (B, 1, L), g: (B, gin, 1) or (B, gin).
    public func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?) -> MLXArray {
        var x = x
        var output = MLXArray.zeros(like: x)

        var gCond: MLXArray? = nil
        if var g = g {
            if g.ndim == 2 { g = g[0..., 0..., .newAxis] }
            gCond = condLayer(g)  // (B, 2*hidden*n_layers, 1)
        }

        for i in 0 ..< nLayers {
            let xIn = inLayers[i](x)  // (B, 2*hidden, L)

            let gL: MLXArray
            if let gCond {
                let condOffset = i * 2 * hiddenChannels
                gL = gCond[0..., condOffset ..< (condOffset + 2 * hiddenChannels), 0...]
            } else {
                gL = MLXArray.zeros(like: xIn)
            }

            let acts = fusedAddTanhSigmoidMultiply(xIn, gL, nChannels: hiddenChannels)
            let resSkipActs = resSkipLayers[i](acts)

            if i < nLayers - 1 {
                let resActs = resSkipActs[0..., ..<hiddenChannels, 0...]
                x = (x + resActs) * xMask
                output = output + resSkipActs[0..., hiddenChannels..., 0...]
            } else {
                output = output + resSkipActs
            }
        }

        return output * xMask
    }
}
