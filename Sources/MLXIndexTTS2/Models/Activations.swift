// Activations.swift — Snake activations + anti-aliased up/down sampling for BigVGAN v2.
//
// Isomorphic port of donor `mlx_indextts/models/activations.py`. The Kaiser-sinc low-pass
// filters are COMPUTED (never in the checkpoint) — UpSample1d/DownSample1d are plain
// non-Module classes so the filters stay out of the parameter contract (the donor equivalently
// underscore-prefixes `_filter`). Filter math replicated from numpy in Double, cast to fp32 at
// the end exactly like `np.kaiser(...)`→`astype(float32)`.
//
// Depthwise (groups=C, identical filter per channel) convolutions are implemented by folding
// channels into the batch axis — equivalent math, no groups dependency.

import Foundation
import MLX
import MLXNN

/// Modified Bessel function of the first kind, order 0 (series; converges for the beta range
/// used here, matches np.i0 to ~1e-15 relative).
private func besselI0(_ x: Double) -> Double {
    var sum = 1.0
    var term = 1.0
    var k = 1.0
    let halfX = x / 2
    while true {
        term *= (halfX / k) * (halfX / k)
        sum += term
        if term < sum * 1e-18 { break }
        k += 1
        if k > 200 { break }
    }
    return sum
}

/// Kaiser-windowed sinc low-pass filter (donor kaiser_sinc_filter1d), shape (kernelSize,).
func kaiserSincFilter1d(cutoff: Double, halfWidth: Double, kernelSize: Int) -> [Float] {
    let even = kernelSize % 2 == 0
    let halfSize = kernelSize / 2

    let deltaF = 4 * halfWidth
    let A = 2.285 * Double(halfSize - 1) * Double.pi * deltaF + 7.95
    let beta: Double
    if A > 50 {
        beta = 0.1102 * (A - 8.7)
    } else if A >= 21 {
        beta = 0.5842 * pow(A - 21, 0.4) + 0.07886 * (A - 21)
    } else {
        beta = 0
    }

    // np.kaiser(M, beta)
    let alpha = Double(kernelSize - 1) / 2
    let i0Beta = besselI0(beta)
    var window = [Double](repeating: 0, count: kernelSize)
    for n in 0 ..< kernelSize {
        let ratio = (Double(n) - alpha) / alpha
        window[n] = besselI0(beta * (1 - ratio * ratio).squareRoot()) / i0Beta
    }

    var time = [Double](repeating: 0, count: kernelSize)
    for n in 0 ..< kernelSize {
        time[n] = even ? Double(n - halfSize) + 0.5 : Double(n - halfSize)
    }

    func sinc(_ x: Double) -> Double {
        x == 0 ? 1.0 : sin(Double.pi * x) / (Double.pi * x)
    }

    var filter = [Double](repeating: 0, count: kernelSize)
    if cutoff != 0 {
        for n in 0 ..< kernelSize {
            filter[n] = 2 * cutoff * window[n] * sinc(2 * cutoff * time[n])
        }
        let total = filter.reduce(0, +)
        for n in 0 ..< kernelSize { filter[n] /= total }
    }
    return filter.map { Float($0) }
}

/// Edge/replicate-pad along an axis via explicit index gather (np.pad mode='edge').
func edgePadded1d(_ x: MLXArray, padLeft: Int, padRight: Int, axis: Int) -> MLXArray {
    if padLeft == 0 && padRight == 0 { return x }
    let L = x.dim(axis)
    var idx = [Int32](repeating: 0, count: padLeft)
    idx.append(contentsOf: (0 ..< L).map(Int32.init))
    idx.append(contentsOf: [Int32](repeating: Int32(L - 1), count: padRight))
    return take(x, MLXArray(idx), axis: axis)
}

/// Snake: x + (1/a)·sin²(a·x), NCL input (donor Snake).
public final class Snake: Module {
    public let alphaLogscale: Bool
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    public init(channels: Int, alphaLogscale: Bool = true) {
        self.alphaLogscale = alphaLogscale
        self._alpha.wrappedValue =
            alphaLogscale ? MLXArray.zeros([channels]) : MLXArray.ones([channels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var alpha = self.alpha[.newAxis, 0..., .newAxis]
        if alphaLogscale { alpha = exp(alpha) }
        let s = sin(alpha * x)
        return x + (1.0 / (alpha + 1e-9)) * (s * s)
    }
}

/// SnakeBeta: x + (1/b)·sin²(a·x), NCL input (donor SnakeBeta).
public final class SnakeBeta: Module {
    public let alphaLogscale: Bool
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    public var alphaValue: MLXArray { alpha }
    public var betaValue: MLXArray { beta }

    public init(channels: Int, alphaLogscale: Bool = true) {
        self.alphaLogscale = alphaLogscale
        // PITFALL (banked): alpha/beta must be DISTINCT MLXArray instances — sharing one init
        // array aliases the two parameters by object identity and update() then writes BOTH
        // keys into one array (last write wins; alpha silently ends up with beta's values).
        self._alpha.wrappedValue =
            alphaLogscale ? MLXArray.zeros([channels]) : MLXArray.ones([channels])
        self._beta.wrappedValue =
            alphaLogscale ? MLXArray.zeros([channels]) : MLXArray.ones([channels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var alpha = self.alpha[.newAxis, 0..., .newAxis]
        var beta = self.beta[.newAxis, 0..., .newAxis]
        if alphaLogscale {
            alpha = exp(alpha)
            beta = exp(beta)
        }
        let s = sin(alpha * x)
        return x + (1.0 / (beta + 1e-9)) * (s * s)
    }
}

/// Anti-aliased 2× upsampler (donor UpSample1d). Plain class — the filter is not a parameter.
public final class UpSample1d {
    let ratio: Int
    let kernelSize: Int
    let stride: Int
    let pad: Int
    let padLeft: Int
    let padRight: Int
    let filter: MLXArray  // (1, K, 1)

    public init(ratio: Int = 2, kernelSize: Int? = nil) {
        self.ratio = ratio
        self.kernelSize = kernelSize ?? (6 * ratio / 2) * 2
        self.stride = ratio
        self.pad = self.kernelSize / ratio - 1
        self.padLeft = pad * stride + (self.kernelSize - stride) / 2
        self.padRight = pad * stride + (self.kernelSize - stride + 1) / 2
        let f = kaiserSincFilter1d(
            cutoff: 0.5 / Double(ratio), halfWidth: 0.6 / Double(ratio),
            kernelSize: self.kernelSize)
        self.filter = MLXArray(f).reshaped(1, self.kernelSize, 1)
    }

    /// x: (B, C, L) NCL → (B, C, L·ratio).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, C, L) = (x.dim(0), x.dim(1), x.dim(2))
        var xp = edgePadded1d(x, padLeft: pad, padRight: pad, axis: 2)  // (B, C, Lp)
        let Lp = L + 2 * pad

        // depthwise transposed conv via channel-fold: identical filter per channel
        xp = xp.reshaped(B * C, Lp, 1)
        var out = convTransposed1d(xp, filter, stride: stride)  // (B*C, Lout, 1)
        out = out * Float(ratio)
        let Lout = out.dim(1)
        out = out.reshaped(B, C, Lout)
        return out[0..., 0..., padLeft ..< (Lout - padRight)]
    }
}

/// Anti-aliased 2× downsampler (donor DownSample1d). Plain class.
public final class DownSample1d {
    let ratio: Int
    let kernelSize: Int
    let padLeft: Int
    let padRight: Int
    let filter: MLXArray  // (1, K, 1)

    public init(ratio: Int = 2, kernelSize: Int? = nil) {
        self.ratio = ratio
        self.kernelSize = kernelSize ?? (6 * ratio / 2) * 2
        let even = self.kernelSize % 2 == 0
        self.padLeft = self.kernelSize / 2 - (even ? 1 : 0)
        self.padRight = self.kernelSize / 2
        let f = kaiserSincFilter1d(
            cutoff: 0.5 / Double(ratio), halfWidth: 0.6 / Double(ratio),
            kernelSize: self.kernelSize)
        self.filter = MLXArray(f).reshaped(1, self.kernelSize, 1)
    }

    /// x: (B, C, L) NCL → (B, C, L/ratio).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, C, L) = (x.dim(0), x.dim(1), x.dim(2))
        var xp = edgePadded1d(x, padLeft: padLeft, padRight: padRight, axis: 2)
        let Lp = L + padLeft + padRight

        xp = xp.reshaped(B * C, Lp, 1)
        let out = conv1d(xp, filter, stride: ratio)  // (B*C, Lout, 1)
        return out.reshaped(B, C, out.dim(1))
    }
}

/// Anti-aliased activation: upsample → act → downsample, NCL (donor Activation1d).
///
/// PITFALL (banked): the activation child must be a CONCRETE Module type. With an existential
/// (`@ModuleInfo var act: Module`) the key contract still enumerates `act.alpha`/`act.beta` and
/// `update(verify: .all)` passes — but the values are silently NOT written; the module keeps its
/// init-time parameters. This checkpoint is snakebeta-only, so `act` is typed SnakeBeta.
public final class Activation1d: Module {
    @ModuleInfo(key: "act") var act: SnakeBeta
    public let upsample: UpSample1d
    public let downsample: DownSample1d

    public var actModule: SnakeBeta { act }

    public func applyAct(_ x: MLXArray) -> MLXArray {
        act(x)
    }

    public init(
        activation: SnakeBeta, upRatio: Int = 2, downRatio: Int = 2,
        upKernelSize: Int = 12, downKernelSize: Int = 12
    ) {
        self._act.wrappedValue = activation
        self.upsample = UpSample1d(ratio: upRatio, kernelSize: upKernelSize)
        self.downsample = DownSample1d(ratio: downRatio, kernelSize: downKernelSize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downsample(applyAct(upsample(x)))
    }
}
