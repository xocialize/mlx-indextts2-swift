// BigVGANV2.swift — BigVGAN v2 vocoder (nvidia/bigvgan_v2_22khz_80band_256x weights).
//
// Isomorphic port of donor `mlx_indextts/models/bigvgan_v2.py`. Pure mel→audio, no speaker
// conditioning. v2 specifics: use_tanh_at_final=False (final = clip to [-1,1]),
// use_bias_at_final=False (conv_post has NO bias), SnakeBeta activations wrapped in
// anti-aliased Activation1d.
//
// bigvgan.safetensors (449 keys) is already donor-MLX layout with weight norms pre-fused —
// keys map 1:1 (ups.N.*, resblocks.N.convs1/2.M.*, resblocks.N.activations.M.act.alpha/beta,
// conv_pre.*, conv_post.weight, activation_post.act.*). No sanitize needed.

import Foundation
import MLX
import MLXNN

public struct BigVGANV2Config {
    public var numMels = 80
    public var upsampleRates = [4, 4, 2, 2, 2, 2]
    public var upsampleKernelSizes = [8, 8, 4, 4, 4, 4]
    public var upsampleInitialChannel = 1536
    public var resblockKernelSizes = [3, 7, 11]
    public var resblockDilationSizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var activation = "snakebeta"
    public var snakeLogscale = true
    public var useTanhAtFinal = false
    public var useBiasAtFinal = false
    public var resblock = "1"

    public init() {}
}

func getPadding(kernelSize: Int, dilation: Int = 1) -> Int {
    (kernelSize * dilation - dilation) / 2
}

// This checkpoint uses snakebeta everywhere; Snake exists for donor isomorphism but a
// snake-configured Activation1d would need its own concrete-typed variant (see the
// existential-Module pitfall in Activations.swift).
func makeActivation(_ name: String, channels: Int, alphaLogscale: Bool) -> SnakeBeta {
    precondition(name == "snakebeta", "only snakebeta is wired (checkpoint uses snakebeta)")
    return SnakeBeta(channels: channels, alphaLogscale: alphaLogscale)
}

/// Anti-aliased Multi-Periodicity block, type 1 (donor AMPBlock1). NCL in/out.
public final class AMPBlock1: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "activations") var activations: [Activation1d]

    public var activationModules: [Activation1d] { activations }

    public init(
        channels: Int, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5],
        activation: String = "snakebeta", alphaLogscale: Bool = true
    ) {
        self._convs1.wrappedValue = dilations.map { d in
            Conv1d(
                inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                padding: getPadding(kernelSize: kernelSize, dilation: d), dilation: d)
        }
        self._convs2.wrappedValue = dilations.map { _ in
            Conv1d(
                inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                padding: getPadding(kernelSize: kernelSize, dilation: 1), dilation: 1)
        }
        self._activations.wrappedValue = (0 ..< 2 * dilations.count).map { _ in
            Activation1d(activation: makeActivation(
                activation, channels: channels, alphaLogscale: alphaLogscale))
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for i in 0 ..< convs1.count {
            let a1 = activations[2 * i]
            let a2 = activations[2 * i + 1]

            var xt = a1(x)  // NCL
            xt = convs1[i](xt.transposed(0, 2, 1)).transposed(0, 2, 1)
            xt = a2(xt)
            xt = convs2[i](xt.transposed(0, 2, 1)).transposed(0, 2, 1)
            x = xt + x
        }
        return x
    }
}

/// Anti-aliased Multi-Periodicity block, type 2 (donor AMPBlock2; unused by this checkpoint).
public final class AMPBlock2: Module {
    @ModuleInfo(key: "convs") var convs: [Conv1d]
    @ModuleInfo(key: "activations") var activations: [Activation1d]

    public init(
        channels: Int, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5],
        activation: String = "snakebeta", alphaLogscale: Bool = true
    ) {
        self._convs.wrappedValue = dilations.map { d in
            Conv1d(
                inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                padding: getPadding(kernelSize: kernelSize, dilation: d), dilation: d)
        }
        self._activations.wrappedValue = dilations.map { _ in
            Activation1d(activation: makeActivation(
                activation, channels: channels, alphaLogscale: alphaLogscale))
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for i in 0 ..< convs.count {
            var xt = activations[i](x)
            xt = convs[i](xt.transposed(0, 2, 1)).transposed(0, 2, 1)
            x = xt + x
        }
        return x
    }
}

/// BigVGAN v2 vocoder. Input mel (B, n_mels, T) NCL → audio (B, 1, T·256) NCL.
public final class BigVGANV2: Module {
    public let config: BigVGANV2Config
    public let numKernels: Int
    public let numUpsamples: Int

    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resblocks: [AMPBlock1]
    @ModuleInfo(key: "activation_post") var activationPost: Activation1d
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    // Ladder access for the parity gate.
    public var convPreLayer: Conv1d { convPre }
    public var upsLayers: [ConvTransposed1d] { ups }
    public var resblockModules: [AMPBlock1] { resblocks }
    public var activationPostModule: Activation1d { activationPost }
    public var convPostLayer: Conv1d { convPost }

    public init(_ config: BigVGANV2Config = BigVGANV2Config()) {
        precondition(config.resblock == "1", "this checkpoint uses AMPBlock1")
        self.config = config
        self.numKernels = config.resblockKernelSizes.count
        self.numUpsamples = config.upsampleRates.count

        self._convPre.wrappedValue = Conv1d(
            inputChannels: config.numMels, outputChannels: config.upsampleInitialChannel,
            kernelSize: 7, padding: 3)

        var upsList: [ConvTransposed1d] = []
        var ch = config.upsampleInitialChannel
        for (rate, kernel) in zip(config.upsampleRates, config.upsampleKernelSizes) {
            let outCh = ch / 2
            upsList.append(ConvTransposed1d(
                inputChannels: ch, outputChannels: outCh, kernelSize: kernel,
                stride: rate, padding: (kernel - rate) / 2))
            ch = outCh
        }
        self._ups.wrappedValue = upsList

        var resblocksList: [AMPBlock1] = []
        ch = config.upsampleInitialChannel
        for _ in 0 ..< numUpsamples {
            ch = ch / 2
            for (k, d) in zip(config.resblockKernelSizes, config.resblockDilationSizes) {
                resblocksList.append(AMPBlock1(
                    channels: ch, kernelSize: k, dilations: d,
                    activation: config.activation, alphaLogscale: config.snakeLogscale))
            }
        }
        self._resblocks.wrappedValue = resblocksList

        self._activationPost.wrappedValue = Activation1d(activation: makeActivation(
            config.activation, channels: ch, alphaLogscale: config.snakeLogscale))
        self._convPost.wrappedValue = Conv1d(
            inputChannels: ch, outputChannels: 1, kernelSize: 7, padding: 3,
            bias: config.useBiasAtFinal)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convPre(x.transposed(0, 2, 1)).transposed(0, 2, 1)  // NCL → conv NLC → NCL

        for i in 0 ..< numUpsamples {
            x = ups[i](x.transposed(0, 2, 1)).transposed(0, 2, 1)

            var xs: MLXArray? = nil
            for j in 0 ..< numKernels {
                let res = resblocks[i * numKernels + j](x)
                xs = xs.map { $0 + res } ?? res
            }
            x = xs! / Float(numKernels)
        }

        x = activationPost(x)
        x = convPost(x.transposed(0, 2, 1)).transposed(0, 2, 1)

        if config.useTanhAtFinal {
            x = tanh(x)
        } else {
            x = clip(x, min: -1.0, max: 1.0)
        }
        return x
    }
}
