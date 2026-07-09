// CFM.swift — Conditional Flow Matching (Euler ODE) for S2Mel.
//
// Isomorphic port of donor `mlx_indextts/models/s2mel/cfm.py`. Time runs 0→1 with an
// additive Euler step (NOTE: the opposite convention from VoxCPM's UnifiedCFM — only the
// loop idioms are lifted from there, the math follows this donor).
//
// Donor behaviors carried verbatim:
// - The prompt region of x is zeroed before the loop and re-zeroed after each step, but the
//   RETURNED mel is the last appended step BEFORE re-zeroing (callers trim the prompt region).
// - CFG stacks [cond, uncond] into one batch-2 estimator call; uncond zeroes prompt/style/mu.
// - Initial noise: `z = normal(B, in_channels, T) * temperature` — the FIRST draw after
//   `MLXRandom.seed(42)` in the replay-golden convention (seed streams are bit-identical
//   across MLX bindings); `inference(injectedZ:)` lets the gate bypass RNG entirely.

import Foundation
import MLX
import MLXNN
import MLXRandom

public final class CFM: Module {
    public let sigmaMin: Float = 1e-6
    public let inChannels: Int
    public let zeroPromptSpeechToken: Bool

    @ModuleInfo(key: "estimator") var estimator: DiT

    public init(
        inChannels: Int = 80, hiddenDim: Int = 512, numHeads: Int = 8, depth: Int = 13,
        contentDim: Int = 512, styleDim: Int = 192,
        longSkipConnection: Bool = true, uvitSkipConnection: Bool = true,
        timeAsToken: Bool = false, styleAsToken: Bool = false, styleCondition: Bool = true,
        wavenetHiddenDim: Int = 512, wavenetNumLayers: Int = 8,
        wavenetKernelSize: Int = 5, wavenetDilationRate: Int = 1,
        zeroPromptSpeechToken: Bool = false
    ) {
        self.inChannels = inChannels
        self.zeroPromptSpeechToken = zeroPromptSpeechToken
        self._estimator.wrappedValue = DiT(
            hiddenDim: hiddenDim, numHeads: numHeads, depth: depth, inChannels: inChannels,
            contentDim: contentDim, styleDim: styleDim,
            longSkipConnection: longSkipConnection, uvitSkipConnection: uvitSkipConnection,
            timeAsToken: timeAsToken, styleAsToken: styleAsToken, styleCondition: styleCondition,
            wavenetHiddenDim: wavenetHiddenDim, wavenetNumLayers: wavenetNumLayers,
            wavenetKernelSize: wavenetKernelSize, wavenetDilationRate: wavenetDilationRate)
    }

    public var estimatorModule: DiT { estimator }

    /// mu: (B, T, content_dim); xLens: (B,); prompt: (B, in, prompt_len); style: (B, style_dim).
    /// Returns (B, in_channels, T). `injectedZ` bypasses the internal noise draw (parity gates).
    public func inference(
        mu: MLXArray, xLens: MLXArray, prompt: MLXArray, style: MLXArray,
        nTimesteps: Int, temperature: Float = 1.0, inferenceCfgRate: Float = 0.7,
        injectedZ: MLXArray? = nil
    ) -> MLXArray {
        let (B, T) = (mu.dim(0), mu.dim(1))
        let z = (injectedZ ?? MLXRandom.normal([B, inChannels, T])) * temperature
        let tSpan = linspace(Float(0), Float(1), count: nTimesteps + 1)
        return solveEuler(
            x: z, xLens: xLens, prompt: prompt, mu: mu, style: style,
            tSpan: tSpan, inferenceCfgRate: inferenceCfgRate)
    }

    public func solveEuler(
        x: MLXArray, xLens: MLXArray, prompt: MLXArray, mu: MLXArray, style: MLXArray,
        tSpan: MLXArray, inferenceCfgRate: Float = 0.5
    ) -> MLXArray {
        let T = x.dim(2)
        let promptLen = prompt.dim(2)
        let nSteps = tSpan.dim(0) - 1

        eval(tSpan)
        var tValues: [Float] = []
        for i in 0 ... nSteps { tValues.append(tSpan[i].item(Float.self)) }

        // prompt_x: prompt in the first prompt_len frames, zeros after
        let promptX = concatenated([
            prompt[0..., 0..., ..<promptLen],
            MLXArray.zeros([x.dim(0), x.dim(1), T - promptLen]),
        ], axis: 2)

        // zero the prompt region of the noise
        var x = concatenated([
            MLXArray.zeros([x.dim(0), x.dim(1), promptLen]),
            x[0..., 0..., promptLen...],
        ], axis: 2)

        var mu = mu
        if zeroPromptSpeechToken {
            mu = concatenated([
                MLXArray.zeros([mu.dim(0), promptLen, mu.dim(2)]),
                mu[0..., promptLen..., 0...],
            ], axis: 1)
        }

        var t = tValues[0]
        var dt = tValues[1] - tValues[0]
        var lastAppended = x

        for step in 1 ... nSteps {
            dt = tValues[step] - tValues[step - 1]

            var dphiDt: MLXArray
            if inferenceCfgRate > 0 {
                let stackedPromptX = concatenated([promptX, MLXArray.zeros(like: promptX)], axis: 0)
                let stackedStyle = concatenated([style, MLXArray.zeros(like: style)], axis: 0)
                let stackedMu = concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
                let stackedX = concatenated([x, x], axis: 0)
                let stackedT = MLXArray([t, t])

                let stackedDphiDt = estimator(
                    stackedX, promptX: stackedPromptX, xLens: xLens, t: stackedT,
                    style: stackedStyle, cond: stackedMu)

                let parts = split(stackedDphiDt, parts: 2, axis: 0)
                dphiDt = (1.0 + inferenceCfgRate) * parts[0] - inferenceCfgRate * parts[1]
            } else {
                dphiDt = estimator(
                    x, promptX: promptX, xLens: xLens, t: MLXArray([t]),
                    style: style, cond: mu)
            }

            x = x + dt * dphiDt
            t = t + dt
            lastAppended = x

            if step < nSteps {
                dt = tValues[step + 1] - t
            }

            // keep the prompt region zero for the next step
            x = concatenated([
                MLXArray.zeros([x.dim(0), x.dim(1), promptLen]),
                x[0..., 0..., promptLen...],
            ], axis: 2)

            eval(x)
        }

        return lastAppended
    }
}
