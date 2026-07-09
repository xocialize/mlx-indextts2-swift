// UnifiedVoiceV2+Generate.swift — the AR sampling loop (P7).
//
// Isomorphic port of gpt_v2.py `prepare_inputs` / `generate_step` /
// `_apply_repetition_penalty` / `_sample` plus the generate_v2.py AR driver
// (mel-start embedding, incremental KV-cache decode, stop-token check) and
// generate.py `compress_silence`.
//
// Deviations from the donor (behavior-preserving):
// - `_apply_repetition_penalty`'s per-token one-hot loop is a single vectorized
//   vocab-mask pass (identical math: >0 → /penalty, <0 → ×penalty on visited tokens).
// - `_sample`'s top-p un-sort scatter goes through the donor's numpy round-trip; here
//   it is native `putAlong` on the same sorted-removal flags (same result, no host trip).
// Sampling uses the global MLX RNG stream like the donor (`mx.random.categorical`), so
// seeded runs are stream-compatible with Python; token-exactness across backends is NOT
// a gate (exact-match ceiling — AR amplifies knife-edge flips).

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Compress consecutive silence tokens (mirrors generate.py `compress_silence`).
public func compressSilence(
    _ melCodes: [Int], silentToken: Int = 52, maxConsecutive: Int = 30, keep: Int = 10
) -> [Int] {
    let count = melCodes.lazy.filter { $0 == silentToken }.count
    if count <= maxConsecutive { return melCodes }

    var result: [Int] = []
    var consecutive = 0
    for code in melCodes {
        if code != silentToken {
            result.append(code)
            consecutive = 0
        } else if consecutive < keep {
            result.append(code)
            consecutive += 1
        }
        // else: skip excess silence token
    }
    return result
}

extension UnifiedVoiceV2 {

    /// Sampled-generation result: raw mel codes (pre-`compressSilence`) + stop flag.
    public struct GenerationResult {
        public let melCodes: [Int]
        public let stopped: Bool
    }

    /// Prepare the [conditioning, text] prefix embeddings (mirrors `prepare_inputs`).
    /// The returned all-ones mask matches the donor surface; the driver ignores it
    /// (GPT2Model builds its own causal mask).
    public func prepareInputs(
        conditioning: MLXArray, textTokens: MLXArray
    ) -> (MLXArray, MLXArray) {
        let batchSize = textTokens.dim(0)

        let startTokens = MLXArray.full([batchSize, 1], values: MLXArray(Int32(config.startTextToken)))
        let stopTokens = MLXArray.full([batchSize, 1], values: MLXArray(Int32(config.stopTextToken)))
        let text = concatenated([startTokens, textTokens.asType(.int32), stopTokens], axis: 1)

        var textEmb = textEmbedding(text)
        textEmb = textEmb + textPosEmbedding(textEmb)

        let emb = concatenated([conditioning, textEmb], axis: 1)
        let mask = MLXArray.ones([batchSize, emb.dim(1)])
        return (emb, mask)
    }

    /// One AR step: forward (with KV cache), final norm on the last position, mel head,
    /// sample (mirrors `generate_step`).
    public func generateStep(
        _ inputEmb: MLXArray,
        cache: [(MLXArray, MLXArray)]?,
        temperature: Float = 1.0,
        topK: Int = 30,
        topP: Float = 0.8,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int] = []
    ) -> (MLXArray, MLXArray, [(MLXArray, MLXArray)]) {
        let (hidden, newCache) = gpt(inputEmb, cache: cache)

        let last = hidden.dim(1) - 1
        let h = finalNorm(hidden[0..., last..., 0...])
        let logits = melHead(h)

        let nextToken = sample(
            logits[0..., 0, 0...], temperature: temperature, topK: topK, topP: topP,
            repetitionPenalty: repetitionPenalty, generatedTokens: generatedTokens)

        return (nextToken, logits, newCache)
    }

    /// Repetition penalty (mirrors `_apply_repetition_penalty`; vectorized, same math).
    public func applyRepetitionPenalty(
        _ logits: MLXArray, generatedTokens: [Int], penalty: Float
    ) -> MLXArray {
        if penalty == 1.0 || generatedTokens.isEmpty { return logits }

        let vocabSize = logits.dim(-1)
        var flags = [Float](repeating: 0, count: vocabSize)
        for tokenId in generatedTokens where tokenId >= 0 && tokenId < vocabSize {
            flags[tokenId] = 1
        }
        let oneHot = MLXArray(flags).reshaped(1, vocabSize)

        let penalized = MLX.where(logits .> 0, logits / penalty, logits * penalty)
        return logits * (1 - oneHot) + penalized * oneHot
    }

    /// Sample from logits with repetition penalty + temperature + top-k + top-p
    /// (mirrors `_sample`). `temperature == 0` = greedy argmax (after penalty).
    public func sample(
        _ logits: MLXArray,
        temperature: Float = 1.0,
        topK: Int = 30,
        topP: Float = 0.8,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int] = []
    ) -> MLXArray {
        var logits = logits

        // Apply repetition penalty first (before temperature)
        if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
            logits = applyRepetitionPenalty(
                logits, generatedTokens: generatedTokens, penalty: repetitionPenalty)
        }

        if temperature == 0 {
            return argMax(logits, axis: -1)
        }

        logits = logits / temperature

        // Top-k filtering (top() slice [:, :1] = the k-th largest, like mx.topk)
        if topK > 0 {
            let k = min(topK, logits.dim(-1))
            let topKValues = top(logits, k: k, axis: -1)
            let threshold = topKValues[0..., ..<1]
            logits = MLX.where(logits .< threshold, MLXArray(-Float.infinity), logits)
        }

        // Top-p (nucleus) filtering
        if topP < 1.0 {
            let sortedIndices = argSort(-logits, axis: -1)  // descending
            let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
            let cumulativeProbs = cumsum(softmax(sortedLogits, axis: -1), axis: -1)

            var sortedRemove = cumulativeProbs .> topP
            let lastCol = sortedRemove.dim(-1) - 1
            let firstCol = MLXArray.zeros([sortedRemove.dim(0), 1], type: Bool.self)
            sortedRemove = concatenated([firstCol, sortedRemove[0..., ..<lastCol]], axis: -1)

            let removeMask = putAlong(
                MLXArray.zeros(logits.shape, type: Bool.self),
                sortedIndices, values: sortedRemove, axis: -1)
            logits = MLX.where(removeMask, MLXArray(-Float.infinity), logits)
        }

        let probs = softmax(logits, axis: -1)
        return categorical(log(probs + 1e-10))
    }

    /// Autoregressive mel-code generation (mirrors the generate_v2.py AR driver:
    /// [conditioning, text, mel-start] prefix, then incremental single-token decode with
    /// KV cache and per-step positional embedding; stops on `stopMelToken`).
    ///
    /// Returns RAW codes — callers apply `compressSilence` as generate_v2 does.
    /// `stepLogitsHook` receives the raw (1, 1, vocab) logits per step (parity gates).
    public func generateMelCodes(
        conditioning: MLXArray,
        textTokens: MLXArray,
        maxMelTokens: Int = 1500,
        temperature: Float = 0.8,
        topK: Int = 30,
        topP: Float = 0.8,
        repetitionPenalty: Float = 10.0,
        stepLogitsHook: ((Int, MLXArray) -> Void)? = nil
    ) -> GenerationResult {
        var (inputEmb, _) = prepareInputs(conditioning: conditioning, textTokens: textTokens)

        // Add start mel token
        let melStart = MLXArray.full([1, 1], values: MLXArray(Int32(config.startMelToken)))
        var melStartEmb = melEmbedding(melStart)
        melStartEmb = melStartEmb + melPosEmbedding.getFixedEmbedding(0)
        inputEmb = concatenated([inputEmb, melStartEmb], axis: 1)

        var melCodes: [Int] = []
        var cache: [(MLXArray, MLXArray)]? = nil
        var stopped = false

        for i in 0 ..< maxMelTokens {
            let step: (MLXArray, MLXArray, [(MLXArray, MLXArray)])

            if cache == nil {
                step = generateStep(
                    inputEmb, cache: nil, temperature: temperature, topK: topK, topP: topP,
                    repetitionPenalty: repetitionPenalty, generatedTokens: melCodes)
            } else {
                let lastToken = MLXArray.full([1, 1], values: MLXArray(Int32(melCodes.last!)))
                var lastEmb = melEmbedding(lastToken)
                let melPos = melCodes.count + 1
                lastEmb = lastEmb + melPosEmbedding.getFixedEmbedding(melPos)
                step = generateStep(
                    lastEmb, cache: cache, temperature: temperature, topK: topK, topP: topP,
                    repetitionPenalty: repetitionPenalty, generatedTokens: melCodes)
            }
            let (nextToken, logits) = (step.0, step.1)
            cache = step.2

            stepLogitsHook?(i, logits)

            let tokenId = Int(nextToken[0].asType(.int32).item(Int32.self))
            if tokenId == config.stopMelToken {
                stopped = true
                break
            }
            melCodes.append(tokenId)
            if let cache {
                for (k, v) in cache { eval(k, v) }
            }
        }

        return GenerationResult(melCodes: melCodes, stopped: stopped)
    }
}
