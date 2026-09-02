// UnifiedVoiceV25.swift — the IndexTTS-2.5 GPT AR model (`indextts/gpt/model_v2_5.py`
// UnifiedVoice; donor mlx_indextts/models/gpt_v25.py).
//
// Versus 2.0: the speaker Conformer + Perceiver resampler and the speed embedding are gone.
// Speaker conditioning is ONE token — `spk_emb_proj` Linear(192→1280) over the CAMPPlus
// embedding — added to the emotion vector, followed by two zero rows (conds = 3 tokens). Text
// embeddings gain a learned per-utterance `lang_embedding` (107 rows: the 106-entry Whisper
// language table + 1) added to every text position. The emotion conditioner (Conformer +
// 1-latent Perceiver → emovec_layer → emo_layer) is unchanged from 2.0.
//
// The GPT latent → S2Mel branch (`forward_latent` + s2mel gpt_layer) is not part of 2.5
// inference (the codec decoder alone feeds the length regulator), so it is not carried.
//
// Conditioner config = checkpoint config.yaml truths: emo = Conformer(1024→512, ff 1024,
// 4 heads, 4 blocks) + Perceiver(1024, ctx 512, 1 latent, 4 heads, mult 2).

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Learned position embedding (upstream stores it as `<name>.emb.weight`).
public final class LearnedPositionEmbedding: Module {
    @ModuleInfo(key: "emb") public var emb: Embedding

    public init(maxSeqLen: Int, dim: Int) {
        self._emb.wrappedValue = Embedding(embeddingCount: maxSeqLen, dimensions: dim)
    }

    /// Position embeddings for a (B, L, D) or (B, L) input.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.ndim >= 2 ? x.dim(1) : x.dim(0)
        return emb(MLXArray(0..<Int32(seqLen)))
    }

    public func getFixedEmbedding(_ position: Int) -> MLXArray {
        emb(MLXArray([Int32(position)])).expandedDimensions(axis: 0)
    }
}

/// Resolved GPT config (values = the 2.5 checkpoint's config.yaml).
public struct GPTV25Config: Sendable {
    public var modelDim = 1280
    public var heads = 20
    public var layers = 24
    public var maxMelTokens = 1815
    public var maxTextTokens = 600
    public var numberTextTokens = 60509
    public var numberMelCodes = 8194
    public var startMelToken = 8192
    public var stopMelToken = 8193
    public var startTextToken = 0
    public var stopTextToken = 1
    public var speakerEmbeddingDim = 192
    /// `len(LANGUAGE_DICT) + 1`.
    public var numberLanguages = TiktokenBPE.languageCodes.count + 1

    public init() {}
}

public final class UnifiedVoiceV25: Module {
    public let config: GPTV25Config

    @ModuleInfo(key: "emo_conditioning_encoder") public var emoConditioningEncoder: ConformerEncoder
    @ModuleInfo(key: "emo_perceiver_encoder") public var emoPerceiverEncoder: PerceiverResampler
    @ModuleInfo(key: "emo_layer") public var emoLayer: Linear
    @ModuleInfo(key: "emovec_layer") public var emovecLayer: Linear
    @ModuleInfo(key: "spk_emb_proj") public var spkEmbProj: Linear
    @ModuleInfo(key: "lang_embedding") public var langEmbedding: Embedding
    @ModuleInfo(key: "text_embedding") public var textEmbedding: Embedding
    @ModuleInfo(key: "mel_embedding") public var melEmbedding: Embedding
    @ModuleInfo(key: "mel_pos_embedding") public var melPosEmbedding: LearnedPositionEmbedding
    @ModuleInfo(key: "text_pos_embedding") public var textPosEmbedding: LearnedPositionEmbedding
    @ModuleInfo(key: "gpt") public var gpt: GPT2Model
    @ModuleInfo(key: "final_norm") public var finalNorm: LayerNorm
    @ModuleInfo(key: "text_head") public var textHead: Linear
    @ModuleInfo(key: "mel_head") public var melHead: Linear

    public init(config: GPTV25Config = GPTV25Config()) {
        self.config = config

        let emoCondConfig = ConformerConfig(
            inputSize: 1024, outputSize: 512, linearUnits: 1024, attentionHeads: 4, numBlocks: 4)
        self._emoConditioningEncoder.wrappedValue = ConformerEncoder(emoCondConfig)
        self._emoPerceiverEncoder.wrappedValue = PerceiverResampler(
            dim: 1024, nDimContext: emoCondConfig.outputSize,
            nLatents: 1, nHeads: emoCondConfig.attentionHeads, nFFMult: 2)
        self._emoLayer.wrappedValue = Linear(config.modelDim, config.modelDim)
        self._emovecLayer.wrappedValue = Linear(1024, config.modelDim)
        self._spkEmbProj.wrappedValue = Linear(config.speakerEmbeddingDim, config.modelDim)
        self._langEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numberLanguages, dimensions: config.modelDim)
        self._textEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numberTextTokens + 1, dimensions: config.modelDim)
        self._melEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numberMelCodes, dimensions: config.modelDim)
        self._melPosEmbedding.wrappedValue = LearnedPositionEmbedding(
            maxSeqLen: config.maxMelTokens + 2 + 1, dim: config.modelDim)
        self._textPosEmbedding.wrappedValue = LearnedPositionEmbedding(
            maxSeqLen: config.maxTextTokens + 2, dim: config.modelDim)
        self._gpt.wrappedValue = GPT2Model(
            dim: config.modelDim, numHeads: config.heads, numLayers: config.layers)
        self._finalNorm.wrappedValue = LayerNorm(dimensions: config.modelDim)
        self._textHead.wrappedValue = Linear(config.modelDim, config.numberTextTokens + 1)
        self._melHead.wrappedValue = Linear(config.modelDim, config.numberMelCodes)
    }

    // MARK: - Conditioning

    /// Emotion conditioning (`get_emo_conditioning`): (B, 1024, T) NCL → (B, 1024).
    public func getEmoConditioning(_ emoConditioningInput: MLXArray) -> MLXArray {
        let x = emoConditioningInput.transposed(0, 2, 1)
        return emoPerceiverEncoder(emoConditioningEncoder(x)).squeezed(axis: 1)
    }

    /// Emotion vector (`get_emovec`): (B, 1024, T) NCL → (B, modelDim).
    public func getEmovec(_ emoConditioningInput: MLXArray) -> MLXArray {
        emoLayer(emovecLayer(getEmoConditioning(emoConditioningInput)))
    }

    /// `[spk_emb_proj(style) + emo_vec, 0, 0]` (mirrors 2.5 `prepare_conditioning_latents`).
    /// - style: (B, 192) CAMPPlus embedding; emoVec: (B, modelDim). Returns (B, 3, modelDim).
    public func prepareConditioningLatents(style: MLXArray, emoVec: MLXArray) -> MLXArray {
        let speaker = spkEmbProj(style).expandedDimensions(axis: 1)          // (B, 1, D)
        let conditioned = speaker + emoVec.expandedDimensions(axis: 1)
        let zeros = MLXArray.zeros([conditioned.dim(0), 2, config.modelDim]).asType(conditioned.dtype)
        return concatenated([conditioned, zeros], axis: 1)
    }

    /// `[conditioning, text]` prefix embeddings (mirrors `prepare_inputs` / the official
    /// `prepare_gpt_inputs`): start/stop ids are stripped from `textTokens` and one canonical
    /// pair re-added; every text position gets `lang_embedding[languageID]`. The official
    /// generator left-pads the batch by one masked row here; with a fully-masked key that row
    /// contributes nothing, so this port omits it (no padding, no mask).
    public func prepareInputs(
        conditioning: MLXArray, textTokens: [Int], languageID: Int
    ) -> MLXArray {
        let body = textTokens.filter { $0 != config.startTextToken && $0 != config.stopTextToken }
        let ids = [config.startTextToken] + body + [config.stopTextToken]
        let text = MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0)   // (1, L)
        var textEmb = textEmbedding(text)
        textEmb = textEmb + textPosEmbedding(textEmb)
        textEmb = textEmb + langEmbedding(MLXArray([Int32(languageID)]))        // (1, D) broadcast
        return concatenated([conditioning, textEmb], axis: 1)
    }

    /// gpt.safetensors keys → this module's tree: only the emotion Perceiver's ModuleList-of-
    /// pairs (`layers.N.0.*` / `layers.N.1.*` → `layers.N.attn.*` / `layers.N.ff.*`) needs a remap
    /// (numeric module keys collide with array-index unflattening).
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            var nk = k
            if nk.contains("perceiver_encoder.layers.") {
                if let range = nk.range(of: #"(layers\.\d+)\.0\."#, options: .regularExpression) {
                    nk = nk.replacingCharacters(in: range, with: String(nk[range].dropLast(3)) + ".attn.")
                } else if let range = nk.range(of: #"(layers\.\d+)\.1\."#, options: .regularExpression) {
                    nk = nk.replacingCharacters(in: range, with: String(nk[range].dropLast(3)) + ".ff.")
                }
            }
            out[nk] = v
        }
        return out
    }
}

// MARK: - Sampling / AR driver (unchanged from the 2.0 port; `_sample` + generate loop)

/// Compress consecutive silence tokens (mirrors `compress_silence`).
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
    }
    return result
}

extension UnifiedVoiceV25 {

    public struct GenerationResult {
        public let melCodes: [Int]
        public let stopped: Bool
    }

    /// One AR step: forward (KV cache), final norm on the last position, mel head, sample.
    /// Returns (nextToken, raw logits (1, 1, vocab), cache).
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
        let logits = melHead(finalNorm(hidden[0..., last..., 0...]))
        let nextToken = sample(
            logits[0..., 0, 0...], temperature: temperature, topK: topK, topP: topP,
            repetitionPenalty: repetitionPenalty, generatedTokens: generatedTokens)
        return (nextToken, logits, newCache)
    }

    /// Repetition penalty (`_apply_repetition_penalty`; vectorized, same math).
    public func applyRepetitionPenalty(
        _ logits: MLXArray, generatedTokens: [Int], penalty: Float
    ) -> MLXArray {
        if penalty == 1.0 || generatedTokens.isEmpty { return logits }
        let vocabSize = logits.dim(-1)
        var flags = [Float](repeating: 0, count: vocabSize)
        for tokenId in generatedTokens where tokenId >= 0 && tokenId < vocabSize { flags[tokenId] = 1 }
        let oneHot = MLXArray(flags).reshaped(1, vocabSize)
        let penalized = MLX.where(logits .> 0, logits / penalty, logits * penalty)
        return logits * (1 - oneHot) + penalized * oneHot
    }

    /// `_sample`: repetition penalty → temperature → top-k → top-p → categorical.
    /// `temperature == 0` = greedy argmax (after penalty).
    public func sample(
        _ logits: MLXArray,
        temperature: Float = 1.0,
        topK: Int = 30,
        topP: Float = 0.8,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int] = []
    ) -> MLXArray {
        var logits = logits
        if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
            logits = applyRepetitionPenalty(logits, generatedTokens: generatedTokens, penalty: repetitionPenalty)
        }
        if temperature == 0 { return argMax(logits, axis: -1) }
        logits = logits / temperature
        if topK > 0 {
            let k = min(topK, logits.dim(-1))
            let threshold = top(logits, k: k, axis: -1)[0..., ..<1]
            logits = MLX.where(logits .< threshold, MLXArray(-Float.infinity), logits)
        }
        if topP < 1.0 {
            let sortedIndices = argSort(-logits, axis: -1)
            let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
            let cumulativeProbs = cumsum(softmax(sortedLogits, axis: -1), axis: -1)
            var sortedRemove = cumulativeProbs .> topP
            let lastCol = sortedRemove.dim(-1) - 1
            let firstCol = MLXArray.zeros([sortedRemove.dim(0), 1], type: Bool.self)
            sortedRemove = concatenated([firstCol, sortedRemove[0..., ..<lastCol]], axis: -1)
            let removeMask = putAlong(
                MLXArray.zeros(logits.shape, type: Bool.self), sortedIndices, values: sortedRemove, axis: -1)
            logits = MLX.where(removeMask, MLXArray(-Float.infinity), logits)
        }
        let probs = softmax(logits, axis: -1)
        return categorical(log(probs + 1e-10))
    }

    /// Autoregressive mel-code generation: `[conditioning, text(+lang), mel-start]` prefix,
    /// then incremental single-token decode with KV cache; stops on `stopMelToken`.
    /// Returns RAW codes — callers apply `compressSilence` as the reference does.
    /// `stepLogitsHook` receives the raw (1, 1, vocab) logits per step (parity gates).
    public func generateMelCodes(
        conditioning: MLXArray,
        textTokens: [Int],
        languageID: Int,
        maxMelTokens: Int = 1500,
        temperature: Float = 0.8,
        topK: Int = 30,
        topP: Float = 0.8,
        repetitionPenalty: Float = 10.0,
        stepLogitsHook: ((Int, MLXArray) -> Void)? = nil
    ) -> GenerationResult {
        var inputEmb = prepareInputs(conditioning: conditioning, textTokens: textTokens, languageID: languageID)
        let melStart = MLXArray.full([1, 1], values: MLXArray(Int32(config.startMelToken)))
        inputEmb = concatenated([inputEmb, melEmbedding(melStart) + melPosEmbedding.getFixedEmbedding(0)], axis: 1)

        var melCodes: [Int] = []
        var cache: [(MLXArray, MLXArray)]? = nil
        var stopped = false

        for i in 0 ..< maxMelTokens {
            // Cooperative cancellation bail (engine CAN gate): once per generated mel token.
            if Task.isCancelled { break }

            let step: (MLXArray, MLXArray, [(MLXArray, MLXArray)])
            if cache == nil {
                step = generateStep(
                    inputEmb, cache: nil, temperature: temperature, topK: topK, topP: topP,
                    repetitionPenalty: repetitionPenalty, generatedTokens: melCodes)
            } else {
                let lastToken = MLXArray.full([1, 1], values: MLXArray(Int32(melCodes.last!)))
                let lastEmb = melEmbedding(lastToken) + melPosEmbedding.getFixedEmbedding(melCodes.count)
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
            if let cache { for (k, v) in cache { eval(k, v) } }
        }
        return GenerationResult(melCodes: melCodes, stopped: stopped)
    }
}
