// UnifiedVoiceV2.swift — the IndexTTS2 GPT AR model (P2 subset).
//
// Isomorphic port of mlx_indextts/models/gpt_v2.py. P2 scope: embeddings + GPT backbone +
// final norm / heads + speed embedding + `forwardLatent` (teacher-forced latent extraction,
// gated against the `gpt_latent` golden with the `conditioning` golden injected).
// P3 adds the conditioner stack (ConformerEncoder speaker/emotion + PerceiverResamplers) —
// those parameter families (conditioning_encoder., perceiver_encoder., emo_*, emovec_layer.,
// emo_layer.) exist in gpt.safetensors and are deliberately NOT declared yet; the loader
// works on the declared-subset contract until P3 completes the module tree.

import Foundation
import MLX
import MLXNN

/// Learned position embedding (upstream stores it as `<name>.emb.weight`).
public final class LearnedPositionEmbedding: Module {
    @ModuleInfo(key: "emb") var emb: Embedding

    public init(maxSeqLen: Int, dim: Int) {
        self._emb.wrappedValue = Embedding(embeddingCount: maxSeqLen, dimensions: dim)
    }

    /// Position embeddings for a (B, L, D) or (B, L) input; `offset > 0` = single position.
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        if offset > 0 {
            return emb(MLXArray([Int32(offset)]))
        }
        let seqLen = x.ndim >= 2 ? x.dim(1) : x.dim(0)
        return emb(MLXArray(0..<Int32(seqLen)))
    }

    public func getFixedEmbedding(_ position: Int) -> MLXArray {
        emb(MLXArray([Int32(position)]))[.newAxis, 0...]
    }
}

/// Resolved GPT config (values = oracle truths from the checkpoint's config.yaml).
public struct GPTV2Config: Sendable {
    public var modelDim = 1280
    public var heads = 20
    public var layers = 24
    public var maxMelTokens = 1815
    public var maxTextTokens = 600
    public var numberTextTokens = 12000
    public var numberMelCodes = 8194
    public var startMelToken = 8192
    public var stopMelToken = 8193
    public var startTextToken = 0
    public var stopTextToken = 1
    public var conditionNumLatent = 32
    public var melLengthCompression = 1024

    public init() {}
}

/// UnifiedVoice v2 (P2 subset — see header).
public final class UnifiedVoiceV2: Module {
    public let config: GPTV2Config

    @ModuleInfo(key: "speed_emb") var speedEmb: Embedding
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    @ModuleInfo(key: "mel_embedding") var melEmbedding: Embedding
    @ModuleInfo(key: "mel_pos_embedding") var melPosEmbedding: LearnedPositionEmbedding
    @ModuleInfo(key: "text_pos_embedding") var textPosEmbedding: LearnedPositionEmbedding
    @ModuleInfo(key: "gpt") var gpt: GPT2Model
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "mel_head") var melHead: Linear

    public init(config: GPTV2Config = GPTV2Config()) {
        self.config = config
        self._speedEmb.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.modelDim)
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

    /// Full conditioning = speaker conds + emo vec + [half-speed, normal-speed] embeddings
    /// (mirrors `prepare_conditioning_latents`).
    public func prepareConditioningLatents(
        speechConditioning: MLXArray, emoVec: MLXArray, batchSize: Int
    ) -> MLXArray {
        let condsWithEmo = speechConditioning + emoVec[0..., .newAxis, 0...]
        let zeros = MLXArray.zeros([batchSize]).asType(.int32)
        let ones = MLXArray.ones([batchSize]).asType(.int32)
        let durationEmb = speedEmb(zeros)[0..., .newAxis, 0...]
        let durationEmbHalf = speedEmb(ones)[0..., .newAxis, 0...]
        return concatenated([condsWithEmo, durationEmbHalf, durationEmb], axis: 1)
    }

    /// Teacher-forced latent extraction for S2Mel (mirrors `forward_latent`).
    ///
    /// - Parameters:
    ///   - conditioning: (B, condNum+2, D) — P2 injects the captured golden here.
    ///   - textTokens: (B, textLen) token ids (no start/stop; added inside).
    ///   - melCodes: (B, melLen) generated mel codes (no start/stop; added inside).
    /// - Returns: (B, melLen, D) latents.
    public func forwardLatent(
        conditioning: MLXArray, textTokens: MLXArray, melCodes: MLXArray
    ) -> MLXArray {
        let batchSize = textTokens.dim(0)
        let melLen = melCodes.dim(1)

        // [start, text..., stop]
        let startTokens = MLXArray.full([batchSize, 1], values: MLXArray(Int32(config.startTextToken)))
        let stopTokens = MLXArray.full([batchSize, 1], values: MLXArray(Int32(config.stopTextToken)))
        let text = concatenated([startTokens, textTokens.asType(.int32), stopTokens], axis: 1)

        var textEmb = textEmbedding(text)
        textEmb = textEmb + textPosEmbedding(textEmb)

        // [start, mel..., stop]
        let melStart = MLXArray.full([batchSize, 1], values: MLXArray(Int32(config.startMelToken)))
        let melStop = MLXArray.full([batchSize, 1], values: MLXArray(Int32(config.stopMelToken)))
        let mel = concatenated([melStart, melCodes.asType(.int32), melStop], axis: 1)

        var melEmb = melEmbedding(mel)
        melEmb = melEmb + melPosEmbedding(melEmb)

        let emb = concatenated([conditioning, textEmb, melEmb], axis: 1)
        let (hidden, _) = gpt(emb)

        let condLen = conditioning.dim(1)
        let enc = finalNorm(hidden[0..., condLen..., 0...])

        let textLenWithTokens = textEmb.dim(1)
        return enc[0..., textLenWithTokens..<(textLenWithTokens + melLen), 0...]
    }
}
