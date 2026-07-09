// UnifiedVoiceV2.swift — the IndexTTS2 GPT AR model.
//
// Isomorphic port of mlx_indextts/models/gpt_v2.py. P2 delivered embeddings + GPT backbone +
// final norm / heads + speed embedding + `forwardLatent` (teacher-forced, gated vs the
// `gpt_latent` golden). P3b adds the conditioner stack: speaker ConformerEncoder +
// PerceiverResampler (32 latents), emotion ConformerEncoder + PerceiverResampler (1 latent),
// emovec/emo projection layers — `getConditioning` / `getEmovec` /
// `prepareConditioningLatents` now produce the conditioning natively (gated vs the
// core_gpt_speech_cond / core_gpt_base_emovec / core_gpt_conditioning goldens).
//
// Conditioner configs are checkpoint config.yaml truths (resolved-config pitfall):
// cond = Conformer(1024→512, ff 2048, 8 heads, 6 blocks) + Perceiver(1280, ctx 512, 32 lat,
// 8 heads, mult 2); emo = Conformer(1024→512, ff 1024, 4 heads, 4 blocks) +
// Perceiver(1024, ctx 512, 1 lat, 4 heads, mult 2).

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

    @ModuleInfo(key: "conditioning_encoder") var conditioningEncoder: ConformerEncoder
    @ModuleInfo(key: "perceiver_encoder") var perceiverEncoder: PerceiverResampler
    @ModuleInfo(key: "emo_conditioning_encoder") var emoConditioningEncoder: ConformerEncoder
    @ModuleInfo(key: "emo_perceiver_encoder") var emoPerceiverEncoder: PerceiverResampler
    @ModuleInfo(key: "emo_layer") var emoLayer: Linear
    @ModuleInfo(key: "emovec_layer") var emovecLayer: Linear
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

        // Speaker conditioning: conformer_perceiver
        let condConfig = ConformerConfig(
            inputSize: 1024, outputSize: 512, linearUnits: 2048, attentionHeads: 8, numBlocks: 6)
        self._conditioningEncoder.wrappedValue = ConformerEncoder(condConfig)
        self._perceiverEncoder.wrappedValue = PerceiverResampler(
            dim: config.modelDim, nDimContext: condConfig.outputSize,
            nLatents: config.conditionNumLatent, nHeads: condConfig.attentionHeads, nFFMult: 2)

        // Emotion conditioning (v2)
        let emoCondConfig = ConformerConfig(
            inputSize: 1024, outputSize: 512, linearUnits: 1024, attentionHeads: 4, numBlocks: 4)
        self._emoConditioningEncoder.wrappedValue = ConformerEncoder(emoCondConfig)
        self._emoPerceiverEncoder.wrappedValue = PerceiverResampler(
            dim: 1024, nDimContext: emoCondConfig.outputSize,
            nLatents: 1, nHeads: emoCondConfig.attentionHeads, nFFMult: 2)

        self._emoLayer.wrappedValue = Linear(config.modelDim, config.modelDim)
        self._emovecLayer.wrappedValue = Linear(1024, config.modelDim)

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

    /// Speaker conditioning from w2v-BERT semantic features (mirrors `get_conditioning`,
    /// condition_type == "conformer_perceiver").
    /// - Parameter speechConditioningInput: (B, 1024, T) NCL semantic features.
    /// - Returns: (B, condNum, modelDim) conditioning latents.
    public func getConditioning(_ speechConditioningInput: MLXArray) -> MLXArray {
        let x = speechConditioningInput.transposed(0, 2, 1)  // NCL → NLC for the Conformer
        return perceiverEncoder(conditioningEncoder(x))
    }

    /// Emotion conditioning (mirrors `get_emo_conditioning`): (B, 1024, T) NCL → (B, 1024).
    public func getEmoConditioning(_ emoConditioningInput: MLXArray) -> MLXArray {
        let x = emoConditioningInput.transposed(0, 2, 1)
        return emoPerceiverEncoder(emoConditioningEncoder(x)).squeezed(axis: 1)
    }

    /// Emotion vector (mirrors `get_emovec`): (B, 1024, T) NCL → (B, modelDim).
    public func getEmovec(_ emoConditioningInput: MLXArray) -> MLXArray {
        emoLayer(emovecLayer(getEmoConditioning(emoConditioningInput)))
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

    /// Remap gpt.safetensors keys → this module's parameter tree. The weights are already
    /// in solar2ain MLX naming; the only remap is the perceiver ModuleList-of-pairs
    /// `layers.N.0.*`/`layers.N.1.*` → `layers.N.attn.*`/`layers.N.ff.*` (numeric module
    /// keys collide with array-index unflattening).
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            var nk = k
            if nk.contains("perceiver_encoder.layers.") {
                if let range = nk.range(of: #"(layers\.\d+)\.0\."#, options: .regularExpression) {
                    nk = nk.replacingCharacters(
                        in: range, with: String(nk[range].dropLast(3)) + ".attn.")
                } else if let range = nk.range(of: #"(layers\.\d+)\.1\."#, options: .regularExpression) {
                    nk = nk.replacingCharacters(
                        in: range, with: String(nk[range].dropLast(3)) + ".ff.")
                }
            }
            out[nk] = v
        }
        return out
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
