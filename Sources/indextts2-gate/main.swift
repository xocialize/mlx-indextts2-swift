// indextts2-gate — the IndexTTS-2.5 parity-gate CLI lane (`swift run indextts2-gate <mode>`).
//
// Every mode replays one stage of the port against the goldens captured from the donor
// Python-MLX oracle (vanch007/mlx-indextts2 `v25`, fp32 CPU; PORTING/goldens-v25, captured by
// WIP/indextts25/tools/capture_v25_goldens.py) and fails loudly outside the stated tolerance.
// CPU lanes upcast the fp16 checkpoint to fp32 (the oracle's dtype) and pin the CPU device;
// the GPU lanes (`e2e`, `quant`, `footprint`) run the production dtype on Metal.
//
// Modes: tok | ref | gpt | codec | s2mel | e2e | quant | footprint | all
//   --weights <dir>   the converted 2.5 checkpoint (default: WIP/indextts25/weights/mlx-indextts2-2.5-fp16)
//   --w2v <dir>       facebook/w2v-bert-2.0 model.safetensors directory (default: the 2.0 store copy)
//   --goldens <dir>   golden directory (default: PORTING/goldens-v25 under the cwd)

import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXIndexTTS2
import MLXIndexTTS2TTS
import MLXServeCore
import MLXToolKit

// MARK: - Plumbing

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("✗ " + message + "\n").data(using: .utf8)!)
    exit(1)
}

func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).reshaped(-1)
    let y = b.asType(.float32).reshaped(-1)
    let dot = sum(x * y).item(Float.self)
    let nx = sqrt(sum(x * x)).item(Float.self)
    let ny = sqrt(sum(y * y)).item(Float.self)
    return dot / max(nx * ny, 1e-12)
}

func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func dbfs(_ samples: [Float]) -> Float {
    let rms = sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(max(samples.count, 1)))
    return 20 * log10(max(rms, 1e-9))
}

func argValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: name), idx + 1 < args.count else { return nil }
    return args[idx + 1]
}

let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let weightsDir = argValue("--weights").map { URL(fileURLWithPath: $0) }
    ?? URL(fileURLWithPath: "/Volumes/Satechi/Development/mlxengine-audio/WIP/indextts25/weights/mlx-indextts2-2.5-fp16")
let w2vDir = argValue("--w2v").map { URL(fileURLWithPath: $0) }
    ?? URL(fileURLWithPath: NSString(string: "~/.cache/huggingface/hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b").expandingTildeInPath)
let goldensDir = argValue("--goldens").map { URL(fileURLWithPath: $0) } ?? cwd.appending(path: "PORTING/goldens-v25")

func golden(_ name: String) throws -> MLXArray {
    try NPY.load(goldensDir.appending(path: "\(name).npy")).asType(.float32)
}

func goldenInts(_ name: String) throws -> [Int] {
    try NPY.load(goldensDir.appending(path: "\(name).npy")).asType(.int32).asArray(Int32.self).map(Int.init)
}

func check(_ name: String, _ ours: MLXArray, _ gold: MLXArray, cosMin: Float, madMax: Float) {
    guard ours.shape == gold.shape else { fail("\(name): shape \(ours.shape) vs golden \(gold.shape)") }
    let cos = cosine(ours, gold)
    let mad = maxAbsDiff(ours, gold)
    print(String(format: "  %@  cos=%.7f  max_abs=%.3e", name.padding(toLength: 26, withPad: " ", startingAt: 0), cos, mad))
    if cos < cosMin || mad > madMax {
        fail(String(format: "%@ gate failed (cos %.7f < %.5f or max_abs %.3e > %.3e)", name, cos, cosMin, mad, madMax))
    }
}

struct GoldenText: Codable { let text: String; let language: String; let language_id: Int; let token_ids: [Int] }

func loadGenerator(fp32: Bool, quantBits: Int? = nil) throws -> IndexTTS2Generator {
    let t0 = Date()
    let g = try IndexTTS2Generator.load(modelDirectory: weightsDir, w2vBertDirectory: w2vDir, quantBits: quantBits)
    if fp32 { g.upcast(to: .float32) }
    print(String(format: "→ generator loaded%@ in %.1fs", fp32 ? " (fp32 upcast)" : "", Date().timeIntervalSince(t0)))
    return g
}

func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
    var data = Data()
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    let pcm = samples.map { Int16(max(-32768, min(32767, ($0 * 32767).rounded()))) }
    data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + pcm.count * 2))
    data.append(contentsOf: Array("WAVE".utf8)); data.append(contentsOf: Array("fmt ".utf8))
    u32(16); u16(1); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
    data.append(contentsOf: Array("data".utf8)); u32(UInt32(pcm.count * 2))
    for v in pcm { u16(UInt16(bitPattern: v)) }
    try data.write(to: url)
}

// MARK: - tok: tiktoken + frontend (the XCTest fixture, re-run here against the weights dir)

func gateTok() throws {
    let vocab = weightsDir.appending(path: IndexTTS2Generator.tokenizerFile)
    let tokenizer = try TiktokenBPE(vocabularyURL: vocab)
    let frontend = IndexTTSTextFrontend(tokenizer: tokenizer)
    struct F: Codable { let language: String; let text: String; let token_ids: [[Int]]; let raw_bpe_ids: [Int]; let max_text_tokens_per_segment: Int? }
    struct B: Codable { let text: String; let ids: [Int] }
    struct Fix: Codable { let frontend: [F]; let bpe: [B] }
    let fix = try JSONDecoder().decode(Fix.self, from: Data(contentsOf: goldensDir.appending(path: "text_fixtures.json")))
    var exact = 0, gaps: [String] = []
    for b in fix.bpe {
        guard tokenizer.encode(b.text) == b.ids else { fail("bpe mismatch: \(b.text.debugDescription)") }
        exact += 1
    }
    for f in fix.frontend {
        guard tokenizer.encode(f.text) == f.raw_bpe_ids else { fail("raw bpe mismatch: \(f.text.debugDescription)") }
        let prepared = try frontend.prepare(f.text, language: IndexTTSLanguage(rawValue: f.language)!,
                                            maxTokensPerSegment: f.max_text_tokens_per_segment ?? 120)
        if prepared.tokenIDs == f.token_ids { exact += 1 } else { gaps.append(f.text) }
    }
    print("  vocab \(tokenizer.vocabularySize)  bpe \(fix.bpe.count)/\(fix.bpe.count) exact  frontend \(fix.frontend.count - gaps.count)/\(fix.frontend.count) exact")
    for g in gaps { print("  gap (WeText digit expansion, documented): \(g)") }
    let digitGaps = gaps.filter { $0.unicodeScalars.contains { CharacterSet.decimalDigits.contains($0) } }
    guard digitGaps.count == gaps.count else { fail("non-digit frontend mismatch: \(gaps)") }
    print("TOK GATE PASSED")
}

// MARK: - ref: reference conditioning chain (fp32 CPU)

func gateRef() throws {
    Device.setDefault(device: Device(.cpu))
    let g = try loadGenerator(fp32: true)
    let wav16 = try golden("audio_16k")
    let wav22 = try golden("audio_22k")

    guard let (features, mask) = SeamlessFeatureExtractor.callAsFeatures(wav16) else { fail("audio too short") }
    check("seamless features", features, try golden("seamless_input_features"), cosMin: 0.99999, madMax: 2e-3)
    let goldMaskSum = try golden("seamless_attention_mask").sum().item(Float.self)
    guard Float(mask.asType(.int32).sum().item(Int32.self)) == goldMaskSum else { fail("mask mismatch") }
    let (_, hs) = g.w2v(inputFeatures: features, attentionMask: mask)
    let spk = Wav2Vec2BertModel.semanticTap(hs, mean: g.semanticMean, std: g.semanticStd)
    eval(spk)
    check("w2v spk_cond_emb", spk, try golden("spk_cond_emb"), cosMin: 0.9999, madMax: 5e-2)

    guard let cmn = CampPlusFbank.fbankCMN(wav16) else { fail("audio too short") }
    check("campplus fbank cmn", cmn, try golden("campplus_fbank_cmn"), cosMin: 0.99999, madMax: 2e-3)
    let style = g.campplus(cmn.expandedDimensions(axis: 0)); eval(style)
    check("campplus style", style, try golden("style"), cosMin: 0.9999, madMax: 5e-3)

    let refMel = RefMel.melSpectrogram(wav22); eval(refMel)
    check("ref_mel", refMel, try golden("ref_mel"), cosMin: 0.99999, madMax: 5e-3)

    // Use the GOLDEN upstream tensors below so each stage is judged on its own numerics.
    let spkG = try golden("spk_cond_emb"), styleG = try golden("style")
    let prompt = g.s2mel.lengthRegulatorModule(spkG, ylens: MLXArray([Int32(refMel.dim(2))])); eval(prompt)
    check("prompt_condition (LR)", prompt, try golden("prompt_condition"), cosMin: 0.99999, madMax: 1e-3)
    let emovec = g.gpt.getEmovec(spkG.transposed(0, 2, 1)); eval(emovec)
    check("base_emovec", emovec, try golden("base_emovec"), cosMin: 0.9999, madMax: 2e-2)
    let proj = g.gpt.spkEmbProj(styleG); eval(proj)
    check("spk_emb_proj", proj, try golden("spk_emb_proj"), cosMin: 0.99999, madMax: 1e-3)
    let cond = g.gpt.prepareConditioningLatents(style: styleG, emoVec: try golden("base_emovec")); eval(cond)
    check("conditioning", cond, try golden("conditioning"), cosMin: 0.99999, madMax: 1e-3)
    var weights = [Float](repeating: 0, count: 8); weights[0] = 0.6
    let mat = EmotionPresets.emovecMat(weights: weights, style: styleG); eval(mat)
    check("emovec_mat happy 0.6", mat, try golden("emovec_mat_happy06"), cosMin: 0.999999, madMax: 1e-4)
    let blend = EmotionPresets.blend(weights: weights, style: styleG, baseEmovec: try golden("base_emovec")); eval(blend)
    check("emovec blend", blend, try golden("emovec_happy06"), cosMin: 0.999999, madMax: 1e-4)
    print("REF GATE PASSED")
}

// MARK: - gpt: language-fused inputs, teacher-forced logits, greedy rollout (fp32 CPU)

func gateGPT() throws {
    Device.setDefault(device: Device(.cpu))
    let g = try loadGenerator(fp32: true)
    let text = try JSONDecoder().decode(GoldenText.self, from: Data(contentsOf: goldensDir.appending(path: "gpt_text_tokens.json")))
    let cond = try golden("conditioning")
    let inputEmb = g.gpt.prepareInputs(conditioning: cond, textTokens: text.token_ids, languageID: text.language_id)
    eval(inputEmb)
    // The oracle left-pads one masked zero row; ours has none — compare the tail.
    let goldEmb = try golden("gpt_input_emb")
    let pad = goldEmb.dim(1) - inputEmb.dim(1)
    guard pad >= 0 else { fail("input_emb longer than golden (\(inputEmb.dim(1)) vs \(goldEmb.dim(1)))") }
    check("gpt input_emb (unpadded)", inputEmb, goldEmb[0..., pad..., 0...], cosMin: 0.999999, madMax: 1e-4)

    let codes = try goldenInts("gpt_greedy_codes")
    // Teacher-forced: [cond, text, mel-start, codes] in one prefill → logits over mel positions.
    let melSeq = MLXArray(([g.gpt.config.startMelToken] + codes).map(Int32.init)).expandedDimensions(axis: 0)
    let melEmb = g.gpt.melEmbedding(melSeq) + g.gpt.melPosEmbedding(melSeq)
    let (hidden, _) = g.gpt.gpt(concatenated([inputEmb, melEmb], axis: 1))
    let logits = g.gpt.melHead(g.gpt.finalNorm(hidden[0..., inputEmb.dim(1)..., 0...])); eval(logits)
    check("teacher-forced logits", logits, try golden("gpt_teacher_forced_logits"), cosMin: 0.9999, madMax: 0.5)
    check("step-0 logits", logits[0, 0], try golden("gpt_step0_logits")[0], cosMin: 0.9999, madMax: 0.5)

    // Greedy rollout — token-exact.
    let result = g.gpt.generateMelCodes(conditioning: cond, textTokens: text.token_ids, languageID: text.language_id,
                                        maxMelTokens: 400, temperature: 0, topK: 0, topP: 1.0, repetitionPenalty: 1.0)
    let matched = zip(result.melCodes, codes).prefix { $0 == $1 }.count
    print("  greedy rollout: \(result.melCodes.count) codes (golden \(codes.count)), stopped=\(result.stopped), prefix match \(matched)")
    guard result.melCodes == codes, result.stopped else { fail("greedy rollout not token-exact") }
    print("GPT GATE PASSED")
}

// MARK: - codec: EnhancedCodec decode (fp32 CPU)

func gateCodec() throws {
    Device.setDefault(device: Device(.cpu))
    let g = try loadGenerator(fp32: true)
    let codes = MLXArray(try goldenInts("codes_compressed").map(Int32.init)).expandedDimensions(axis: 0)
    let vq = g.codec.vq2emb(codes); eval(vq)
    check("codec vq2emb", vq, try golden("codec_vq2emb"), cosMin: 0.999999, madMax: 1e-4)
    let s = g.codec(codes); eval(s)
    check("codec S_infer", s, try golden("codec_s_infer"), cosMin: 0.99999, madMax: 5e-3)
    print("CODEC GATE PASSED")
}

// MARK: - s2mel: length regulator → CFM (seed 42) → BigVGAN (fp32 CPU)

func gateS2Mel() throws {
    Device.setDefault(device: Device(.cpu))
    let g = try loadGenerator(fp32: true)
    let sInfer = try golden("codec_s_infer")
    let target = Int(Double(sInfer.dim(1)) * 1.72)
    let cond = g.s2mel.lengthRegulatorModule(sInfer, ylens: MLXArray([Int32(target)])); eval(cond)
    check("lenreg cond", cond, try golden("lenreg_cond"), cosMin: 0.99999, madMax: 2e-3)

    let prompt = try golden("prompt_condition"), refMel = try golden("ref_mel"), style = try golden("style")
    let combined = concatenated([prompt, try golden("lenreg_cond")], axis: 1)
    MLXRandom.seed(42)
    let z = MLXRandom.normal([1, 80, combined.dim(1)]); eval(z)
    check("cfm z (seed 42)", z, try golden("cfm_z_seed42"), cosMin: 0.999999, madMax: 1e-5)
    MLXRandom.seed(42)
    let mel = g.s2mel.cfmModule.inference(mu: combined, xLens: MLXArray([Int32(combined.dim(1))]), prompt: refMel,
                                          style: style, nTimesteps: 25, temperature: 1.0, inferenceCfgRate: 0.7)
    eval(mel)
    // The 2.5 donor's solver returns x AFTER the final prompt-region re-zero; this port (like the
    // original donor and upstream) returns it before. The prompt frames are trimmed before the
    // vocoder either way, so parity is judged on the generated region.
    let goldMel = try golden("cfm_mel_seed42")
    check("cfm mel (25 steps, generated)", mel[0..., 0..., refMel.dim(2)...], goldMel[0..., 0..., refMel.dim(2)...],
          cosMin: 0.9999, madMax: 0.2)
    let wav = g.bigvgan(try golden("cfm_mel_seed42")[0..., 0..., refMel.dim(2)...]); eval(wav)
    check("bigvgan wav", wav, try golden("bigvgan_wav_seed42"), cosMin: 0.999, madMax: 0.05)
    print("S2MEL GATE PASSED")
}

// MARK: - e2e: production dtype on the GPU stream

func gateE2E() throws {
    let g = try loadGenerator(fp32: false)
    let text = try JSONDecoder().decode(GoldenText.self, from: Data(contentsOf: goldensDir.appending(path: "gpt_text_tokens.json")))
    let wav16 = try golden("audio_16k").asArray(Float.self)
    let wav22 = try golden("audio_22k").asArray(Float.self)
    let t0 = Date()
    let reference = try g.prepareReference(samples16k: wav16, samples22k: wav22)
    print(String(format: "  reference prepared in %.2fs", Date().timeIntervalSince(t0)))
    check("style (fp16 GPU)", reference.style, try golden("style"), cosMin: 0.999, madMax: 0.05)
    check("base_emovec (fp16 GPU)", reference.baseEmovec, try golden("base_emovec"), cosMin: 0.999, madMax: 0.5)

    // Greedy chain vs the fp32-CPU golden codes: fp16 Metal is allowed to diverge late
    // (AR knife-edges), so report the prefix and require a long exact prefix, not identity.
    let cond = g.gpt.prepareConditioningLatents(style: reference.style, emoVec: reference.baseEmovec)
    let greedy = g.gpt.generateMelCodes(conditioning: cond, textTokens: text.token_ids, languageID: text.language_id,
                                        maxMelTokens: 400, temperature: 0, topK: 0, topP: 1.0, repetitionPenalty: 1.0)
    let goldCodes = try goldenInts("gpt_greedy_codes")
    let prefix = zip(greedy.melCodes, goldCodes).prefix { $0 == $1 }.count
    print("  greedy (fp16 GPU): \(greedy.melCodes.count) codes, exact prefix \(prefix)/\(goldCodes.count), stopped=\(greedy.stopped)")
    guard prefix >= 20, greedy.stopped else { fail("fp16 GPU greedy diverged too early or did not stop") }

    // Full sampled synthesis (seed 42), quantified.
    var params = IndexTTS2Generator.SynthesisParams()
    params.maxTextTokensPerSegment = 120
    MLXRandom.seed(42)
    let t1 = Date()
    let audio = try g.synthesize(text: text.text, reference: reference, language: .en, params: params)
    let secs = Double(audio.count) / 22050.0
    print(String(format: "  e2e sampled: %.2fs audio in %.2fs wall (rtf %.2f), %.1f dBFS", secs, Date().timeIntervalSince(t1),
                 Date().timeIntervalSince(t1) / secs, dbfs(audio)))
    guard dbfs(audio) > -35, dbfs(audio) < -10, secs > 3, secs < 12 else { fail("e2e audio outside the validity envelope") }
    try writeWAV(audio, sampleRate: 22050, to: cwd.appending(path: "PORTING/v25_e2e_seed42.wav"))

    // The E12 levers: happy preset, 3 s target, 1.3× rate.
    MLXRandom.seed(42)
    var happy = [Float](repeating: 0, count: 8); happy[0] = 0.6
    let happyAudio = try g.synthesize(text: text.text, reference: reference, language: .en, emotionWeights: happy, params: params)
    print(String(format: "  happy 0.6: %.2fs, %.1f dBFS", Double(happyAudio.count) / 22050.0, dbfs(happyAudio)))
    try writeWAV(happyAudio, sampleRate: 22050, to: cwd.appending(path: "PORTING/v25_happy_seed42.wav"))
    MLXRandom.seed(42)
    let fitted = try g.synthesize(text: text.text, reference: reference, language: .en, targetDurationSeconds: 3.0, params: params)
    let fittedSecs = Double(fitted.count) / 22050.0
    print(String(format: "  targetDuration 3.0: %.2fs, %.1f dBFS", fittedSecs, dbfs(fitted)))
    guard abs(fittedSecs - 3.0) < 0.05 else { fail("targetDuration missed: \(fittedSecs)") }
    MLXRandom.seed(42)
    let fast = try g.synthesize(text: text.text, reference: reference, language: .en, speechRate: 1.3, params: params)
    print(String(format: "  speechRate 1.3: %.2fs (natural %.2fs → expect ≈%.2fs)", Double(fast.count) / 22050.0, secs, secs / 1.3))

    // A second language through the same clone (zh), validity only.
    MLXRandom.seed(42)
    let zh = try g.synthesize(text: "今天的天气真不错，我们一起去公园散步吧。", reference: reference, language: .zh, params: params)
    print(String(format: "  zh: %.2fs, %.1f dBFS", Double(zh.count) / 22050.0, dbfs(zh)))
    guard dbfs(zh) > -35, dbfs(zh) < -10 else { fail("zh audio outside the validity envelope") }
    try writeWAV(zh, sampleRate: 22050, to: cwd.appending(path: "PORTING/v25_zh_seed42.wav"))
    print("E2E GATE PASSED")
}

// MARK: - quant: int8 / int4 GPT backbone on the GPU stream

func gateQuant() throws {
    let text = try JSONDecoder().decode(GoldenText.self, from: Data(contentsOf: goldensDir.appending(path: "gpt_text_tokens.json")))
    let cond = try golden("conditioning")
    let goldCodes = try goldenInts("gpt_greedy_codes")
    let goldLogits = try golden("gpt_teacher_forced_logits")
    let wav16 = try golden("audio_16k").asArray(Float.self)
    let wav22 = try golden("audio_22k").asArray(Float.self)
    for bits in [8, 4] {
        let g = try loadGenerator(fp32: false, quantBits: bits)
        let inputEmb = g.gpt.prepareInputs(conditioning: cond, textTokens: text.token_ids, languageID: text.language_id)
        let melSeq = MLXArray(([g.gpt.config.startMelToken] + goldCodes).map(Int32.init)).expandedDimensions(axis: 0)
        let melEmb = g.gpt.melEmbedding(melSeq) + g.gpt.melPosEmbedding(melSeq)
        let (hidden, _) = g.gpt.gpt(concatenated([inputEmb, melEmb], axis: 1))
        let logits = g.gpt.melHead(g.gpt.finalNorm(hidden[0..., inputEmb.dim(1)..., 0...])); eval(logits)
        let cos = cosine(logits, goldLogits)
        let greedy = g.gpt.generateMelCodes(conditioning: cond, textTokens: text.token_ids, languageID: text.language_id,
                                            maxMelTokens: 400, temperature: 0, topK: 0, topP: 1.0, repetitionPenalty: 1.0)
        let prefix = zip(greedy.melCodes, goldCodes).prefix { $0 == $1 }.count
        print(String(format: "  int%d: teacher-forced logits cos=%.6f, greedy prefix %d/%d, %d codes, stopped=%@",
                     bits, cos, prefix, goldCodes.count, greedy.melCodes.count, greedy.stopped ? "yes" : "no"))
        guard cos >= (bits == 8 ? 0.999 : 0.99), greedy.stopped else { fail("int\(bits) gate failed") }
        let reference = try g.prepareReference(samples16k: wav16, samples22k: wav22)
        MLXRandom.seed(42)
        let audio = try g.synthesize(text: text.text, reference: reference, language: .en)
        print(String(format: "  int%d e2e: %.2fs, %.1f dBFS", bits, Double(audio.count) / 22050.0, dbfs(audio)))
        guard dbfs(audio) > -35, dbfs(audio) < -10 else { fail("int\(bits) e2e audio outside the validity envelope") }
        try writeWAV(audio, sampleRate: 22050, to: cwd.appending(path: "PORTING/v25_int\(bits)_seed42.wav"))
        Memory.clearCache()
    }
    print("QUANT GATE PASSED")
}

// MARK: - footprint: MLX-active resident floor + run peak for ONE tier (manifest numbers)
// One tier per process (`--bits 8|4`, default fp16): MLX's peak counter is process-cumulative.

func gateFootprint() throws {
    let text = try JSONDecoder().decode(GoldenText.self, from: Data(contentsOf: goldensDir.appending(path: "gpt_text_tokens.json")))
    let wav16 = try golden("audio_16k").asArray(Float.self)
    let wav22 = try golden("audio_22k").asArray(Float.self)
    let long = String(repeating: text.text + " ", count: 3)
    let bits: Int? = argValue("--bits").flatMap(Int.init)
    do {
        let g = try loadGenerator(fp32: false, quantBits: bits)
        let resident = Memory.activeMemory
        let reference = try g.prepareReference(samples16k: wav16, samples22k: wav22)
        MLXRandom.seed(42)
        let a1 = try g.synthesize(text: text.text, reference: reference, language: .en)
        let peakShort = Memory.peakMemory
        let a2 = try g.synthesize(text: long, reference: reference, language: .en)
        let peakLong = Memory.peakMemory
        print(String(format: "  [FOOT] tier=%@ resident=%d MB peak(%.1fs)=%d MB peak(%.1fs)=%d MB transient=%d MB",
                     bits.map { "int\($0)" } ?? "fp16", resident / 1_000_000, Double(a1.count) / 22050, peakShort / 1_000_000,
                     Double(a2.count) / 22050, peakLong / 1_000_000, (peakLong - resident) / 1_000_000))
    }
    print("FOOTPRINT DONE")
}

// MARK: - engine: the consumer path — MLXServeEngine (DEFAULT policy) register → prepare
// (engine-executed materialization from the mlx-community repo into `--store`) → run → evict.
// This is the exact call sequence ML[X] Audio Studio's VoiceLane makes.

func gateEngine() async throws {
    let store = URL(fileURLWithPath: argValue("--store")
        ?? "/Volumes/Satechi/Development/mlxengine-audio/WIP/indextts25/store")
    let refURL = URL(fileURLWithPath: argValue("--ref")
        ?? "/Volumes/Satechi/Development/mlxengine-audio/Archive/MLXEngineAudio/MLXEngineAudio/TTSValidation/indextts2-ref.wav")
    let clip = try Data(contentsOf: refURL)
    let engine = MLXServeEngine()   // .permissiveOnly — the weights must admit here, no acknowledgement
    await engine.useModelStore(ModelStore(root: store))
    let t0 = Date()
    let id = try await engine.register(IndexTTS2Package.registration, configuration: IndexTTS2Configuration(),
                                       id: PackageID("indextts2"))
    let advisories = await engine.licenseAdvisories
    print("  registered \(id.rawValue) under .permissiveOnly; license advisories: \(advisories.count)")
    guard advisories.isEmpty else { fail("license advisory raised — the weights are not admitted cleanly: \(advisories)") }
    let needs = await engine.needsDownload(.tts, package: id)
    print("  needsDownload=\(needs) (store \(store.path))")
    try await engine.prepare(.tts, package: id)
    print(String(format: "  prepared in %.1fs (download + load)", Date().timeIntervalSince(t0)))

    func take(_ text: String, meta: MetaData, label: String) async throws {
        let t = Date()
        let request = TTSRequest(text: text, voice: VoiceSelector(.referenceAudio(Audio(format: .wav, data: clip))),
                                 metaData: meta)
        let response = try await engine.run(request, package: id)
        guard let tts = response as? TTSResponse else { fail("unexpected response") }
        let wav = tts.audio.data
        // 16-bit PCM body after the 44-byte header.
        let samples = wav.dropFirst(44).withUnsafeBytes { raw -> [Float] in
            raw.bindMemory(to: Int16.self).map { Float($0) / 32768 }
        }
        let secs = Double(samples.count) / Double(tts.audio.sampleRate ?? 22_050)
        print(String(format: "  [RUN] %@: %.2fs audio, %.1f dBFS, %.2fs wall (rtf %.2f)", label, secs, dbfs(samples),
                     Date().timeIntervalSince(t), Date().timeIntervalSince(t) / max(secs, 0.01)))
        guard dbfs(samples) > -35, dbfs(samples) < -10, secs > 1 else { fail("\(label): audio outside the validity envelope") }
        try wav.write(to: cwd.appending(path: "PORTING/v25_engine_\(label).wav"))
    }
    try await take("The afternoon light settles quietly on the river, and nobody is in a hurry.",
                   meta: ["seed": .int(42)], label: "en")
    try await take("The afternoon light settles quietly on the river, and nobody is in a hurry.",
                   meta: ["seed": .int(42), "emotion": .string("happy"), "emoAlpha": .double(0.6), "targetDuration": .double(4.0)],
                   label: "happy_4s")
    try await take("今天的天气真不错，我们一起去公园散步吧。", meta: ["seed": .int(42), "language": .string("zh")], label: "zh")
    await engine.evict(package: id)
    print("ENGINE GATE PASSED")
}

// MARK: - Entry

let mode = CommandLine.arguments.dropFirst().first ?? "all"
do {
    switch mode {
    case "tok": try gateTok()
    case "ref": try gateRef()
    case "gpt": try gateGPT()
    case "codec": try gateCodec()
    case "s2mel": try gateS2Mel()
    case "e2e": try gateE2E()
    case "quant": try gateQuant()
    case "footprint": try gateFootprint()
    case "engine": try await gateEngine()   // top-level await: no main-thread wait to deadlock actors
    case "all":
        try gateTok(); try gateRef(); try gateGPT(); try gateCodec(); try gateS2Mel()
    default: fail("unknown mode \(mode) (tok | ref | gpt | codec | s2mel | e2e | quant | footprint | engine | all)")
    }
} catch {
    fail("\(error)")
}
