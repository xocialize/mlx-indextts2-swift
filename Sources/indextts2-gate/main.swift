// indextts2-gate — CLI parity-gate lane for the IndexTTS2 Swift port.
//
// Gates run here (plain `swift run`), not XCTest, per the mlx-swift-integration doctrine
// (the SPM test product's metallib is unreliable for kernel work). fp32 gates pin the CPU
// stream; quant gates (P7, later) must run the forward on GPU.
//
// Usage:
//   swift run indextts2-gate p2 [--weights <dir>] [--goldens <dir>]
//
// p2: teacher-forced UnifiedVoiceV2.forwardLatent vs the Stage-0 `gpt_latent` golden,
//     with the captured `conditioning` golden injected. Gate: cosine ≥ 0.999 (fp32, CPU).

import Foundation
import MLX
import MLXNN
import MLXIndexTTS2

// MARK: - Helpers

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("FAIL: " + message + "\n").data(using: .utf8)!)
    exit(1)
}

func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).reshaped(-1)
    let y = b.asType(.float32).reshaped(-1)
    let num = sum(x * y)
    let den = sqrt(sum(x * x)) * sqrt(sum(y * y))
    return (num / den).item(Float.self)
}

func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func argValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: name), idx + 1 < args.count else { return nil }
    return args[idx + 1]
}

let home = FileManager.default.homeDirectoryForCurrentUser
let defaultWeights = home.appending(
    path: ".cache/huggingface/hub/models--vanch007--mlx-indextts2-standard-fp16/snapshots/31118db400202a438e4d42bbce8e426298072d50")
let defaultGoldens = home.appending(path: "Development/_indextts2-oracle/goldens")

let weightsDir = argValue("--weights").map { URL(fileURLWithPath: $0) } ?? defaultWeights
let goldensDir = argValue("--goldens").map { URL(fileURLWithPath: $0) } ?? defaultGoldens

/// Shared UnifiedVoiceV2 loader: full-model key contract (0-missing/0-unused after
/// sanitize), fp32 upcast materialized before any forward (watchdog corollary).
func loadUnifiedVoiceV2() throws -> UnifiedVoiceV2 {
    let model = UnifiedVoiceV2()
    let declared = Set(model.parameters().flattened().map(\.0))

    let raw = try loadArrays(url: weightsDir.appending(path: "gpt.safetensors"))
    let sanitized = UnifiedVoiceV2.sanitize(raw)

    let missing = declared.subtracting(sanitized.keys)
    let unused = Set(sanitized.keys).subtracting(declared)
    guard missing.isEmpty else { fail("missing keys: \(missing.sorted().prefix(8)) …") }
    guard unused.isEmpty else { fail("unused keys: \(unused.sorted().prefix(8)) …") }
    print("  keys: \(sanitized.count) (declared \(declared.count)); contract 0-missing/0-unused OK")

    let fp32 = sanitized.mapValues { $0.asType(.float32) }
    try model.update(parameters: ModuleParameters.unflattened(fp32), verify: .all)
    eval(model)
    return model
}

// MARK: - P2 gate

func gateP2() throws {
    // fp32 parity on the CPU stream (weight loads must be CPU-stream anyway).
    Device.setDefault(device: Device(.cpu))

    print("→ loading goldens from \(goldensDir.path)")
    let conditioning = try NPY.load(goldensDir.appending(path: "core_gpt_conditioning.npy")).asType(.float32)
    let textTokens = try NPY.load(goldensDir.appending(path: "core_gpt_latent__in0.npy"))
    let melCodes = try NPY.load(goldensDir.appending(path: "core_gpt_latent__in1.npy"))
    let goldenLatent = try NPY.load(goldensDir.appending(path: "core_gpt_latent.npy")).asType(.float32)
    print("  conditioning \(conditioning.shape)  text \(textTokens.shape)  mel \(melCodes.shape)  golden \(goldenLatent.shape)")

    print("→ building UnifiedVoiceV2 + loading gpt.safetensors")
    let model = try loadUnifiedVoiceV2()

    print("→ teacher-forced forwardLatent")
    let start = Date()
    let latent = model.forwardLatent(
        conditioning: conditioning, textTokens: textTokens, melCodes: melCodes)
    eval(latent)
    let elapsed = Date().timeIntervalSince(start)

    guard latent.shape == goldenLatent.shape else {
        fail("shape mismatch: \(latent.shape) vs golden \(goldenLatent.shape)")
    }
    let cos = cosine(latent, goldenLatent)
    let mad = maxAbsDiff(latent, goldenLatent)
    print(String(format: "  cos=%.7f  max_abs=%.5f  (%.2fs)", cos, mad, elapsed))

    if cos >= 0.999 {
        print("P2 GATE PASSED (cos ≥ 0.999)")
    } else {
        fail(String(format: "P2 gate cos %.7f < 0.999", cos))
    }
}

// MARK: - P3 front-end gate (fbank heads vs HF/torchaudio goldens)

func gateP3Frontend() throws {
    Device.setDefault(device: Device(.cpu))
    let fe = goldensDir.appending(path: "frontend")

    func check(_ name: String, _ ours: MLXArray, _ goldenFile: String,
               cosMin: Float, madMax: Float) throws {
        let golden = try NPY.load(fe.appending(path: goldenFile)).asType(.float32)
        let mine = ours.asType(.float32)
        guard mine.shape == golden.shape else {
            fail("\(name): shape \(mine.shape) vs golden \(golden.shape)")
        }
        let cos = cosine(mine, golden)
        let mad = maxAbsDiff(mine, golden)
        print(String(format: "  %@  cos=%.7f  max_abs=%.6f", name, cos, mad))
        if cos < cosMin || mad > madMax {
            fail(String(format: "%@ gate failed (cos %.7f < %.4f or max_abs %.6f > %.4f)",
                        name, cos, cosMin, mad, madMax))
        }
    }

    for (tag, audioFile) in [("ref", "audio_16k.npy"), ("synth", "synth_16k.npy")] {
        var wav = try NPY.load(fe.appending(path: audioFile)).asType(.float32)
        if wav.ndim == 2 { wav = wav[0] }  // (1, T) → (T,)
        print("→ \(tag): \(wav.dim(0)) samples")

        guard let sFbank = SeamlessFeatureExtractor.fbank(wav),
              let (features, mask) = SeamlessFeatureExtractor.callAsFeatures(wav),
              let cFbank = CampPlusFbank.fbankCMN(wav)
        else { fail("\(tag): audio shorter than one frame") }
        eval(sFbank, features, mask, cFbank)

        let suffix = tag == "ref" ? "" : "_synth"
        if tag == "ref" {
            try check("seamless fbank raw", sFbank, "seamless_fbank_raw.npy",
                      cosMin: 0.99999, madMax: 5e-3)
            let goldenMask = try NPY.load(fe.appending(path: "seamless_attention_mask.npy"))
            let maskSum = mask.asType(.int32).sum().item(Int32.self)
            let goldenSum = goldenMask.asType(.int32).sum().item(Int32.self)
            guard maskSum == goldenSum else { fail("mask sum \(maskSum) vs golden \(goldenSum)") }
            print("  attention mask sum \(maskSum) ✓")
        }
        try check("seamless input_features\(suffix)", features,
                  "seamless_input_features\(suffix).npy", cosMin: 0.9999, madMax: 2e-2)
        try check("campplus fbank cmn\(suffix)", cFbank,
                  "campplus_fbank_cmn\(suffix).npy", cosMin: 0.99999, madMax: 5e-3)
    }
    print("P3-FRONTEND GATE PASSED")
}

// MARK: - P3b w2v-BERT Conformer gate

func gateP3W2VBert() throws {
    Device.setDefault(device: Device(.cpu))
    let ladder = goldensDir.appending(path: "w2vbert")

    let defaultW2V = home.appending(
        path: ".cache/huggingface/hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b")
    let w2vDir = argValue("--w2v-weights").map { URL(fileURLWithPath: $0) } ?? defaultW2V

    print("→ building Wav2Vec2BertModel + loading model.safetensors")
    let model = Wav2Vec2BertModel()
    let declared = Set(model.parameters().flattened().map(\.0))

    let raw = try loadArrays(url: w2vDir.appending(path: "model.safetensors"))
    let sanitized = Wav2Vec2BertModel.sanitize(raw).mapValues { $0.asType(.float32) }

    let onDisk = Set(sanitized.keys)
    let missing = declared.subtracting(onDisk)
    let unused = onDisk.subtracting(declared)
    guard missing.isEmpty else { fail("missing keys: \(missing.sorted().prefix(8)) …") }
    guard unused.isEmpty else { fail("unused keys: \(unused.sorted().prefix(8)) …") }
    print("  keys: \(sanitized.count) (declared \(declared.count)); contract 0-missing/0-unused OK")

    try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
    eval(model)

    let mean = try NPY.load(ladder.appending(path: "semantic_mean.npy")).asType(.float32)
    let std = try NPY.load(ladder.appending(path: "semantic_std.npy")).asType(.float32)
    let golden = try NPY.load(goldensDir.appending(path: "frontend_ref__spk_cond_emb.npy")).asType(.float32)

    // --- Ladder A: injected golden input_features (isolates the Conformer port) ---
    print("→ ladder A: injected golden input_features")
    let inputFeatures = try NPY.load(ladder.appending(path: "input_features.npy")).asType(.float32)
    let attentionMask = try NPY.load(ladder.appending(path: "attention_mask.npy")).asType(.int32)
    print("  input \(inputFeatures.shape)  mask sum \(attentionMask.sum().item(Int32.self))")

    let start = Date()
    let (_, hs) = model(inputFeatures: inputFeatures, attentionMask: attentionMask)
    eval(hs)
    print(String(format: "  forward %.2fs (%d hidden states)", Date().timeIntervalSince(start), hs.count))

    var worst: (Float, Int) = (0, -1)
    for i in 0 ..< hs.count {
        let g = try NPY.load(ladder.appending(path: String(format: "hidden_states_%02d.npy", i)))
        let mad = maxAbsDiff(hs[i], g)
        if mad > worst.0 { worst = (mad, i) }
        if mad > 1e-3 { fail(String(format: "hidden_states[%d] max_abs %.6f > 1e-3", i, mad)) }
    }
    print(String(format: "  25 hidden states ≤ 1e-3 ✓ (worst %.2e at hs[%d])", worst.0, worst.1))

    let tapA = Wav2Vec2BertModel.semanticTap(hs, mean: mean, std: std)
    let madA = maxAbsDiff(tapA, golden)
    let cosA = cosine(tapA, golden)
    print(String(format: "  normalized hs[17] vs spk_cond_emb golden: cos=%.7f max_abs=%.2e", cosA, madA))
    guard madA < 1e-3 else { fail(String(format: "ladder-A tap max_abs %.6f > 1e-3", madA)) }

    // --- Chain B: full Swift front-end (audio → SeamlessFeatureExtractor → Conformer) ---
    print("→ chain B: full Swift chain from audio_16k")
    var wav = try NPY.load(goldensDir.appending(path: "frontend/audio_16k.npy")).asType(.float32)
    if wav.ndim == 2 { wav = wav[0] }
    guard let (features, mask) = SeamlessFeatureExtractor.callAsFeatures(wav) else {
        fail("feature extraction failed")
    }
    let (_, hsB) = model(inputFeatures: features, attentionMask: mask)
    let tapB = Wav2Vec2BertModel.semanticTap(hsB, mean: mean, std: std)
    eval(tapB)
    guard tapB.shape == golden.shape else {
        fail("chain-B shape \(tapB.shape) vs golden \(golden.shape)")
    }
    let cosB = cosine(tapB, golden)
    let madB = maxAbsDiff(tapB, golden)
    print(String(format: "  spk_cond_emb: cos=%.7f max_abs=%.5f", cosB, madB))
    guard cosB >= 0.999 else { fail(String(format: "chain-B cos %.7f < 0.999", cosB)) }

    print("P3B-W2VBERT GATE PASSED")
}

// MARK: - P3b MaskGCT RepCodec gate

func gateP3MaskGCT() throws {
    Device.setDefault(device: Device(.cpu))
    let ladder = goldensDir.appending(path: "maskgct")

    let defaultMGC = home.appending(
        path: ".cache/huggingface/hub/models--amphion--MaskGCT/snapshots/265c6cef07625665d0c28d2faafb1415562379dc/semantic_codec")
    let mgcDir = argValue("--maskgct-weights").map { URL(fileURLWithPath: $0) } ?? defaultMGC

    print("→ building RepCodec + loading semantic_codec/model.safetensors")
    let model = RepCodec()
    let declared = Set(model.parameters().flattened().map(\.0))

    let raw = try loadArrays(url: mgcDir.appending(path: "model.safetensors"))
    let sanitized = RepCodec.sanitize(raw).mapValues { $0.asType(.float32) }

    let missing = declared.subtracting(sanitized.keys)
    let unused = Set(sanitized.keys).subtracting(declared)
    guard missing.isEmpty else { fail("missing keys: \(missing.sorted().prefix(8)) …") }
    guard unused.isEmpty else { fail("unused keys: \(unused.sorted().prefix(8)) …") }
    print("  keys: \(sanitized.count) (declared \(declared.count)); contract 0-missing/0-unused OK")

    try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
    eval(model)

    let x = try NPY.load(goldensDir.appending(path: "frontend_ref__spk_cond_emb.npy")).asType(.float32)
    let sRefGolden = try NPY.load(goldensDir.appending(path: "frontend_ref__S_ref.npy")).asType(.float32)

    // Some ladder goldens were captured channels-first (B,C,T); transpose to our (B,T,C).
    func check(_ name: String, _ ours: MLXArray, _ file: String, thr: Float = 1e-3) throws {
        var g = try NPY.load(ladder.appending(path: file)).asType(.float32)
        if ours.shape != g.shape && g.ndim == 3 && ours.ndim == 3
            && g.shape == [ours.dim(0), ours.dim(2), ours.dim(1)] {
            g = g.transposed(0, 2, 1)
        }
        guard ours.shape == g.shape else { fail("\(name): shape \(ours.shape) vs golden \(g.shape)") }
        let mad = maxAbsDiff(ours, g)
        print(String(format: "  %@ max_abs = %.3e", name.padding(toLength: 20, withPad: " ", startingAt: 0), mad))
        if mad >= thr { fail(String(format: "%@ max_abs %.3e ≥ %.0e", name, mad, thr)) }
    }

    print("→ encoder ladder")
    let vb = model.encoderBackboneModule
    let ex = vb.embedLayer(x)
    try check("enc_vb_embed", ex, "enc_vb_embed.npy")
    var cx = vb.normLayer(ex)
    try check("enc_vb_norm", cx, "enc_vb_norm.npy")
    for (i, block) in vb.convnextBlocks.enumerated() {
        cx = block(cx)
        if [0, 5, 11].contains(i) {
            try check("enc_vb_convnext_\(i)", cx, "enc_vb_convnext_\(i).npy")
        }
    }
    let fx = vb.finalLayerNormLayer(cx)
    try check("enc_vb_final", fx, "enc_vb_final.npy")
    let z = model.encoderProjLayer(fx)
    try check("enc_out_z", z, "enc_out_z.npy")

    print("→ quantizer ladder")
    let fvq = model.quantizerModule.firstQuantizer
    let zE = fvq.inProject(z)
    try check("fvq_z_e", zE, "fvq_z_e.npy")
    let (zQ, indices) = fvq.decodeLatents(zE)
    try check("fvq_zq_prelatent", zQ, "fvq_zq_prelatent.npy")
    let ladIdx = try NPY.load(ladder.appending(path: "fvq_indices.npy"))
        .asType(.int32).reshaped(indices.shape)
    let idxMatches = (indices.asType(.int32) .== ladIdx).sum().item(Int32.self)
    let idxTotal = Int32(indices.size)
    print("  fvq_indices \(idxMatches)/\(idxTotal) exact")
    guard idxMatches == idxTotal else { fail("fvq_indices only \(idxMatches)/\(idxTotal) match") }
    try check("fvq_zq_out", fvq.outProject(zQ), "fvq_zq_out.npy")

    print("→ final gate: quantize() vs pipeline golden")
    let (codes, sRef) = model.quantize(x)
    eval(codes, sRef)
    let goldCodes = try NPY.load(ladder.appending(path: "codes.npy"))
        .asType(.int32).reshaped(codes.shape)
    let codeMatches = (codes.asType(.int32) .== goldCodes).sum().item(Int32.self)
    let codeTotal = Int32(codes.size)
    let lo = codes.min().item(Int32.self), hi = codes.max().item(Int32.self)
    print("  codes \(codeMatches)/\(codeTotal) exact (range \(lo)..\(hi))")
    guard codeMatches == codeTotal else { fail("codes only \(codeMatches)/\(codeTotal) match") }

    let madFinal = maxAbsDiff(sRef, sRefGolden)
    let cosFinal = cosine(sRef, sRefGolden)
    print(String(format: "  S_ref vs golden: cos=%.7f max_abs=%.3e", cosFinal, madFinal))
    guard madFinal < 1e-3 else { fail(String(format: "S_ref max_abs %.3e ≥ 1e-3", madFinal)) }

    print("P3B-MASKGCT GATE PASSED")
}

// MARK: - P3b CampPlus gate

func gateP3CampPlus() throws {
    Device.setDefault(device: Device(.cpu))
    let ladder = goldensDir.appending(path: "campplus")

    let defaultCPP = home.appending(path: "Development/_indextts2-oracle/campplus_cn_common.safetensors")
    let cppFile = argValue("--campplus-weights").map { URL(fileURLWithPath: $0) } ?? defaultCPP

    print("→ building CAMPPlus(80, 192) + loading campplus_cn_common.safetensors")
    let model = CAMPPlus()
    model.train(false)  // BatchNorms must use running stats
    let declared = Set(model.parameters().flattened().map(\.0))

    let raw = try loadArrays(url: cppFile)
    let sanitized = CAMPPlus.sanitize(raw).mapValues { $0.asType(.float32) }

    let missing = declared.subtracting(sanitized.keys)
    let unused = Set(sanitized.keys).subtracting(declared)
    guard missing.isEmpty else { fail("missing keys: \(missing.sorted().prefix(8)) …") }
    guard unused.isEmpty else { fail("unused keys: \(unused.sorted().prefix(8)) …") }
    print("  keys: \(sanitized.count) (declared \(declared.count)); contract 0-missing/0-unused OK")

    try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
    eval(model)

    func check(_ name: String, _ ours: MLXArray, thr: Float = 1e-3) throws {
        var g = try NPY.load(ladder.appending(path: "\(name).npy")).asType(.float32)
        if ours.shape != g.shape && g.ndim == 3 && ours.ndim == 3
            && g.shape == [ours.dim(0), ours.dim(2), ours.dim(1)] {
            g = g.transposed(0, 2, 1)
        }
        guard ours.shape == g.shape else { fail("\(name): shape \(ours.shape) vs golden \(g.shape)") }
        let mad = maxAbsDiff(ours, g)
        print(String(format: "  %@ max_abs = %.3e", name.padding(toLength: 20, withPad: " ", startingAt: 0), mad))
        if mad >= thr { fail(String(format: "%@ max_abs %.3e ≥ %.0e", name, mad, thr)) }
    }

    print("→ ladder (input = campplus_fbank_cmn golden)")
    let feat = try NPY.load(goldensDir.appending(path: "frontend/campplus_fbank_cmn.npy")).asType(.float32)
    var h = feat[.newAxis, 0..., 0...]  // (1, T, 80)
    h = model.headModule(h)
    try check("head_out", h)
    for (name, stage) in model.xvectorModule.stages {
        h = stage(h)
        try check(name, h)
    }

    print("→ final gate: style vs pipeline golden")
    let golden = try NPY.load(goldensDir.appending(path: "frontend_ref__style.npy")).asType(.float32)
    let style = model(feat[.newAxis, 0..., 0...])
    eval(style)
    guard style.shape == golden.shape else { fail("style shape \(style.shape) vs \(golden.shape)") }
    let cos = cosine(style, golden)
    let mad = maxAbsDiff(style, golden)
    print(String(format: "  style: cos=%.7f max_abs=%.3e", cos, mad))
    guard mad < 1e-3 else { fail(String(format: "style max_abs %.3e ≥ 1e-3", mad)) }

    // Full Swift chain: audio → CampPlusFbank.fbankCMN → CAMPPlus.
    print("→ chain: full Swift front-end from audio_16k")
    var wav = try NPY.load(goldensDir.appending(path: "frontend/audio_16k.npy")).asType(.float32)
    if wav.ndim == 2 { wav = wav[0] }
    guard let cmn = CampPlusFbank.fbankCMN(wav) else { fail("fbank failed") }
    let styleChain = model(cmn[.newAxis, 0..., 0...])
    eval(styleChain)
    let cosChain = cosine(styleChain, golden)
    let madChain = maxAbsDiff(styleChain, golden)
    print(String(format: "  style (chain): cos=%.7f max_abs=%.5f", cosChain, madChain))
    guard cosChain >= 0.999 else { fail(String(format: "chain cos %.7f < 0.999", cosChain)) }

    print("P3B-CAMPPLUS GATE PASSED")
}

// MARK: - P3b GPT conditioner gate (conformer + perceiver + emovec + full conditioning)

func gateP3Conditioners() throws {
    Device.setDefault(device: Device(.cpu))

    print("→ building UnifiedVoiceV2 + loading gpt.safetensors")
    let model = try loadUnifiedVoiceV2()

    // Oracle inputs/goldens (fp16-weight MLX-Metal capture → cos gates, like P2).
    let spkCondEmb = try NPY.load(goldensDir.appending(path: "frontend_ref__spk_cond_emb.npy")).asType(.float32)
    let spkNCL = spkCondEmb.transposed(0, 2, 1)  // (1, 1024, T) NCL, as generate_v2 feeds it

    func gate(_ name: String, _ ours: MLXArray, _ goldenFile: String) throws {
        let golden = try NPY.load(goldensDir.appending(path: goldenFile)).asType(.float32)
        guard ours.shape == golden.shape else {
            fail("\(name): shape \(ours.shape) vs golden \(golden.shape)")
        }
        let cos = cosine(ours, golden)
        let mad = maxAbsDiff(ours, golden)
        print(String(format: "  %@ cos=%.7f max_abs=%.5f",
                     name.padding(toLength: 22, withPad: " ", startingAt: 0), cos, mad))
        guard cos >= 0.999 else { fail(String(format: "%@ cos %.7f < 0.999", name, cos)) }
    }

    print("→ get_conditioning (speaker conformer + 32-latent perceiver)")
    let speechCond = model.getConditioning(spkNCL)
    eval(speechCond)
    try gate("speech_cond", speechCond, "core_gpt_speech_cond.npy")

    print("→ get_emovec (emotion conformer + 1-latent perceiver + emovec/emo layers)")
    let baseEmovec = model.getEmovec(spkNCL)
    eval(baseEmovec)
    try gate("base_emovec", baseEmovec, "core_gpt_base_emovec.npy")

    print("→ prepare_conditioning_latents (emotion 'happy' α=0.6 blend)")
    // generate_v2: weights={happy: 1.0}·α → weight_sum=0.6 → emo_vec = mat + 0.4·base
    let emovecMat = try NPY.load(goldensDir.appending(path: "frontend_emovec_mat.npy")).asType(.float32)
    let emoVec = emovecMat + 0.4 * baseEmovec
    let conditioning = model.prepareConditioningLatents(
        speechConditioning: speechCond, emoVec: emoVec, batchSize: 1)
    eval(conditioning)
    try gate("conditioning", conditioning, "core_gpt_conditioning.npy")

    print("P3B-CONDITIONERS GATE PASSED")
}

// MARK: - Shared S2Mel loader (P4/P6)

func loadS2Mel() throws -> S2Mel {
    let model = S2Mel()
    let declared = Set(model.parameters().flattened().map(\.0))

    let raw = try loadArrays(url: weightsDir.appending(path: "s2mel.safetensors"))
    let sanitized = S2Mel.sanitize(raw)

    let missing = declared.subtracting(sanitized.keys)
    let unused = Set(sanitized.keys).subtracting(declared)
    guard missing.isEmpty else { fail("missing keys: \(missing.sorted().prefix(8)) …") }
    guard unused.isEmpty else { fail("unused keys: \(unused.sorted().prefix(8)) …") }
    print("  keys: \(sanitized.count) (declared \(declared.count)); contract 0-missing/0-unused OK")

    let fp32 = sanitized.mapValues { $0.asType(.float32) }
    try model.update(parameters: ModuleParameters.unflattened(fp32), verify: .all)
    eval(model)
    return model
}

// MARK: - P4 gate: S2Mel (gpt_layer → length_regulator → CFM)

// The original Stage-0 cfm_mel golden sits past the AR sampler's RNG consumption and is not
// reproducible from seed(42) alone; P4 gates against the seed-42 REPLAY goldens
// (tools/dump_s2mel_replay.py — deterministic stages verified bitwise vs the originals).
func gateP4() throws {
    Device.setDefault(device: Device(.cpu))

    print("→ building S2Mel + loading s2mel.safetensors")
    let model = try loadS2Mel()

    func gate(_ name: String, _ ours: MLXArray, _ goldenFile: String, cosMin: Float = 0.999) throws {
        let golden = try NPY.load(goldensDir.appending(path: goldenFile)).asType(.float32)
        guard ours.shape == golden.shape else {
            fail("\(name): shape \(ours.shape) vs golden \(golden.shape)")
        }
        let cos = cosine(ours, golden)
        let mad = maxAbsDiff(ours, golden)
        print(String(format: "  %@ cos=%.7f max_abs=%.5f",
                     name.padding(toLength: 22, withPad: " ", startingAt: 0), cos, mad))
        guard cos >= cosMin else { fail(String(format: "%@ cos %.7f < %.4f", name, cos, cosMin)) }
    }

    // --- gpt_layer ---
    print("→ gpt_layer (1280→256→128→1024)")
    let latent = try NPY.load(goldensDir.appending(path: "core_gpt_latent.npy")).asType(.float32)
    let gptOut = model.gptLayerModule(latent)
    eval(gptOut)
    try gate("gptlayer", gptOut, "core_s2mel_gptlayer.npy")

    // --- length_regulator (golden-injected input for stage isolation) ---
    print("→ length_regulator (nearest ×1.72 + conv-norm-mish ×4)")
    let gptGolden = try NPY.load(goldensDir.appending(path: "core_s2mel_gptlayer.npy")).asType(.float32)
    let vq2emb = try NPY.load(goldensDir.appending(path: "core_vq2emb.npy")).asType(.float32)
    let sInfer = vq2emb.transposed(0, 2, 1) + gptGolden
    let codeLen = latent.dim(1)
    let targetLengths = MLXArray([Int32(Float(codeLen) * 1.72)])
    let lenregOut = model.lengthRegulatorModule(sInfer, ylens: targetLengths)
    eval(lenregOut)
    try gate("lenreg", lenregOut, "core_s2mel_lenreg__0.npy")

    // --- CFM inputs (golden-injected) ---
    let lenregGolden = try NPY.load(goldensDir.appending(path: "core_s2mel_lenreg__0.npy")).asType(.float32)
    let promptCondition = try NPY.load(goldensDir.appending(path: "frontend_ref__prompt_condition.npy")).asType(.float32)
    let refMel = try NPY.load(goldensDir.appending(path: "frontend_ref__ref_mel.npy")).asType(.float32)
    let style = try NPY.load(goldensDir.appending(path: "frontend_ref__style.npy")).asType(.float32)
    let catCondition = concatenated([promptCondition, lenregGolden], axis: 1)
    let xLens = MLXArray([Int32(catCondition.dim(1))])

    // --- RNG cross-binding check: seed(42) → first normal draw must equal the replay z ---
    print("→ RNG stream check (seed 42 → normal(1,80,\(catCondition.dim(1))))")
    let zGolden = try NPY.load(goldensDir.appending(path: "core_s2mel_cfm_z_seed42.npy")).asType(.float32)
    MLXRandom.seed(42)
    let zSwift = MLXRandom.normal([1, 80, catCondition.dim(1)])
    eval(zSwift)
    let zMad = maxAbsDiff(zSwift, zGolden)
    print(String(format: "  z draw max_abs=%.2e %@", zMad,
                 zMad == 0 ? "(bit-identical)" : "(NOT bit-identical — CFM gate uses injected z)"))

    // --- single DiT forward (step-1 ladder; isolates the estimator from the ODE loop) ---
    print("→ DiT single forward (step 1, stacked CFG batch)")
    let promptLen = refMel.dim(2)
    let T = catCondition.dim(1)
    let promptX = concatenated([refMel, MLXArray.zeros([1, 80, T - promptLen])], axis: 2)
    let x0 = concatenated(
        [MLXArray.zeros([1, 80, promptLen]), zGolden[0..., 0..., promptLen...]], axis: 2)
    let stackedDphi = model.cfmModule.estimatorModule(
        concatenated([x0, x0], axis: 0),
        promptX: concatenated([promptX, MLXArray.zeros(like: promptX)], axis: 0),
        xLens: xLens,
        t: MLXArray([Float(0), Float(0)]),
        style: concatenated([style, MLXArray.zeros(like: style)], axis: 0),
        cond: concatenated([catCondition, MLXArray.zeros(like: catCondition)], axis: 0))
    eval(stackedDphi)
    try gate("dit_step1", stackedDphi, "core_s2mel_dit_step1_seed42.npy")

    // --- full 25-step CFM (injected z isolates the loop from RNG) ---
    print("→ CFM inference (25 steps, cfg_rate 0.7, injected z)")
    let start = Date()
    let mel = model.cfmModule.inference(
        mu: catCondition, xLens: xLens, prompt: refMel, style: style,
        nTimesteps: 25, temperature: 1.0, inferenceCfgRate: 0.7, injectedZ: zGolden)
    eval(mel)
    print(String(format: "  (%.2fs)", Date().timeIntervalSince(start)))
    try gate("cfm_mel", mel, "core_s2mel_cfm_mel_seed42.npy")

    print("P4 GATE PASSED")
}

// MARK: - Shared BigVGAN loader (P5/P6)

func loadBigVGAN() throws -> BigVGANV2 {
    let model = BigVGANV2()
    let declared = Set(model.parameters().flattened().map(\.0))

    let raw = try loadArrays(url: weightsDir.appending(path: "bigvgan.safetensors"))

    let missing = declared.subtracting(raw.keys)
    let unused = Set(raw.keys).subtracting(declared)
    guard missing.isEmpty else { fail("missing keys: \(missing.sorted().prefix(8)) …") }
    guard unused.isEmpty else { fail("unused keys: \(unused.sorted().prefix(8)) …") }
    print("  keys: \(raw.count) (declared \(declared.count)); contract 0-missing/0-unused OK")

    let fp32 = raw.mapValues { $0.asType(.float32) }
    try model.update(parameters: ModuleParameters.unflattened(fp32), verify: .all)
    eval(model)
    return model
}

// MARK: - P5 gate: BigVGAN v2 vocoder

func gateP5() throws {
    Device.setDefault(device: Device(.cpu))

    print("→ building BigVGANV2 + loading bigvgan.safetensors")
    let model = try loadBigVGAN()

    func gate(_ name: String, mel: MLXArray, goldenFile: String) throws {
        let promptLen = 431
        let wav = model(mel[0..., 0..., promptLen...])
        eval(wav)
        let golden = try NPY.load(goldensDir.appending(path: goldenFile)).asType(.float32)
        guard wav.shape == golden.shape else {
            fail("\(name): shape \(wav.shape) vs golden \(golden.shape)")
        }
        let cos = cosine(wav, golden)
        let mad = maxAbsDiff(wav, golden)
        print(String(format: "  %@ cos=%.7f max_abs=%.5f",
                     name.padding(toLength: 22, withPad: " ", startingAt: 0), cos, mad))
        guard cos >= 0.999 else { fail(String(format: "%@ cos %.7f < 0.999", name, cos)) }
    }

    // --- per-stage ladder (goldens/bigvgan, dumped by tools/dump_bigvgan_ladder.py) ---
    // Report-only: stage max_abs drifts benignly vs the Metal-fp16 goldens (the Python
    // reference itself shows max_abs 0.026 e2e on the CPU stream); a structural break shows
    // as orders of magnitude (the snake-alias bug read 1.5e0 here). Cosine also reported —
    // it stays ~1.0 for benign drift. The hard gate is the final waveform cosine below.
    let ladder = goldensDir.appending(path: "bigvgan")
    if FileManager.default.fileExists(atPath: ladder.path) {
        func check(_ name: String, _ ours: MLXArray) throws {
            let g = try NPY.load(ladder.appending(path: "\(name).npy")).asType(.float32)
            guard ours.shape == g.shape else { fail("\(name): shape \(ours.shape) vs \(g.shape)") }
            print(String(format: "  %@ max_abs = %.3e  cos=%.7f",
                         name.padding(toLength: 18, withPad: " ", startingAt: 0),
                         maxAbsDiff(ours, g), cosine(ours, g)))
        }

        print("→ ladder: anti-alias primitive probes")
        let melL = try NPY.load(goldensDir.appending(path: "core_s2mel_cfm_mel_seed42.npy"))
            .asType(.float32)[0..., 0..., 431...]
        var x = model.convPreLayer(melL.transposed(0, 2, 1)).transposed(0, 2, 1)
        try check("conv_pre", x)

        let probe = try NPY.load(ladder.appending(path: "probe_in.npy")).asType(.float32)
        let act1d = model.resblockModules[0].activationModules[0]
        let probeUp = act1d.upsample(probe)
        try check("probe_upsample", probeUp)
        try check("probe_act", act1d.applyAct(probeUp))
        try check("probe_act1d", act1d(probe))
        try check("probe_downsample", act1d.downsample(probe))

        print("→ ladder: upsample stages")
        for i in 0 ..< model.numUpsamples {
            x = model.upsLayers[i](x.transposed(0, 2, 1)).transposed(0, 2, 1)
            try check("ups_\(i)", x)
            var xs: MLXArray? = nil
            for j in 0 ..< model.numKernels {
                let res = model.resblockModules[i * model.numKernels + j](x)
                xs = xs.map { $0 + res } ?? res
            }
            x = xs! / Float(model.numKernels)
            try check("stage_\(i)", x)
        }
        try check("activation_post", model.activationPostModule(x))
    }

    print("→ vocoding seed-42 replay mel (trim prompt 431 → (1,80,190))")
    let melSeed42 = try NPY.load(goldensDir.appending(path: "core_s2mel_cfm_mel_seed42.npy")).asType(.float32)
    let start = Date()
    try gate("bigvgan_wav(seed42)", mel: melSeed42, goldenFile: "core_bigvgan_wav_seed42.npy")
    print(String(format: "  (%.2fs)", Date().timeIntervalSince(start)))

    print("→ vocoding ORIGINAL Stage-0 mel golden")
    let melOrig = try NPY.load(goldensDir.appending(path: "core_s2mel_cfm_mel.npy")).asType(.float32)
    try gate("bigvgan_wav(orig)", mel: melOrig, goldenFile: "core_bigvgan_wav.npy")

    print("P5 GATE PASSED")
}

// MARK: - Entry

let mode = CommandLine.arguments.dropFirst().first ?? "p2"
do {
    switch mode {
    case "p2": try gateP2()
    case "p3fe": try gateP3Frontend()
    case "p3w2v": try gateP3W2VBert()
    case "p3mgc": try gateP3MaskGCT()
    case "p3cpp": try gateP3CampPlus()
    case "p3cond": try gateP3Conditioners()
    case "p4": try gateP4()
    case "p5": try gateP5()
    default: fail("unknown mode \(mode) (expected: p2 | p3fe | p3w2v | p3mgc | p3cpp | p3cond | p4 | p5)")
    }
} catch {
    fail("\(error)")
}
