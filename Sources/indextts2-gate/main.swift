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

    print("→ building UnifiedVoiceV2 (P2 subset) + loading gpt.safetensors subset")
    let model = UnifiedVoiceV2()
    let declared = Set(model.parameters().flattened().map(\.0))

    let all = try loadArrays(url: weightsDir.appending(path: "gpt.safetensors"))
    let subset = all.filter { declared.contains($0.key) }

    // Declared-subset key contract: 0 missing / 0 unused within the declared tree.
    let onDisk = Set(subset.keys)
    let missing = declared.subtracting(onDisk)
    let p3Families = ["conditioning_encoder.", "perceiver_encoder.", "emo_conditioning_encoder.",
                      "emo_perceiver_encoder.", "emo_layer.", "emovec_layer."]
    let unusedOutsideP3 = Set(all.keys).subtracting(declared)
        .filter { key in !p3Families.contains(where: { key.hasPrefix($0) }) }
    guard missing.isEmpty else { fail("missing keys: \(missing.sorted().prefix(8)) …") }
    guard unusedOutsideP3.isEmpty else { fail("unexpected non-P3 keys on disk: \(unusedOutsideP3.sorted().prefix(8)) …") }
    print("  subset keys: \(subset.count) (declared \(declared.count)); P3 families deferred: OK")

    // fp32 upcast, materialized before the forward (watchdog corollary).
    let fp32 = subset.mapValues { $0.asType(.float32) }
    try model.update(parameters: ModuleParameters.unflattened(fp32), verify: .all)
    eval(model)

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

// MARK: - Entry

let mode = CommandLine.arguments.dropFirst().first ?? "p2"
do {
    switch mode {
    case "p2": try gateP2()
    default: fail("unknown mode \(mode) (expected: p2)")
    }
} catch {
    fail("\(error)")
}
