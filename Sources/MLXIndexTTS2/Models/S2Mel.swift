// S2Mel.swift — complete Semantic-to-Mel model (gpt_layer → length_regulator → CFM).
//
// Isomorphic port of donor `mlx_indextts/models/s2mel/s2mel.py`. Constructor defaults ARE the
// resolved config for this checkpoint (generate_v2 builds `S2Mel()` bare; cross-checked against
// config.yaml s2mel section 2026-07-08 — only DiT block_size differs, which is inert).
//
// Weight source: s2mel.safetensors (264 keys, already in donor-MLX layout — no conv transposes).
// sanitize remaps only the heterogeneous torch-Sequential numeric keys:
//   length_regulator.model.{0,3,6,9}  -> length_regulator.convs.{0..3}
//   length_regulator.model.{1,4,7,10} -> length_regulator.norms.{0..3}
//   length_regulator.model.12         -> length_regulator.out_proj
//   ...final_layer.adaLN_modulation.layers.1.* -> ...final_layer.adaLN_modulation.linear.*

import Foundation
import MLX
import MLXNN

/// Linear(1280→256→128→1024) projection of GPT output to content dim (donor GPTLayer).
public final class GPTLayer: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    public init(inDim: Int = 1280, hiddenDims: [Int] = [256, 128], outDim: Int = 1024) {
        let dims = [inDim] + hiddenDims + [outDim]
        self._layers.wrappedValue = (0 ..< dims.count - 1).map { Linear(dims[$0], dims[$0 + 1]) }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers { x = layer(x) }
        return x
    }
}

public final class S2Mel: Module {
    @ModuleInfo(key: "gpt_layer") var gptLayer: GPTLayer
    @ModuleInfo(key: "length_regulator") var lengthRegulator: InterpolateRegulator
    @ModuleInfo(key: "cfm") var cfm: CFM

    public override init() {
        self._gptLayer.wrappedValue = GPTLayer()
        self._lengthRegulator.wrappedValue = InterpolateRegulator()
        self._cfm.wrappedValue = CFM()
    }

    public var gptLayerModule: GPTLayer { gptLayer }
    public var lengthRegulatorModule: InterpolateRegulator { lengthRegulator }
    public var cfmModule: CFM { cfm }

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        let lrModelMap: [String: String] = [
            "model.0.": "convs.0.", "model.3.": "convs.1.",
            "model.6.": "convs.2.", "model.9.": "convs.3.",
            "model.1.": "norms.0.", "model.4.": "norms.1.",
            "model.7.": "norms.2.", "model.10.": "norms.3.",
            "model.12.": "out_proj.",
        ]
        for (k, v) in weights {
            var nk = k
            if nk.hasPrefix("length_regulator.model.") {
                let rest = String(nk.dropFirst("length_regulator.".count))
                for (old, new) in lrModelMap where rest.hasPrefix(old) {
                    nk = "length_regulator." + new + rest.dropFirst(old.count)
                    break
                }
            } else if nk.contains(".adaLN_modulation.layers.1.") {
                nk = nk.replacingOccurrences(
                    of: ".adaLN_modulation.layers.1.", with: ".adaLN_modulation.linear.")
            }
            out[nk] = v
        }
        return out
    }
}
