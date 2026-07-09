// CampPlus.swift — CAM++ DTDNN speaker embedding (3D-Speaker, funasr/campplus).
//
// Isomorphic port of solar2ain's vendored torch reference
// `indextts/s2mel/modules/campplus/{DTDNN.py,layers.py}` (Apache-2.0), instantiated as
// CAMPPlus(feat_dim=80, embedding_size=192). Input = kaldi 80-fbank − time-mean
// (`CampPlusFbank.fbankCMN`), (B, T, 80) channels-last → style (B, 192).
//
// Layout notes: torch runs channels-first; this port is channels-last throughout.
// FCM treats (freq, time) as the 2D image (H=F, W=T); its (B,C,F',T) → (B,C·F',T) flatten
// becomes (B,F',T,C) → transposed(0,2,3,1) → (B,T,C·F') — C-major, matching torch.
// BatchNorms run in inference mode (running stats) — the gate calls `train(false)`.
// avg_pool1d(ceil_mode=True) in CAMLayer divides the partial tail window by its TRUE length.

import Foundation
import MLX
import MLXNN

/// `get_nonlinear` for the two config strings this checkpoint uses:
/// "batchnorm-relu" (affine BN → ReLU) and "batchnorm_" (affine-free BN only).
public final class CampPlusNonLinear: Module {
    let hasRelu: Bool
    @ModuleInfo(key: "batchnorm") var batchnorm: BatchNorm

    public init(channels: Int, affine: Bool = true, relu: Bool = true) {
        self.hasRelu = relu
        self._batchnorm.wrappedValue = BatchNorm(featureCount: channels, affine: affine)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = batchnorm(x)
        return hasRelu ? relu(x) : x
    }
}

public final class CampPlusBasicResBlock: Module {
    /// torch `shortcut` is Sequential(conv, bn) with numeric keys 0/1; those are remapped to
    /// conv/bn in `sanitize` (numeric module keys collide with array-index unflattening).
    public final class Shortcut: Module {
        @ModuleInfo(key: "conv") var conv: Conv2d
        @ModuleInfo(key: "bn") var bn: BatchNorm

        public init(inPlanes: Int, planes: Int, stride: Int) {
            self._conv.wrappedValue = Conv2d(
                inputChannels: inPlanes, outputChannels: planes, kernelSize: 1,
                stride: IntOrPair((stride, 1)), bias: false)
            self._bn.wrappedValue = BatchNorm(featureCount: planes)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray { bn(conv(x)) }
    }

    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm
    @ModuleInfo(key: "shortcut") var shortcut: Shortcut?

    public init(inPlanes: Int, planes: Int, stride: Int = 1) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inPlanes, outputChannels: planes, kernelSize: 3,
            stride: IntOrPair((stride, 1)), padding: 1, bias: false)
        self._bn1.wrappedValue = BatchNorm(featureCount: planes)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: planes, outputChannels: planes, kernelSize: 3,
            stride: 1, padding: 1, bias: false)
        self._bn2.wrappedValue = BatchNorm(featureCount: planes)
        if stride != 1 || inPlanes != planes {
            self._shortcut.wrappedValue = Shortcut(inPlanes: inPlanes, planes: planes, stride: stride)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        out = out + (shortcut.map { $0(x) } ?? x)
        return relu(out)
    }
}

/// Front-end conv module: (B, T, 80) → (B, T, 320).
public final class CampPlusFCM: Module {
    public let outChannels: Int

    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "layer1") var layer1: [CampPlusBasicResBlock]
    @ModuleInfo(key: "layer2") var layer2: [CampPlusBasicResBlock]
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm

    public init(mChannels: Int = 32, featDim: Int = 80) {
        self.outChannels = mChannels * (featDim / 8)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: mChannels, kernelSize: 3,
            stride: 1, padding: 1, bias: false)
        self._bn1.wrappedValue = BatchNorm(featureCount: mChannels)
        self._layer1.wrappedValue = [
            CampPlusBasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 2),
            CampPlusBasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 1),
        ]
        self._layer2.wrappedValue = [
            CampPlusBasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 2),
            CampPlusBasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 1),
        ]
        self._conv2.wrappedValue = Conv2d(
            inputChannels: mChannels, outputChannels: mChannels, kernelSize: 3,
            stride: IntOrPair((2, 1)), padding: 1, bias: false)
        self._bn2.wrappedValue = BatchNorm(featureCount: mChannels)
    }

    /// x: (B, T, F) → (B, T, C·F/8).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // (B, T, F) → image (B, H=F, W=T, C=1)
        var out = x.transposed(0, 2, 1).expandedDimensions(axis: -1)
        out = relu(bn1(conv1(out)))
        for block in layer1 { out = block(out) }
        for block in layer2 { out = block(out) }
        out = relu(bn2(conv2(out)))                    // (B, F/8, T, C)
        // torch: (B, C, F', T) → (B, C·F', T), C-major flatten
        out = out.transposed(0, 2, 3, 1)               // (B, T, C, F')
        return out.reshaped(out.dim(0), out.dim(1), out.dim(2) * out.dim(3))
    }
}

public final class CampPlusTDNNLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: CampPlusNonLinear

    public init(inChannels: Int, outChannels: Int, kernelSize: Int,
                stride: Int = 1, dilation: Int = 1) {
        let padding = (kernelSize - 1) / 2 * dilation
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize,
            stride: stride, padding: padding, dilation: dilation, bias: false)
        self._nonlinear.wrappedValue = CampPlusNonLinear(channels: outChannels)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        nonlinear(linear(x))
    }
}

/// Context-aware masking layer: y = conv(x) · sigmoid(linear2(relu(linear1(mean + segMean)))).
public final class CampPlusCAMLayer: Module {
    public static let segLen = 100

    @ModuleInfo(key: "linear_local") var linearLocal: Conv1d
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "linear2") var linear2: Conv1d

    public init(bnChannels: Int, outChannels: Int, kernelSize: Int,
                dilation: Int, reduction: Int = 2) {
        let padding = (kernelSize - 1) / 2 * dilation
        self._linearLocal.wrappedValue = Conv1d(
            inputChannels: bnChannels, outputChannels: outChannels, kernelSize: kernelSize,
            padding: padding, dilation: dilation, bias: false)
        self._linear1.wrappedValue = Conv1d(
            inputChannels: bnChannels, outputChannels: bnChannels / reduction, kernelSize: 1)
        self._linear2.wrappedValue = Conv1d(
            inputChannels: bnChannels / reduction, outputChannels: outChannels, kernelSize: 1)
    }

    /// avg_pool1d(k=100, s=100, ceil_mode=True) + per-segment broadcast back, truncated to T.
    /// The partial tail window averages over its TRUE length (torch divisor semantics).
    func segPooling(_ x: MLXArray) -> MLXArray {
        let T = x.dim(1)
        var segments: [MLXArray] = []
        var start = 0
        while start < T {
            let end = Swift.min(start + Self.segLen, T)
            let segMean = x[0..., start ..< end, 0...].mean(axis: 1, keepDims: true)
            segments.append(broadcast(segMean, to: [x.dim(0), end - start, x.dim(2)]))
            start = end
        }
        return concatenated(segments, axis: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = linearLocal(x)
        var context = x.mean(axis: 1, keepDims: true) + segPooling(x)
        context = relu(linear1(context))
        let m = sigmoid(linear2(context))
        return y * m
    }
}

public final class CampPlusCAMDenseTDNNLayer: Module {
    @ModuleInfo(key: "nonlinear1") var nonlinear1: CampPlusNonLinear
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "nonlinear2") var nonlinear2: CampPlusNonLinear
    @ModuleInfo(key: "cam_layer") var camLayer: CampPlusCAMLayer

    public init(inChannels: Int, outChannels: Int, bnChannels: Int,
                kernelSize: Int, dilation: Int = 1) {
        self._nonlinear1.wrappedValue = CampPlusNonLinear(channels: inChannels)
        self._linear1.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: bnChannels, kernelSize: 1, bias: false)
        self._nonlinear2.wrappedValue = CampPlusNonLinear(channels: bnChannels)
        self._camLayer.wrappedValue = CampPlusCAMLayer(
            bnChannels: bnChannels, outChannels: outChannels,
            kernelSize: kernelSize, dilation: dilation)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        camLayer(nonlinear2(linear1(nonlinear1(x))))
    }
}

/// Dense block: x = cat([x, layer(x)]) per layer (channel axis).
/// torch keys tdnnd1..tdnndN are remapped to layers.0..N-1 in `sanitize`.
public final class CampPlusCAMDenseTDNNBlock: Module {
    @ModuleInfo(key: "layers") var layers: [CampPlusCAMDenseTDNNLayer]

    public init(numLayers: Int, inChannels: Int, outChannels: Int, bnChannels: Int,
                kernelSize: Int, dilation: Int) {
        self._layers.wrappedValue = (0 ..< numLayers).map { i in
            CampPlusCAMDenseTDNNLayer(
                inChannels: inChannels + i * outChannels, outChannels: outChannels,
                bnChannels: bnChannels, kernelSize: kernelSize, dilation: dilation)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            x = concatenated([x, layer(x)], axis: -1)
        }
        return x
    }
}

public final class CampPlusTransitLayer: Module {
    @ModuleInfo(key: "nonlinear") var nonlinear: CampPlusNonLinear
    @ModuleInfo(key: "linear") var linear: Conv1d

    public init(inChannels: Int, outChannels: Int) {
        self._nonlinear.wrappedValue = CampPlusNonLinear(channels: inChannels)
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(nonlinear(x))
    }
}

public final class CampPlusDenseLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: CampPlusNonLinear

    public init(inChannels: Int, outChannels: Int) {
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        self._nonlinear.wrappedValue = CampPlusNonLinear(
            channels: outChannels, affine: false, relu: false)  // config_str="batchnorm_"
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // 2-D input (B, C): length-1 sequence through the 1×1 conv (torch unsqueeze(-1))
        if x.ndim == 2 {
            return nonlinear(linear(x.expandedDimensions(axis: 1)).squeezed(axis: 1))
        }
        return nonlinear(linear(x))
    }
}

/// Mean + unbiased-std pooling over time: (B, T, C) → (B, 2C).
func campPlusStatsPool(_ x: MLXArray) -> MLXArray {
    let mean = x.mean(axis: 1)
    let std = sqrt(variance(x, axis: 1, ddof: 1))
    return concatenated([mean, std], axis: -1)
}

/// The `xvector` Sequential (named torch submodules).
public final class CampPlusXVector: Module {
    @ModuleInfo(key: "tdnn") var tdnn: CampPlusTDNNLayer
    @ModuleInfo(key: "block1") var block1: CampPlusCAMDenseTDNNBlock
    @ModuleInfo(key: "transit1") var transit1: CampPlusTransitLayer
    @ModuleInfo(key: "block2") var block2: CampPlusCAMDenseTDNNBlock
    @ModuleInfo(key: "transit2") var transit2: CampPlusTransitLayer
    @ModuleInfo(key: "block3") var block3: CampPlusCAMDenseTDNNBlock
    @ModuleInfo(key: "transit3") var transit3: CampPlusTransitLayer
    @ModuleInfo(key: "out_nonlinear") var outNonlinear: CampPlusNonLinear
    @ModuleInfo(key: "dense") var dense: CampPlusDenseLayer

    public init(inChannels: Int, embeddingSize: Int, growthRate: Int = 32, bnSize: Int = 4,
                initChannels: Int = 128) {
        var channels = initChannels
        self._tdnn.wrappedValue = CampPlusTDNNLayer(
            inChannels: inChannels, outChannels: initChannels, kernelSize: 5, stride: 2)

        var blocks: [CampPlusCAMDenseTDNNBlock] = []
        var transits: [CampPlusTransitLayer] = []
        for (numLayers, kernelSize, dilation) in [(12, 3, 1), (24, 3, 2), (16, 3, 2)] {
            blocks.append(CampPlusCAMDenseTDNNBlock(
                numLayers: numLayers, inChannels: channels, outChannels: growthRate,
                bnChannels: bnSize * growthRate, kernelSize: kernelSize, dilation: dilation))
            channels += numLayers * growthRate
            transits.append(CampPlusTransitLayer(inChannels: channels, outChannels: channels / 2))
            channels /= 2
        }
        self._block1.wrappedValue = blocks[0]
        self._transit1.wrappedValue = transits[0]
        self._block2.wrappedValue = blocks[1]
        self._transit2.wrappedValue = transits[1]
        self._block3.wrappedValue = blocks[2]
        self._transit3.wrappedValue = transits[2]
        self._outNonlinear.wrappedValue = CampPlusNonLinear(channels: channels)
        self._dense.wrappedValue = CampPlusDenseLayer(inChannels: channels * 2, outChannels: embeddingSize)
    }

    // Ladder access for the parity gate.
    public var stages: [(String, (MLXArray) -> MLXArray)] {
        [
            ("tdnn_out", { self.tdnn($0) }),
            ("block1_out", { self.block1($0) }),
            ("transit1_out", { self.transit1($0) }),
            ("block2_out", { self.block2($0) }),
            ("transit2_out", { self.transit2($0) }),
            ("block3_out", { self.block3($0) }),
            ("transit3_out", { self.transit3($0) }),
            ("out_nonlinear_out", { self.outNonlinear($0) }),
            ("stats_out", { campPlusStatsPool($0) }),
            ("style_recompute", { self.dense($0) }),
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for (_, stage) in stages { x = stage(x) }
        return x
    }
}

public final class CAMPPlus: Module {
    @ModuleInfo(key: "head") var head: CampPlusFCM
    @ModuleInfo(key: "xvector") var xvector: CampPlusXVector

    public init(featDim: Int = 80, embeddingSize: Int = 192) {
        let fcm = CampPlusFCM(featDim: featDim)
        self._head.wrappedValue = fcm
        self._xvector.wrappedValue = CampPlusXVector(
            inChannels: fcm.outChannels, embeddingSize: embeddingSize)
    }

    // Ladder access for the parity gate.
    public var headModule: CampPlusFCM { head }
    public var xvectorModule: CampPlusXVector { xvector }

    /// x: (B, T, featDim) fbank−CMN → (B, embeddingSize).
    /// (torch permutes to channels-first here; this port is channels-last throughout.)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        xvector(head(x))
    }

    /// Remap funasr campplus_cn_common keys → this module's parameter tree:
    /// - drop num_batches_tracked
    /// - block{i}.tdnnd{j}.* → block{i}.layers.{j-1}.*
    /// - Conv2d (O,I,kH,kW) → (O,kH,kW,I); Conv1d (O,I,K) → (O,K,I)
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.contains("num_batches_tracked") { continue }
            var nk = k
            if let range = nk.range(of: #"\.tdnnd(\d+)\."#, options: .regularExpression) {
                let idx = Int(nk[range].dropFirst(".tdnnd".count).dropLast())!
                nk = nk.replacingCharacters(in: range, with: ".layers.\(idx - 1).")
            }
            nk = nk.replacingOccurrences(of: "shortcut.0.", with: "shortcut.conv.")
            nk = nk.replacingOccurrences(of: "shortcut.1.", with: "shortcut.bn.")
            if nk.hasSuffix(".weight") && v.ndim == 4 {
                out[nk] = v.transposed(0, 2, 3, 1)   // Conv2d (O,I,kH,kW) → (O,kH,kW,I)
                continue
            }
            if nk.hasSuffix(".weight") && v.ndim == 3 {
                out[nk] = v.transposed(0, 2, 1)      // Conv1d (O,I,K) → (O,K,I)
                continue
            }
            out[nk] = v
        }
        return out
    }
}
