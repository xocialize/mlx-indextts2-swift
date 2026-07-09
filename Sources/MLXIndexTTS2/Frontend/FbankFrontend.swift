// FbankFrontend.swift — P3 reference-audio fbank heads on the MLXAudioDSP leaf.
//
// Two heads share one kaldi-style log-mel core (framing 400/160 snip-edges → DC-offset
// removal → pre-emphasis 0.97 → povey window → |rfft|² @ 512 → mel → ln(max(x, floor))):
//
// - `SeamlessFeatureExtractor` — HF `SeamlessM4TFeatureExtractor` for w2v-BERT 2.0:
//   waveform ×2^15, kaldi-scale 80-mel filters (baked), per-mel-bin CMVN with **ddof=1**
//   (sample variance — the easy-to-miss detail), pad-to-stride with padding_value=1.0,
//   stride-2 stack → (T/2, 160) + attention mask (indices % 2 == 1).
// - `CampPlusFbank` — torchaudio.compliance.kaldi.fbank(num_mel_bins=80, dither=0):
//   NO input scaling (the ×2^15 shows up only as a +ln(2³⁰) offset that the CMN cancels),
//   kaldi mel banks (baked, zero nyquist column), then subtract the time-mean (CMN).
//
// Filterbanks + povey window are BAKED resources dumped from the oracle venv
// (tools/dump_frontend.py) — bake-fixed-transforms rule. Both heads gate bit-close against
// HF/torchaudio goldens in `indextts2-gate p3fe`.

import Foundation
import MLX
import MLXAudioDSP

/// Baked transforms shared by the fbank heads.
enum FrontendResources {
    static let melFloor: Float = 1.192092955078125e-07

    private static func loadNPY(_ name: String) -> MLXArray {
        guard let url = Bundle.module.url(forResource: name, withExtension: "npy",
                                          subdirectory: "Resources"),
              let array = try? NPY.load(url)
        else { fatalError("missing baked resource \(name).npy") }
        return array
    }

    /// (257, 80) — HF kaldi-scale mel filters (freq-bins × mels).
    static let seamlessMelFilters = loadNPY("seamless_mel_filters")
    /// (400,) — povey window (symmetric).
    static let poveyWindow = loadNPY("seamless_window")
    /// (257, 80) — torchaudio kaldi banks, transposed from (80, 257) at load
    /// (zero nyquist column already appended by the dump).
    static let kaldiMelBanks = loadNPY("kaldi_mel_banks").transposed()

    /// Shared kaldi-style log-mel core: 1-D waveform → (t, 80).
    static func kaldiLogMel(_ waveform: MLXArray, scale: Float, filters: MLXArray) -> MLXArray? {
        guard let frames = AudioDSP.framedSnipEdges(
            waveform * scale, frameLength: 400, hop: 160) else { return nil }
        var x = AudioDSP.removeDCOffset(frames)
        x = AudioDSP.preEmphasized(x, coefficient: 0.97)
        let power = AudioDSP.powerSpectrum(x, window: poveyWindow, fftLength: 512)  // (t, 257)
        let mel = power.matmul(filters)                                             // (t, 80)
        return MLX.log(maximum(mel, melFloor))
    }
}

/// HF SeamlessM4TFeatureExtractor head (w2v-BERT 2.0 conditioning).
public enum SeamlessFeatureExtractor {
    public static let stride = 2
    public static let paddingValue: Float = 1.0

    /// Raw per-frame log-mel (pre-CMVN), exposed for gating: (t, 80).
    public static func fbank(_ waveform: MLXArray) -> MLXArray? {
        FrontendResources.kaldiLogMel(waveform, scale: 32768,
                                      filters: FrontendResources.seamlessMelFilters)
    }

    /// waveform (T,) 16 kHz → (1, t/2, 160) input features + (1, t/2) attention mask.
    public static func callAsFeatures(_ waveform: MLXArray) -> (features: MLXArray, mask: MLXArray)? {
        guard let raw = fbank(waveform) else { return nil }
        let t = raw.dim(0)

        // Per-mel-bin CMVN over time, sample variance (ddof=1) + 1e-7 (HF exact).
        let mean = raw.mean(axis: 0, keepDims: true)
        let centered = raw - mean
        let varSample = (centered * centered).sum(axis: 0, keepDims: true) / Float(t - 1)
        var features = centered / sqrt(varSample + 1e-7)

        // Pad to a stride multiple with padding_value; mask marks real frames.
        var mask = MLXArray.ones([t]).asType(.int32)
        let remainder = t % stride
        if remainder != 0 {
            let pad = stride - remainder
            features = padded(features, widths: [IntOrPair((0, pad)), IntOrPair((0, 0))],
                              value: MLXArray(paddingValue))
            mask = padded(mask, widths: [IntOrPair((0, pad))])
        }
        let stacked = features.reshaped(features.dim(0) / stride, 80 * stride)

        // HF keeps mask positions where index % stride == 1 (the 2nd frame of each pair):
        // reshape (t, ) → (t/stride, stride) and take the last column.
        let downMask = mask.reshaped(mask.dim(0) / stride, stride)[0..., stride - 1]
        return (stacked[.newAxis, 0..., 0...], downMask[.newAxis, 0...])
    }
}

/// torchaudio kaldi fbank head (CampPlus speaker embedding).
public enum CampPlusFbank {
    /// Raw kaldi log-mel: (t, 80) (no input scaling — CMN cancels it downstream).
    public static func fbank(_ waveform: MLXArray) -> MLXArray? {
        FrontendResources.kaldiLogMel(waveform, scale: 1,
                                      filters: FrontendResources.kaldiMelBanks)
    }

    /// fbank − time-mean (the CampPlus input): (t, 80).
    public static func fbankCMN(_ waveform: MLXArray) -> MLXArray? {
        guard let raw = fbank(waveform) else { return nil }
        return raw - raw.mean(axis: 0, keepDims: true)
    }
}
