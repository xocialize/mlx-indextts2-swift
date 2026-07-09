// RefMel.swift — the 22.05 kHz reference-mel head (generate_v2 `mel_fn`) for the S2Mel CFM
// prompt, on the MLXAudioDSP leaf.
//
// Verbatim port of the torch `mel_spectrogram` closure in generate_v2._init_mel_config
// (itself matching index-tts): reflect-pad (n_fft − hop)/2 = 384 each side → non-centered
// STFT (n_fft 1024 / hop 256 / win 1024, periodic hann) → magnitude sqrt(power + 1e-9) →
// librosa mel basis (80 × 513, slaney, fmin 0 / fmax None — BAKED, dumped by
// tools/dump_stage2.py) → ln(clamp(min: 1e-5)). Output (1, 80, T).
//
// Gates bitwise-classed against `refmel_ref` / `refmel_synth` in `indextts2-gate refmel`
// (the oracle recompute is bitwise-identical to the Stage-0 `frontend_ref__ref_mel` golden).

import Foundation
import MLX
import MLXAudioDSP

/// The S2Mel reference-mel head: 22 050 Hz waveform → (1, 80, T) log-mel.
public enum RefMel {
    public static let sampleRate = 22_050
    static let nFFT = 1024
    static let hop = 256
    static let logFloor: Float = 1e-5

    /// (80, 513) — librosa mel basis, baked.
    static let melBasis: MLXArray = {
        guard let url = Bundle.module.url(forResource: "refmel_basis", withExtension: "npy",
                                          subdirectory: "Resources"),
              let array = try? NPY.load(url)
        else { fatalError("missing baked resource refmel_basis.npy") }
        return array
    }()

    static let window = AudioDSP.hannWindow(nFFT, periodic: true)

    /// waveform (T,) at 22 050 Hz → (1, 80, frames) log-mel spectrogram.
    public static func melSpectrogram(_ waveform: MLXArray) -> MLXArray {
        let pad = (nFFT - hop) / 2  // 384
        let padded = AudioDSP.reflectPadded(waveform, pad: pad)
        // torch.stft(center=False): frames = 1 + (n − nFFT)/hop — kaldi snip-edges shape.
        guard let frames = AudioDSP.framedSnipEdges(padded, frameLength: nFFT, hop: hop) else {
            fatalError("RefMel: waveform shorter than one frame (\(waveform.dim(0)) samples)")
        }
        let power = AudioDSP.powerSpectrum(frames, window: window)   // (t, 513)
        let magnitude = sqrt(power + 1e-9)                           // torch: sqrt(re²+im²+1e-9)
        let mel = AudioDSP.applyMelFilterbank(magnitude, filters: melBasis)  // (t, 80)
        return MLX.log(maximum(mel, logFloor)).transposed()[.newAxis, 0..., 0...]  // (1, 80, t)
    }
}
