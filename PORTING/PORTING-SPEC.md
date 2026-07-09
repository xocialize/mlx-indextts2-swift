# mlx-indextts2-swift — porting spec & phase gates

Swift-MLX port of IndexTTS2. Donor: `solar2ain/mlx-indextts` (MIT, Python-MLX) + our verified
MLX-Python front-end ports (w2v-BERT 2.0, MaskGCT RepCodec) in `~/Development/_indextts2-oracle/`.
Plan of record: `mlxengine-audio/Docs/IndexTTS2-Swift-Port-Plan.md`. Every phase gates against the
Stage-0 goldens (`_indextts2-oracle/goldens/`, 23 files + manifest; seed=42 tuple).

## Phase table

| Phase | Surface | Gate | Status |
|---|---|---|---|
| P1 | tokenizer + normalize (`Text/`) | bit-exact ids/pieces/normalized vs oracle over 13-fixture corpus incl. golden sentence (17 ids) | **PASSED 2026-07-08** (6/6 tests) |
| P2 | GPT AR (text→semantic) | teacher-forced `gpt_latent` vs golden, cos ≥0.999 fp32 CPU | **PASSED 2026-07-08** (cos 0.9999995, max_abs 0.031; 301-key subset contract 0-missing/0-unused) |
| P3 | conditioners: w2v-BERT + MaskGCT + CampPlus + perceivers (needs `mlx-audio-dsp` leaf) | per-embed goldens (`spk_cond_emb`, `S_ref`, `style`, `emovec`) | — |
| P4 | S2Mel CFM + length regulator | `cfm_mel` golden | — |
| P5 | BigVGAN2 vocoder | `bigvgan_wav` golden + listen | — |
| P6 | e2e | Stage-0 WAV, quantified (dBFS/RMS, not ears) | — |
| P7 | GPU smoke + int8 quant | GPU-stream forward (never CPU-pin quant!) | — |

## P1 notes (banked)

- **SP model is Unigram** (not BPE): 12k pieces, `nmt_nfkc`, `add_dummy_prefix`,
  `remove_extra_whitespaces`, no byte_fallback, unk_id=2. Swift impl = protobuf-free: the
  conversion dumps `(piece, score, type)` JSON (`_indextts2-oracle/tools/dump_tokenizer.py`) and
  `SentencePieceUnigram.swift` runs trie+Viterbi (unk = minScore−10, SP `kUnkPenalty`;
  adjacent unks merge into one piece — verified: "42" → single `<unk>`).
- **wetext (number/date normalization) is a NO-OP in the oracle install** — digits pass through
  and tokenize to `<unk>` ("3", "42" → id 2). Parity target = oracle-as-run, so the Swift port has
  no wetext equivalent. **Upstream quality gap:** for dub text, pre-normalize numbers upstream
  (or add post-parity behind a flag). Do NOT "fix" this inside the parity path.
- **CHAR_REP_MAP carries an upstream source bug we reproduce verbatim:** the smart-quote dict
  entries collapsed into one garbage multi-char key, so `“ ”` are NOT mapped. The effective
  33-entry ordered map was dumped from the running Python; replacement = single left-to-right
  pass, first-matching key in map order.
- Encode pipeline: `TextNormalizer.normalize` → `tokenize_by_cjk_char` (CJK spacing + **UPPERCASE**,
  vocab is uppercase) → SP encode. NFKC roundtrip quirk: normalizer maps `...`→`…`, then SP's
  nmt_nfkc maps `…`→`...` (piece `'...'` exists).
- ICU regex gotcha: `\u{4e00}` (Swift style) is invalid in NSRegularExpression — use `一`.
- Fixtures/vocab live in `Tests/MLXIndexTTS2Tests/Resources/`; regenerate with the oracle venv
  tool if the corpus grows.

## P2 notes (banked)

- Port set = `models/gpt2.py` → `Models/GPT2.swift` + `models/gpt_v2.py` (P2 subset) →
  `Models/UnifiedVoiceV2.swift` (embeddings, LearnedPositionEmbedding as `<name>.emb.weight`,
  24×GPT2Block, final_norm/heads/speed_emb, `forwardLatent`). Conditioning golden INJECTED
  (perceiver/conformer conditioners = P3; their weight families deferred via the
  declared-subset contract in the gate).
- Resolved config truths (config.yaml, not the dataclass defaults!): dim=1280, heads=20,
  layers=24, max_mel=1815, max_text=600, cond_num=32 (+2 speed) — dataclass defaults say
  1024/16/20/605/402 and are WRONG for this checkpoint (classic resolved-config pitfall).
- Gate lane = `swift run indextts2-gate p2` (CLI, not XCTest, per metallib doctrine);
  fp32 upcast materialized with `eval(model)` post-update (watchdog corollary).
- Weight keys map 1:1 with `@ModuleInfo(key:)` (`c_attn`/`c_proj`/`ln_1`/`ln_2`/`h.N`/`ln_f`,
  `final_norm`, `<pos>.emb.weight`) — no sanitize/remap needed (donor is already MLX layout).

## Dependencies by phase

- P1: none (pure Swift). P2+: mlx-swift (+ mlx-swift-lm attention helpers).
- P3: the `mlx-audio-dsp` shared leaf — **BUILT 2026-07-08** (`mlxengine-audio/WIP/mlx-audio-dsp`,
  module `MLXAudioDSP`): hann (periodic/symmetric) + povey windows, reflect-pad, strided framing
  (center-STFT + kaldi snip-edges), DC-offset removal, per-frame pre-emphasis (kaldi x[−1]≡x[0]),
  power/magnitude spectra (with kaldi pad-to-512), mel-filterbank apply. Filterbank GENERATION is
  deliberately out (bake-fixed-transforms rule — heads ship baked filters dumped from the oracle).
  whisper-mlx-swift REFACTORED onto the leaf, **bit-identical** (legacy-inline vs refactored
  max_abs = 0; leaf hann vs baked window ≤1e-6). Whisper consumes it via local path — publish
  `mlx-audio-dsp` + restore a URL dep before tagging whisper-mlx-swift.
  Remaining P3 front-end work: the per-model heads (w2v-BERT Seamless-style normalized 80-mel with
  `wav2vec2bert_stats` mean/std; CampPlus kaldi 80-fbank) + baked-filter dumps + HF-golden gates.
- Stage 2: MLXToolKit (contract), NonCommercial weight gate (C7), Apache code (C8).
