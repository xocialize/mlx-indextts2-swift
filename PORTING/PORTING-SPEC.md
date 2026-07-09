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
| P3a | fbank heads (Seamless w2v-BERT FE + CampPlus kaldi) on `mlx-audio-dsp` | vs HF `SeamlessM4TFeatureExtractor` / `torchaudio.compliance.kaldi.fbank` goldens (ref + synth) | **PASSED 2026-07-08** (`p3fe`: fbank cos 1.0000000, input_features 0.9999999 max_abs 1.2e-4, mask 249 exact) |
| P3b | conditioner models: w2v-BERT Conformer + MaskGCT + CampPlus + conformer/perceivers | per-embed goldens (`spk_cond_emb`, `S_ref`, `style`, `emovec`) | **PASSED 2026-07-08** (`p3w2v` hs-ladder ≤1.93e-04, tap 1.85e-05, FE-chain cos 1.0; `p3mgc` codes 250/250 exact, S_ref 9.5e-07; `p3cpp` ladder ≤4.6e-05, style 5.0e-06; `p3cond` speech_cond/base_emovec/conditioning cos ≥0.999999) |
| P4 | S2Mel CFM + length regulator | `cfm_mel` golden (seed-42 replay set) | **PASSED 2026-07-08** (`p4`: gptlayer cos 1.0000002, lenreg max_abs 0.0 BITWISE, dit_step1 cos 1.0000005, cfm_mel cos 0.9999999 over 25 steps; 264-key contract 0-missing/0-unused; seeded-normal cross-binding max_abs 4.8e-7) |
| P5 | BigVGAN2 vocoder | `bigvgan_wav` golden + listen | **PASSED 2026-07-08** (`p5`: wav(seed42) cos 0.9995050 / wav(orig) cos 0.9995069 — equals the Python reference's own CPU-vs-Metal floor (0.9995355); 449-key contract; all anti-alias primitive probes ≤7.4e-3) |
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

## P3b notes (banked)

- **All four conditioner surfaces passed first-run** (the verified MLX-Python donors +
  per-stage ladders did their job). Gates: `p3w2v`, `p3mgc`, `p3cpp`, `p3cond` — each does a
  full key contract (0-missing/0-unused), a golden-injected ladder, and (where the Swift
  front-end exists) a full audio→embedding chain.
- **Files:** `Models/W2VBert.swift` (donor `w2vbert_mlx/w2vbert.py`), `Models/RepCodec.swift`
  (donor `maskgct_mlx/repcodec.py`), `Models/CampPlus.swift` (direct translation of
  solar2ain's vendored 3D-Speaker torch reference — dots-tts donor not on disk),
  `Models/Conformer.swift` + `Models/Perceiver.swift` (donors solar2ain
  `models/{conformer,perceiver}.py`), conditioning methods on `UnifiedVoiceV2`.
- **Numeric-module-key pitfall (twice):** `ModuleParameters.unflattened` treats numeric path
  segments as ARRAY indices — torch Sequential/ModuleList children exposed as `shortcut.0`,
  `layers.N.0` cannot be Swift module keys "0"/"1". Remap in sanitize
  (`shortcut.{0,1}`→`{conv,bn}`, perceiver `layers.N.{0,1}`→`layers.N.{attn,ff}`).
- **F-order golden pitfall recurred:** several Stage-0 pipeline goldens (`frontend_ref__S_ref`,
  `core_vq2emb`, 4 maskgct ladder files) were fortran-order; rewritten C-contiguous in place
  (NPY.swift rejects F-order by design). Check `.flags['C_CONTIGUOUS']` when dumping.
- **CampPlus:** weights converted `campplus_cn_common.bin` → `_indextts2-oracle/
  campplus_cn_common.safetensors` (raw keys, minus num_batches_tracked) + 12-stage torch
  ladder via `tools/dump_campplus.py` (recompute == pipeline golden exactly). BatchNorms run
  in inference mode — `model.train(false)` before any forward. avg_pool1d(ceil_mode) in the
  CAM seg-pooling divides the partial tail window by its TRUE length. FCM flatten is C-major:
  NHWC `(B,F',T,C)` → transpose → `(B,T,C·F')`.
- **Conditioner resolved configs** (config.yaml, not dataclass defaults): cond =
  Conformer(1024→512, ff 2048, 8 heads, 6 blocks) + Perceiver(1280, ctx 512, 32 latents,
  8 heads, mult 2); emo = Conformer(1024→512, ff 1024, 4 heads, 4 blocks) + Perceiver(1024,
  ctx 512, 1 latent, 4 heads, mult 2). Perceiver FF inner = ⌊dim·mult·2/3⌋ (1706/1365 —
  confirmed by w_1 shapes). No macaron in these conformers; rel_shift unused;
  RelPositionalEncoding multiplies x by √dim and does NOT add pe.
- **Emotion blend** (generate_v2): weights = parse("happy")·α → weight_sum 0.6 →
  `emo_vec = emovec_mat + (1−0.6)·base_emovec`; emovec_mat comes from feat2.pt emo_matrix
  (still oracle-side; port with the E12 param plane at Stage 2).
- **P2 gate refactored** onto a shared `loadUnifiedVoiceV2()` full-model loader (the
  declared-subset contract is retired — all 667 gpt.safetensors keys are now declared).

## P4 notes (banked)

- **The original `core_s2mel_cfm_mel` golden is NOT reproducible from seed(42)** — generate_v2
  seeds once at start and the AR sampler's `mx.random.categorical` draws consume the global
  stream before CFM's noise draw. P4/P6 gate against the **seed-42 replay goldens**
  (`_indextts2-oracle/tools/dump_s2mel_replay.py`): seed(42) → `normal(1,80,621)` is the FIRST
  draw, so both bindings reproduce it. Replay sanity: gpt_layer / length_regulator / bigvgan
  recomputes are **bitwise identical** to the original goldens (Metal is run-to-run
  deterministic here); seed-42 cfm_mel is statistically equivalent to the original (cos 0.996).
- **Cross-binding RNG:** Swift `MLXRandom.seed(42)` → `normal` matches Python within
  max_abs 4.8e-7 — same stream, tiny fp difference in the normal transform. Fine for
  production; the parity gate injects the dumped z (`core_s2mel_cfm_z_seed42.npy`) to stay exact.
- **Files:** `Models/S2Mel.swift` (+GPTLayer), `Models/CFM.swift`, `Models/DiT.swift`,
  `Models/WaveNet.swift`, `Models/LengthRegulator.swift` — isomorphic to donor
  `models/s2mel/{s2mel,cfm,dit,wavenet,length_regulator}.py`. VoxCPM's UnifiedCFM was donor
  for loop idioms only (its math is the opposite 1→0/subtractive convention; key paths don't
  match s2mel.safetensors → translate-not-lift).
- **Sanitize remaps (numeric-Sequential pitfall again):** `length_regulator.model.{0,3,6,9}`→
  `convs.N`, `.{1,4,7,10}`→`norms.N`, `.12`→`out_proj`; `adaLN_modulation.layers.1`→
  `adaLN_modulation.linear`. Everything else is already donor-MLX layout (no conv transposes;
  vanch007 pre-fused all weight norms).
- **Checkpoint-buffer trap:** `t_embedder.freqs` is IN s2mel.safetensors (fp16) and overwrote
  the donor's computed fp32 buffer at load — declared `@ParameterInfo` so Swift loads the same
  values. The RoPE table is NOT in the checkpoint — plain non-Module class (donor ditto), keeps
  it out of the key contract.
- **Donor-over-torch quirks replicated:** SConv1d does symmetric REFLECT padding (torch WN
  zero-pads; donor produced the goldens → donor wins); FinalLayer non-affine LayerNorm eps=1e-6;
  paired/interleaved RoPE (reshape (...,D/2,2), not half-split); hand-rolled attention (no fast
  SDPA); solve_euler returns the last step BEFORE prompt-region re-zeroing.
- **Duration control (E12):** the length-regulator target length is the lever —
  generate_v2 uses `int(code_len * 1.72)`; `InterpolateRegulator(x, ylens:)` keeps it an
  explicit caller-chosen parameter.
- Resolved config = donor constructor defaults, cross-checked vs checkpoint config.yaml
  (only DiT block_size differs, 16384 vs 8192 — positions ≤621, inert; donor kept).

## P5 notes (banked)

- **Files:** `Models/BigVGANV2.swift` (AMPBlock1/2 + BigVGANV2) + `Models/Activations.swift`
  (kaiser-sinc filter, Snake/SnakeBeta, UpSample1d/DownSample1d/Activation1d) — isomorphic to
  donor `models/{bigvgan_v2,activations}.py`. bigvgan.safetensors loads with NO sanitize
  (449 keys 1:1; vanch007 pre-fused weight norms; conv weights already MLX layout).
- **NEW MLX-Swift PITFALL — shared-init parameter aliasing:** assigning ONE MLXArray instance
  to two `@ParameterInfo` wrappers (SnakeBeta alpha/beta both = `initial`) makes update()
  write both keys into the same array — last write wins, alpha silently gets beta's values.
  Key contract AND `verify: .all` both pass. Symptom: gate cos ~0.05; ladder localized it in
  one hop (probe_act 1.5e0 while manual-formula check was 1.6e-3). Always allocate distinct
  init arrays per parameter.
- **Anti-alias filters are computed, not checkpoint keys** — UpSample1d/DownSample1d are plain
  non-Module classes (donor underscore-prefixes `_filter`); np.kaiser/i0 replicated in Double
  then cast fp32. Depthwise (identical-filter, groups=C) convs = channel-fold into batch.
- **Elementwise stage gating is miscalibrated for deep snake stacks:** fp32-CPU vs fp16-Metal
  drift amplifies through 6 stages of oscillatory sin²(αx) (stage_3 max_abs ~2.0 with cos
  0.994 — benign). The Python reference ITSELF diverges from its Metal golden by max_abs 0.026
  / cos 0.9995 e2e on the CPU stream. Ladder = report-only localization; hard gate = final
  waveform cosine (Swift landed at the reference's exact floor: 0.99950 vs 0.99954).
- v2 config quirks: conv_post bias=False, final activation = clip(-1,1) not tanh,
  snakebeta-with-logscale everywhere. Checkpoint = nvidia/bigvgan_v2_22khz_80band_256x
  (rates 4,4,2,2,2,2 · kernels 8,8,4,4,4,4 · 1536→24ch · resblock kernels 3,7,11 × d 1,3,5).

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
