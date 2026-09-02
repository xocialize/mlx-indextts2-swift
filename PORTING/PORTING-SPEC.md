# mlx-indextts2-swift — porting spec & gates (IndexTTS-2.5, v0.4.0)

Swift-MLX port of **IndexTTS-2.5**. Donor / oracle: `vanch007/mlx-indextts2` (MIT, Python-MLX,
`--profile v25`, converter rev `4a32c967`) over the released checkpoint `IndexTeam/IndexTTS-2.5`
rev `d0aa86e75bb6f3437f3831e95056fa72842d89ef`. The upstream PyTorch reference (`index-tts`
`infer_v2_5.py`, `gpt/model_v2_5.py`, `codec/models.py`) was read line by line for every delta;
the donor is the numeric oracle because it is the MLX-native reference the Swift port mirrors
op-for-op (the 2.0 port followed the same doctrine with solar2ain's donor).

Goldens: `PORTING/goldens-v25/` (11 MB, in-repo for durability — the 2.0 oracle workspace
outside the repo did not survive), captured by `WIP/indextts25/tools/capture_v25_goldens.py`
(fp32, CPU, deterministic; reference clip = the Studio's `indextts2-ref.wav`, 48 kHz → 5.0 s;
golden sentence "The quick brown fox jumps over the lazy dog, and the afternoon light settles
quietly on the river."; seed 42 for the CFM noise). `manifest.json` records the tuple.

## History

- **v0.1–v0.3 (2026-07):** IndexTTS-2.0 port (P1–P7 + Stage 2), non-commercial weights,
  eval-acknowledged license gate. Superseded and REMOVED at v0.4.0 (RepCodec/MaskGCT, vq2emb,
  gpt_layer use, SentencePiece tokenizer, speaker Conformer/Perceiver, speed embedding, the
  three-source weight layout). The 2.0 phase table lives in git history (tag v0.3.0).
- **v0.4.0 (2026-09-02):** IndexTTS-2.5 — this document.

## The 2.5 delta (what was ported)

| # | Surface | Reference | Swift |
|---|---|---|---|
| D1 | tiktoken byte-level BPE + Whisper special table (60 509) | `utils/tokenizer.py get_encoding` | `Text/TiktokenBPE.swift` (own `byte_pair_merge`; vocab read from the weight dir; the `=` empty-token line is accepted) |
| D2 | text pipeline: CHAR_REP_MAP → zh/en normalize → case rule → `<word\|pron>` → ja spacing → `<\|XX\|>` → token-budget split → `<\|lang\|> ` prefix + stop 1 | `infer_v2_5.py infer`, `split_text_by_tokens`, `apply_pronunciation_annotations`, `ja_g2p.py` | `Text/TextFrontendV25.swift` (+ `Normalize.swift` gains `applyBaseCharRepMap`) |
| D3 | `spk_emb_proj` (192→1280) + `lang_embedding` (107); conds = `[spk+emo, 0, 0]`; text emb += lang emb | `gpt/model_v2_5.py` | `Models/UnifiedVoiceV25.swift` (emotion conditioner + AR sampler carried from 2.0) |
| D4 | EnhancedCodec DECODE: FVQ codebook+out_project → VocosBackbone(384, ff 2048, ×12 ConvNeXt) → Linear(384→1024) → nearest ×2 → `up` Conv1d k3 | `codec/models.py`, `codec/kmeans/vocos.py`, `amphion_codec/quantize` | `Models/EnhancedCodec.swift` (encoder half dropped in `sanitize`, 0-missing/0-unused otherwise) |
| D5 | prompt_condition = length_regulator(raw w2v-BERT features); target = `len(S_infer)·1.72·duration_factor`; no gpt_layer | `infer_v2_5.py` 620–840 | `IndexTTS2Generator.swift` (S2Mel module keeps `gpt_layer` params for the key contract, never calls them) |
| D6 | engine surface: 2 weight sources in one repo, `language` metaData, allowlisted license | — | `MLXIndexTTS2TTS/*` |

Unchanged and re-gated against fresh goldens: Seamless fbank + w2v-BERT tap, CAMPPlus, ref-mel,
emotion conditioner (Conformer + 1-latent Perceiver), EmotionPresets (feat1/feat2 are
byte-identical to 2.0's baked copies — verified), length regulator, CFM/DiT/WaveNet, BigVGAN v2.

## Gate table

| Gate | Lane | Target | Result (2026-09-02, M5 Max, release build) |
|---|---|---|---|
| tok | XCTest + `indextts2-gate tok` | id-exact: 19 byte-level BPE strings; raw ids on the 15-text corpus; frontend segment ids for zh/en/ja/es/ar, pronunciation markup, 12-sentence split at budget 60 | **PASSED** — bpe 19/19, raw 15/15, frontend 12/15 id-exact (3 WeText digit fixtures listed as the gap; ja exact via NaturalLanguage spacing) |
| ref | `ref` (fp32 CPU) | Seamless features, w2v tap, CAMPPlus fbank + style, ref-mel, LR prompt, base_emovec, spk_emb_proj, conditioning, emovec_mat/blend | **PASSED** — features 1.2e-4 · w2v tap cos 1.0 / 7.8e-5 · fbank 2.5e-4 · style 6.4e-6 · ref-mel 3.9e-5 · LR prompt / spk_emb_proj / conditioning / emovec_mat BITWISE · base_emovec 3.6e-7 · blend 6e-8 |
| gpt | `gpt` (fp32 CPU) | input_emb exact (pad-row equivalence), teacher-forced logits cos ≥ 0.9999, step-0 logits, greedy rollout token-exact + natural stop | **PASSED** — input_emb BITWISE · teacher-forced cos 1.0000001 / 1.9e-5 · step-0 1.8e-5 · greedy **137/137 token-exact**, natural stop |
| codec | `codec` (fp32 CPU) | vq2emb, S_infer | **PASSED** — vq2emb 4.8e-7 · S_infer cos 1.0000001 / 1.0e-5 |
| s2mel | `s2mel` (fp32 CPU) | LR cond; seeded `normal` cross-binding; 25-step CFM mel; BigVGAN wav | **PASSED** — LR cond BITWISE · seeded z BITWISE · CFM mel (generated region) BITWISE · BigVGAN wav cos 0.9999999 / 3.6e-6 |
| e2e | `e2e` (fp16 Metal) | fp16 greedy prefix vs fp32 golden; sampled utterance validity (dBFS, length); happy 0.6; targetDuration 3.0 s exact; speechRate 1.3; zh | **PASSED** — fp16 greedy prefix 96/137, stops at 139 · sampled 5.55 s / −26.3 dBFS · happy 5.67 s / −21.0 · targetDuration 3.0 → 3.00 s · speechRate 1.3 → 4.26 s (≈4.27) · zh 3.42 s / −27.2 dBFS |
| quant | `quant` (Metal) | int8 logits cos ≥ 0.999, int4 ≥ 0.99; greedy stops; e2e valid | **PASSED** — int8 cos 0.999973 (5.58 s / −26.2 dBFS) · int4 cos 0.991873 (5.06 s / −26.4 dBFS); both stop |
| footprint | `footprint [--bits 8\|4]` | MLX-active resident + run peak per tier → manifest | resident fp16 4457 / int8 4017 / int4 3781 MB; peak (15 s) 9250 / 8990 / 8743 MB ⇒ transient ≈ 4.8–5.0 GB → declared 4.6 / 4.2 / 3.9 GB + 5.1 GB |
| offline | `swift test` | 23: tokenizer/frontend parity, manifest C0–C13 (weights allowlisted), MAT-1..5 (2 sources), CAN-1..3, INF | **23/23 green** |

## Notes banked during the 2.5 port

- **The oracle's GPT prefix has one masked left-pad row** (official `prepare_gpt_inputs` pads to
  `cond + len(text incl. stop) + 2` while the canonical text is `len + 2 − 1`). A fully masked key
  contributes exactly 0 after softmax in fp32, so the Swift prefix omits it; `gpt input_emb` is
  compared on the unpadded tail and the greedy rollout is token-exact.
- **tiktoken's vocabulary has an empty-bytes token** (`= 48474`): Python's `b64decode("=")`
  returns `b""`, Foundation's returns nil. Accept it explicitly; it can never match a piece.
- **MeCab at g2p_ratio 0 still changes the text** — it re-joins morphemes with spaces
  (punctuation included) — so "no G2P" is NOT "no-op". Apple's NaturalLanguage word tokenizer
  reproduces the fixture boundaries; keep a ja fixture in the corpus so drift is visible.
- **WeTextProcessing is LIVE in the 2.5 oracle env** (the 2.0 oracle install had it as a no-op),
  so digit-bearing zh/en fixtures now differ (`2024 → twenty twenty four`, and it even expands
  digits INSIDE pronunciation markup: `AH0 → AH ZERO`). Documented gap; the test asserts the
  gap set is non-empty and digit-only so a future port has something to flip.
- The codec's conv weights arrive in MLX layout `(out, k, in)` from the donor converter — no
  transposes; the FVQ 1×1 out-projection `(1024, 1, 8)` squeezes to a Linear.
- The 2.5 `s2mel.safetensors` still ships `gpt_layer.*` (the checkpoint is `DiT_gptlatent_10000`);
  upstream constructs `MyModel` without `use_gpt_latent`, so the layer is dead weight. Kept in
  the Swift module so the 0-unused contract holds; never called.
- **The 2.5 donor's CFM solver returns x AFTER the final prompt-region re-zero** (solar2ain's
  original — and this port — return the last Euler state before it). The prompt frames are
  trimmed before BigVGAN either way, so the `s2mel` gate judges the generated region (BITWISE);
  a whole-tensor compare reads cos 0.99988 / max_abs 0.36 purely from those discarded frames.
- **Concurrent `swift build` / `swift test` corrupt `.build`** — every build in this port was
  serialized behind the gate run it fed.
