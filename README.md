# mlx-indextts2-swift

Swift-MLX port of **IndexTTS-2.5** (bilibili IndexTeam) for MLXEngine: zero-shot voice cloning
from a reference clip in **zh / en / ja / es / ar**, with the two control levers no other fleet
TTS has natively — **emotion decoupled from speaker identity** (8-category preset plane) and
**explicit duration control** (the length-regulator target). Ships as an engine `tts` package
(`MLXIndexTTS2TTS.IndexTTS2Package`) over an engine-free core (`MLXIndexTTS2`).

| | |
|---|---|
| Model | [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) rev `d0aa86e` (2026-08-10) |
| Weights (fleet re-host) | [mlx-community/IndexTTS-2.5-fp16](https://huggingface.co/mlx-community/IndexTTS-2.5-fp16) — main checkpoint + w2v-BERT, one repo |
| Donor / oracle | [vanch007/mlx-indextts2](https://github.com/vanch007/mlx-indextts2) (MIT, Python-MLX, `--profile v25`) |
| Weight license | bilibili Model Use License Agreement — commercial use permitted; separate license above 100 M MAU / RMB 1 B revenue (§2.2) |
| Port license | Apache-2.0 |
| Engine | `mlx-engine-swift` ≥ 0.51.0 (`SPDXLicense.bilibiliModelUse`, allowlisted) |
| Output | `.wav` mono 22.05 kHz |

> **v0.4.0 (2026-09-02) is the 2.5 update.** The IndexTTS-2.0 tier (non-commercial
> INDEX_MODEL_LICENSE, MaskGCT/RepCodec + vq2emb + gpt_layer, SentencePiece tokenizer) was
> **removed**, not kept alongside — one package, one model. Consumers on the 2.0 API: the
> `Configuration` lost `semanticCodecRepo`/`semanticCodecDirectory`, `Generator.load` lost its
> codec directory, and the weight repo changed.

## What changed from 2.0 to 2.5 (the port delta)

| Stage | 2.0 | 2.5 |
|---|---|---|
| Text tokenizer | SentencePiece Unigram (12 k) + CJK spacing + UPPERCASE | **tiktoken byte-level BPE** (58 836 ranks + 1673 Whisper-style specials = 60 509), `<\|lang\|> ` prefix, `<word\|pron>` markup → `SPECIAL_TOKEN_1/2`, stop id 1 appended |
| Language | implicit | explicit `lang_embedding` (107 rows) added to every text position; 5 supported languages |
| Speaker conditioning | Conformer + 32-latent Perceiver over w2v-BERT features (32 tokens) | **`spk_emb_proj` Linear(192→1280) over the CAMPPlus embedding** (1 token) + emotion vector, then 2 zero rows |
| Speed embedding | 2 rows | gone |
| Codes → S2Mel content | MaskGCT RepCodec quantize of the prompt; `vq2emb(codes) + gpt_layer(GPT latent)` | **EnhancedCodec decoder** (8192×8 FVQ → Vocos/ConvNeXt ×12 → ×2 upsample) on the codes; the prompt side is the **raw w2v-BERT features** through the length regulator |
| Length target | `len(codes) · 1.72` | `len(S_infer) · 1.72 · duration_factor` (S_infer = 2 × codes) |
| Emotion conditioner, CFM/DiT, BigVGAN, CAMPPlus, w2v-BERT | unchanged | unchanged (the feat1/feat2 matrices and w2v stats are byte-identical to 2.0's) |

## Layout

```
Sources/MLXIndexTTS2            core (no MLXToolKit)
  Text/TiktokenBPE.swift        tiktoken BPE + special table (vocab read from the weight dir)
  Text/TextFrontendV25.swift    language resolve → normalize → case → markup → ja spacing → split → encode
  Text/Normalize.swift          zh/en TextNormalizer (CHAR_REP_MAP, protections; no WeText)
  Models/UnifiedVoiceV25.swift  GPT (spk_emb_proj + lang_embedding + emotion conditioner) + AR sampler
  Models/EnhancedCodec.swift    codec DECODE path (FVQ → VocosBackbone → Linear → ×2 → up conv)
  Models/S2Mel.swift, LengthRegulator.swift, CFM.swift, DiT.swift, WaveNet.swift   S2Mel (gpt_layer kept for the key contract, unused)
  Models/BigVGANV2.swift, W2VBert.swift, CampPlus.swift, Conformer.swift, Perceiver.swift
  Frontend/                     Seamless fbank, CAMPPlus fbank, ref-mel, EmotionPresets
  IndexTTS2Generator.swift      load → prepareReference → synthesize
Sources/MLXIndexTTS2TTS         engine wrapper: IndexTTS2Configuration (WeightSourcing, BudgetAware,
                                QuantConfigured, WeightPrewarming) + IndexTTS2Package (ModelPackage)
Sources/indextts2-gate          parity-gate CLI lane (see below)
Tests/                          offline: tokenizer/frontend parity, manifest C0–C13, MAT-1..5, CAN-1..3, INF
PORTING/goldens-v25             oracle goldens (fp32 CPU) + WAVs from the gate runs
```

## Using it

```swift
import MLXIndexTTS2TTS

let engine = MLXServeEngine()                       // default policy: the weights are allowlisted
await engine.useModelStore(ModelStore(root: modelsFolder))
let id = try await engine.register(IndexTTS2Package.registration, configuration: IndexTTS2Configuration())
try await engine.prepare(id)                        // materializes ~4.4 GB on first run

let request = TTSRequest(
    text: "The afternoon light settles quietly on the river.",
    voice: VoiceSelector(.referenceAudio(Audio(format: .wav, data: clip))),
    metaData: ["language": .string("en"), "emotion": .string("happy"), "emoAlpha": .double(0.6),
               "speechRate": .double(1.1), "seed": .int(42)])
let audio = (try await engine.run(request, package: id) as! TTSResponse).audio   // .wav 22.05 kHz
```

`metaData` (C5): `language` (zh | en | ja | es | ar or a common name; omit to detect from script —
Latin defaults to English, pass `es` for Spanish), `emotion` (preset name, `"happy:0.8,calm:0.2"`,
or an 8-number array), `emoAlpha` (0.6), `targetDuration` (seconds, wins over rate), `speechRate`
(1.0; = 1 / duration_factor), `seed`. Voice: `.referenceAudio` only.

Quant tiers (`IndexTTS2Configuration(quant:)`): `.fp16` as shipped, `.int8` / `.int4` quantize the
GPT backbone in memory at load (same weight sources). `BudgetAware`: a tight engine budget drops
fp16 → int8.

## Parity — how it was verified

Oracle: the donor's `v25` Python-MLX pipeline run **fp32 on CPU** (PyTorch w2v-BERT + CAMPPlus
preprocessing), captured stage by stage by `WIP/indextts25/tools/capture_v25_goldens.py` into
`PORTING/goldens-v25` (reference clip: the Studio's `indextts2-ref.wav`, 5.0 s; golden sentence
"The quick brown fox jumps over the lazy dog, and the afternoon light settles quietly on the
river."). Gate lane: `swift run indextts2-gate <mode>` (CLI, not XCTest — kernels).

| Mode | What | Result (2026-09-02, M5 Max) |
|---|---|---|
| `tok` | tiktoken ids on a 19-string byte-level corpus + raw ids on the 15-text frontend corpus; frontend segment ids (5 languages, markup, long-text split) | **PASS** — bpe 19/19, raw ids 15/15, frontend 12/15 id-exact (the 3 misses are the WeText digit fixtures, listed) |
| `ref` | Seamless fbank → w2v-BERT tap → CAMPPlus → ref-mel → prompt_condition (LR) → base_emovec → spk_emb_proj → conditioning → emovec_mat / blend | **PASS** — features 1.2e-4, w2v tap cos 1.0 / 7.8e-5, style 6.4e-6, ref-mel 3.9e-5, prompt_condition / spk_emb_proj / conditioning / emovec_mat **bitwise (max_abs 0)**, base_emovec 3.6e-7 |
| `gpt` | language-fused input embeddings; teacher-forced logits over the golden codes; step-0 logits; **greedy rollout token-exact** | **PASS** — input_emb bitwise; teacher-forced logits cos 1.0000001 / 1.9e-5; step-0 1.8e-5; **greedy 137/137 token-exact, natural stop** |
| `codec` | FVQ vq2emb; EnhancedCodec decode S_infer | **PASS** — vq2emb 4.8e-7; S_infer cos 1.0000001 / 1.0e-5 |
| `s2mel` | length regulator; seeded CFM noise cross-binding; 25-step CFM mel; BigVGAN wav | **PASS** — LR cond bitwise; seeded z bitwise; CFM mel (generated region) **bitwise**; BigVGAN wav cos 0.9999999 / 3.6e-6 |
| `e2e` | production dtype on Metal: fp16 greedy prefix vs the fp32 golden; sampled seed-42 utterance quantified (dBFS/length); happy 0.6; `targetDuration` 3.0 s; `speechRate` 1.3; a zh utterance | **PASS** — fp16 greedy exact prefix 96/137 then stops (139 codes); sampled 5.55 s at −26.3 dBFS; happy 0.6 → 5.67 s / −21.0 dBFS; targetDuration 3.0 → **3.00 s**; speechRate 1.3 → 4.26 s (expected 4.27); zh 3.42 s / −27.2 dBFS. WAVs in PORTING/ |
| `quant` | int8 / int4 GPT backbone: teacher-forced logits cosine + greedy prefix + e2e validity | **PASS** — int8 logits cos 0.999973 (5.58 s / −26.2 dBFS); int4 cos 0.991873 (5.06 s / −26.4 dBFS); both stop naturally |
| `footprint` | MLX-active resident floor + run peak per tier (one tier per process) | fp16 resident 4457 MB · int8 4017 · int4 3781; run peak (15 s utterance) 9250 / 8990 / 8743 MB ⇒ transient ≈ 4.8–5.0 GB every tier |

Offline XCTest (23): tokenizer/frontend parity (id-exact; the WeText digit fixtures are the
listed gap), manifest C0–C13, MAT-1..5, CAN-1..3, INF (CAMPPlus BatchNorm choke point).

### Documented deviations from the reference

- **Numbers in zh/en text are not expanded** (upstream: WeTextProcessing). Digits tokenize as
  digits. Pre-normalize numbers upstream for now; a Swift normalizer is an open follow-up.
- **Japanese word spacing** uses Apple's NaturalLanguage tokenizer instead of MeCab/unidic. It
  reproduces MeCab's boundaries on the fixture corpus; not guaranteed identical everywhere.
  `<|…|>` spans are protected (upstream would shred them).
- **Spanish NeMo text normalization** is not ported (upstream falls back to raw text without NeMo;
  so does this).
- No left-padding row in the GPT prefix (the official generator's masked pad row contributes
  nothing); greedy rollouts are token-exact against the padded oracle.
- The GPT-latent → S2Mel branch (`use_gpt_latent`, off by default upstream) is not carried.
- Sampled (non-greedy) rollouts are NOT a parity gate: MLX `categorical` is not stream-identical
  Python↔Swift (AB-L-0090); the seeded CFM `normal` is.

## Footprints

Measured with `indextts2-gate footprint [--bits 8|4]` (MLX-active memory, M5 Max, 2026-09-02):

| Tier | Resident floor (post-load) | Run peak, 5.5 s / 15 s utterance | Declared |
|---|---|---|---|
| fp16 | 4457 MB | 7792 / 9250 MB | resident 4.6 GB + peak 5.1 GB |
| int8 | 4017 MB | 7396 / 8990 MB | resident 4.2 GB + peak 5.1 GB |
| int4 | 3781 MB | 6849 / 8743 MB | resident 3.9 GB + peak 5.1 GB |

The transient is CFM + BigVGAN dominated and grows with utterance length; quant tiers move the
resident floor only.

Timing (release build, `indextts2-gate e2e`, M5 Max): reference prepared in 1.95 s; the 5.55 s golden
sentence sampled in 4.28 s wall (**RTF 0.77**). The debug build reads ~7× slower on the AR loop
(30 s) — quote release numbers only.

## Weight-license obligations (bilibili Model Use License Agreement)

Retain the LICENSE and notices with every copy (§3.4(b)); flow the terms downstream (§3.4(a));
carry the §4.1(a) "not endorsed by the original right-holder" statement on distributed derivative
works (the mlx-community model card does); do not use the model to improve other commercial AI
models (§3.4(c)); no high-risk deployment (§4.2). The commercial-scale gate (§2.2: 100 M MAU /
RMB 1 B revenue) is the condition of the engine's allowlist pass — revisit at crossing.
