# SDNQ Branch Roadmap

Working branch: `feature/svd-quantization` (last merged with `main` 2026-08-03, `8eb384d1f2`).
Branch opened 2026-01-14 (`3cea870a5a`).

## What SDNQ actually buys

**VRAM, not speed.** The port is dequantize-on-forward only: `sdnq_tensor.py`'s dispatch table routes
`aten.linear` / `mm` / `bmm` / `matmul` / `addmm` all through `dequantize_and_run`, and no file in
`invokeai/backend/quantization/sdnq/` contains `_int_mm`, `_scaled_mm`, Triton or any other quantized
kernel. Architecturally this is our GGUF path. SD.Next upstream has quantized matmul; that half was
not ported.

Confirmed by the one public SDNQ Krea-2 checkpoint (`WaveCut/Krea-2-Turbo-SDNQ-uint4`, uint4,
transformer-only, 2048²/8 steps, author's own benchmark):

| | original bf16 | SDNQ uint4 |
|---|---|---|
| hot mean | 90.49 s | 88.56 s (**1.02x**) |
| VRAM at load | 33,553 MB | **16,189 MB** |
| VRAM gen peak | 52,313 MB | **34,865 MB** |

Its `transformer/quantization_config.json` says `"use_quantized_matmul": false` — so the checkpoint
is dequant-only by construction, not merely by our port's limitation. Also `group_size: 0` and
`use_svd: false`, i.e. no per-group scales and no low-rank correction: the simple case.

Set expectations accordingly in the PR description. On a card that already runs fp8 compute, SDNQ is
a *slowdown* traded for headroom, not a free win.

## Phase status

| Phase | Topic | Status |
|---|---|---|
| A | LoRA sidecar parity with GGUF | **done** — `SDNQTensor` handled in all five `custom_modules/` files |
| B | T5 encoder SDNQ | **done** — `T5Encoder_SDNQ_Config` + `(Any, T5Encoder, SDNQQuantized)` loader |
| C | FLUX.2 SDNQ | **done** — `Main_SDNQ_Flux2_Config` + `Main_SDNQ_Diffusers_Flux2_Config` + tests |
| D | SDXL SDNQ + direct-patching fix | open |
| E | Disty0/sdnq package migration | open (deferred by design) |
| F–I | Catch-up for models added since 2026-01-14 | open — see below |

Existing SDNQ configs: `Main_SDNQ_FLUX_Config`, `Main_SDNQ_Diffusers_FLUX_Config`,
`Main_SDNQ_Flux2_Config`, `Main_SDNQ_Diffusers_Flux2_Config`, `Main_SDNQ_ZImage_Config`,
`Main_SDNQ_Diffusers_ZImage_Config`, `T5Encoder_SDNQ_Config`, `Qwen3Encoder_SDNQ_Config`,
`Qwen3Encoder_SDNQ_Folder_Config`.

## Coverage gap

Seven main-model architectures and four text-encoder types landed in `main` while this branch was in
flight. Only FLUX.2 was picked up (Phase C); the rest have no SDNQ config or loader.

| Landed | Base / encoder | Formats today | Covered |
|---|---|---|---|
| 2026-01-27 | Flux2 | ckpt, diffusers, GGUF | yes (Phase C) |
| 2026-04-09 | Anima | ckpt | no |
| 2026-04-12 | QwenImage | ckpt, diffusers, GGUF | no |
| 2026-05-08 | QwenVLEncoder | ckpt, folder | no |
| 2026-07-27 | Wan | diffusers, GGUF | no |
| 2026-07-27 | Ideogram4 | diffusers | no |
| 2026-07-27 | WanT5Encoder | folder | no |
| 2026-07-29 | Krea2 | ckpt, diffusers, GGUF | no |
| 2026-07-29 | Qwen3VLEncoder | ckpt, folder | no |
| 2026-07-30 | Gemma2Encoder | GGUF, folder | no |
| 2026-07-31 | ErnieImage | diffusers | no |

Predating the branch and never planned: CogView4 (2025-03-06), SD1/SD2, SD3.

## Cross-cutting: the `model_is_quantized` footgun

Every denoise node decides sidecar-vs-direct LoRA patching from a **hardcoded format list**. Two
shapes exist in the tree:

- `flux_denoise.py` / `flux2_denoise.py` / `z_image_denoise.py` enumerate formats and `raise
  ValueError` on an unknown one. The branch already added `ModelFormat.SDNQQuantized` there. A
  missing format fails loudly.
- `krea2_denoise.py` and `qwen_image_denoise.py` use
  `model_is_quantized = transformer_config.format in (ModelFormat.GGUFQuantized,)` — a membership
  test that **silently yields False**. An SDNQ model would then be direct-patched: LoRA weights are
  written into `SDNQTensor` weights, producing wrong results with no error.

So every phase below has a mandatory step: add `ModelFormat.SDNQQuantized` to that node's list, and
prefer converting the silent membership tests to the raising form while touching them. Grep anchor:
`force_sidecar_patching=`.

## Phase F — Krea-2 SDNQ

First, because it is the only architecture with a public SDNQ checkpoint to test against, and its
transformer is the largest VRAM consumer we ship (~12.2 GiB fp8 → ~6.8 GiB uint4).

**Files:**
- `configs/main.py` — `Main_SDNQ_Krea2_Config` + `Main_SDNQ_Diffusers_Krea2_Config`, mirroring the
  Z-Image pair (checkpoint and diffusers layouts both exist upstream).
- `configs/factory.py` — register both in the `AnyModelConfig` union.
- `load/model_loaders/krea2.py` — loader on `(Krea2, Main, SDNQQuantized)`. The diffusers-folder case
  is the cheap one; the single-file case needs the native→diffusers key conversion in
  `_convert_krea2_native_to_diffusers` to survive SDNQ's `weight` + `scale` + `zero_point` triplets.
- `app/invocations/krea2_denoise.py` — add `SDNQQuantized` to the quantized-format list (see
  cross-cutting above).

**Test target:** `WaveCut/Krea-2-Turbo-SDNQ-uint4` (diffusers layout, transformer-only, uint4,
`group_size: 0`, `use_svd: false`).

**Note:** that repo ships a **bf16** text encoder (8.27 GiB). Loading the pipeline whole would give
back more VRAM on the encoder than the transformer saves against an fp8 encoder. Krea-2's model
loader is mix-and-match, so pair the SDNQ transformer with an existing fp8/GGUF encoder.

**Acceptance:** transformer loads, denoises end-to-end, LoRA applies via sidecar, resident VRAM
roughly halves versus the fp8 checkpoint.

## Phase G — Qwen-Image and Wan SDNQ

Both are large DiTs and both already have a GGUF path to mirror.

**Files:** `Main_SDNQ_QwenImage_Config` (+ diffusers variant), `Main_SDNQ_Wan_Config`; loaders in
`load/model_loaders/qwen_image.py` and `wan.py`; format lists in `qwen_image_denoise.py` and
`wan_denoise.py`.

**Wan specifics:** it is the only base with a two-stage (high/low) transformer pair — `wan_denoise.py`
tracks `self._high_is_quantized` separately, so both stages need the format check.

**Risk:** no public SDNQ checkpoints known for either. Verify availability before starting, or the
work cannot be validated.

## Phase H — Remaining diffusers-only bases

ErnieImage, CogView4, Ideogram4, Anima. All single-format (diffusers, except Anima = checkpoint), so
each is config + loader + factory registration with no key conversion. Cheap individually; do them
as one batch once F and G have settled the pattern.

**Precondition for all four:** confirm SDNQ checkpoints exist. Without them this is speculative
surface area.

## Phase I — Encoder catch-up

`Qwen3VLEncoder` (Krea-2), `QwenVLEncoder` (Qwen-Image), `WanT5Encoder`, `Gemma2Encoder`
(ERNIE-Image). All follow the `Qwen3Encoder_SDNQ_Config` / `_SDNQ_Folder_Config` pair at
`configs/qwen3_encoder.py:443,500`.

Worth doing even where the matching main model has no SDNQ support yet: encoders are separately
selectable, and a 4B–9B encoder resident next to a 12 GiB transformer is exactly the pressure point
we keep hitting.

## Phase D — SDXL SDNQ (unchanged, still open)

**Files:**
- `configs/main.py` — `Main_SDNQ_SDXL_Config`.
- SDXL UNet loader for SDNQ.
- **Critical:** `backend/stable_diffusion/extensions/lora.py:51` still hardcodes
  `force_direct_patching=True` / `force_sidecar_patching=False`. Replace with a model-aware check
  mirroring `flux_denoise.py`.

**Risk:** SDXL has the largest LoRA ecosystem we support — regressions here are the most visible.

## Phase E — Disty0/sdnq package migration (deferred)

Replace the vendored `invokeai/backend/quantization/sdnq/` with the upstream PyPI package `sdnq`.
Slims the diff by ~1500 LOC and picks up upstream bugfixes. Defer until the rest is stable so
"make it work" and "switch dependencies" do not get conflated.

## Opportunity: fp8 matmul via the existing kernel

Not a phase — a follow-up once this lands. SDNQ already produces `FP8_E4M3` / `FP8_E5M2`
(`SDNQQuantizationType` in `sdnq/utils.py`), and `dequantize_symmetric` is `weight * scale`: exactly
the symmetric scaling `scaled_mm_linear` in `backend/quantization/fp8_scaled.py` consumes. An
`aten.linear.default` handler in SDNQ's dispatch table could route fp8_e4m3 weights to that kernel
instead of dequantizing, turning SDNQ from VRAM-only into VRAM + ~1.6x over bf16 — reusing code that
is already tested.

Two constraints: the SVD correction (`dequantized + svd_up @ svd_down`) must be applied as a
low-rank residual on the activations (same shape as `linear_lora_forward`), never folded into the
weight, or the dequantized weight gets materialized and the point is lost. And only E4M3 qualifies —
`_scaled_mm` will not take E5M2 as the weight operand on Ada.

INT8 is **not** worth pursuing: measured on an RTX 4090 (SM 8.9) at M=4608 against fp8 `_scaled_mm`,
`torch._int_mm` runs at 0.79x (6144×6144), 0.94x (6144×16384) and 0.43x (2560×2560). It beats bf16
(1.31x) but loses to the fp8 path we already have — Ada shares the same tensor cores for both, so
there is no headroom. `nvfp4` and `mxfp8` are Blackwell-only (`props.major < 10`).

## Conventions

- Each phase is its own PR-worthy commit set.
- F, G, H and I are independent of each other and of D.
- Every phase touching a main model must also update that architecture's quantized-format list — see
  the cross-cutting section. Missing it is silent, not loud.
- Before starting F–H, confirm a public SDNQ checkpoint exists for that architecture. Only Krea-2 is
  confirmed today.
