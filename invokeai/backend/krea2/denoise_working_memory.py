"""The working memory of one Krea-2 denoise, as the denoise node reserves it for the transformer's lock.

The transformer attends over one joint sequence: the caption's tokens, the image's (one per 16x16 pixels) and those of
every joint edit reference, which `reference_edit.forward_with_references` appends to the image stream. On a fused SDPA
kernel its activations scale linearly with that sequence: `k_denoise_<dtype>` bytes per attended token plus a base of
`b_denoise_<dtype>` bytes, shipped per platform in `backend/working_memory/data/denoise.json`. Conditional and
unconditional passes run one after the other, so only the longer caption counts and CFG adds nothing; a style
reference's pass runs alone too, over the image and its own caption. In reference K/V-cache mode the references leave
the per-step sequence: they run once, alone, before the loop (`reference_edit.precompute_reference_kv`). The line prices
the longest of these passes. The platform keys are those of the VAE estimators; for a transformer, `conv_backend` just
names the CUDA (`cudnn`) or ROCm (`miopen`) stack it runs on.

That line is `denoise_<dtype>` over the attended tokens (`_line`), and only the plain forward fits it, joint references
included: every analytic term below comes to zero, and no attention mask is passed. `_line` reads that off the terms
themselves, so a term added here takes the forwards it prices out of the fit. The spec does not opt in to user
calibration (`EstimatorSpec.user_calibrated`): the lines are the shipped fit's alone.

The lines are fitted per platform and dtype on fresh-process points of `scripts/calibrate_krea2_working_memory.py`
(their provenance is in the data): an RTX 4090 with the native allocator and with `cudaMallocAsync`, bf16, fp16 and
fp32 from 512 to 2048 px with a 512-token caption, at 1024 and 1536 px with a 77-token one, at 1024 px with one to three
1024 px joint references, and in bf16 joint sequences up to 29184 tokens, on a random-init transformer of the default
geometry with 2 blocks; 4 blocks measured the same as 2. bf16 and fp16 take the same per token; fp32's slope is 1.67
times theirs under the native allocator and 1.89 times under `cudaMallocAsync`. Under the native allocator joint
references lie on the line of the plain forwards; under `cudaMallocAsync` a warm forward with one reference read 18%
above its cold one and holds the line up. CUDA on Linux reuses these lines, as the Z-Image and FLUX.2 entries do.

A caption token reserves more than an image token, while allocated memory grows alike for both: 0.73 to 0.95 against
0.30 MiB per token in bf16 and fp16 under the native allocator, 1.2 to 2.2 against 0.50 in fp32 (the 77- against the
512-token caption at 4096 and 9216 image tokens). One slope prices both up to the 512 tokens the fit grid runs, so a
short caption reserves up to 1.35 times what it demands (`bounds.UPPER_BOUND_EXCEPTIONS`). Beyond 512, which regional
prompting reaches by concatenating its captions, each caption token adds `_CAPTION_EXCESS_BYTES_PER_TOKEN` on top of
the line. The caption checks measure it (`--mode captions`: 1024- and 2048-token captions at 4096 and 9216 image tokens,
every dtype, both allocators, out of the fit): the line alone falls short at 2048 tokens, by up to 16% (0.84 to 0.95
times the demand). The term is the most a check demanded per caption token beyond 512 above the line's slope, over
both allocators, times the 1.3 that bounds a fitted term (`bounds.MAX_OVER_RESERVATION`), rounded up to the next
0.01 MiB: 0.26 MiB in bf16 and fp16 (1024 px, 2048 tokens) gives 0.34 MiB, 0.39 MiB in fp32 (1536 px, 2048 tokens) gives
0.52 MiB. The estimate covers every check, 1.05 to 1.54 times in bf16 and fp16 and 1.03 to 1.61 times in fp32 under
the native allocator, 1.21 to 1.50 and 1.11 to 1.27 times under `cudaMallocAsync`. Each group's largest ratio falls on
a forward the line alone already covers 1.10 to 1.47 times.

Cross-checks in a running server on the RTX 4090 under the native allocator with an ~84-token prompt, not shipped as
points: the estimate reserved 1.22 to 1.66 times what the real 28-block model grew the reserved peak by over the
transformer lock, GGUF Q4_K_M fully resident (1024 to 2048 px, with and without LoRAs, CFG, joint references and the
reference K/V cache) and fp8-stored (partly resident in that run, 1.27 to 1.40 times). The top end falls at 1536 px, at
1536x2048 and at 1152x1536 with one joint reference, where the warm server's allocator reserved below the line. Loaded
partially (the cache told the card has 11.76 GiB, GGUF Q4_K_M at 1152x1536 with two LoRAs, 6.3 to 6.9 of its weights'
GiB resident), the model grew the reserved peak by 2.37 GiB against an estimate of 3.30 GiB (1.39 times), with no
out-of-memory; the device peaked at 10.85 to 10.86 of the 11.76 GiB, 0.3 GiB of simulated foreign use included.

ROCm is not measured, and neither is any other platform but these two. Their entries price with the formula the node
reserved by hand before the lines were measured (`_legacy_formula`, declared as `EstimatorSpec.legacy_formula`): 0.5 MiB
per attended token and a 1.5 GiB base for every dtype, times 1.1 with CFG and 1.35 with a style reference, plus the
whole model's largest dequantization transient and half a GiB per LoRA, with the terms it lacked on top. On the RTX 4090
those constants reserved 2.3 to 2.9 times what the real model demanded. ROCm's entries are `legacy`; every other platform
nothing measures is priced as a measured one (`priced_as`), and the two entries those borrow, CUDA's and ROCm's with the
native allocator, floor their values at the legacy constants (`unmeasured_floor`), which prices that formula there too.
Either way the reservation is the larger of the formula and the terms below.

A forward has two phases that never peak together. The blocks run over the whole sequence, and every term of the
block phase is alive at once: the line, the score matrix where SDPA materializes it, style reference's extra
activations, and the largest block layer's streamed copy, dequantization or fp8 cast transient and LoRA residual. The
layers outside the blocks run on far fewer live activations: the timestep modulation `time_mod_proj` (6144 x 36864, the
model's largest linear) runs before the blocks, also for the references' modulation, and never coincides with the block
activations. Its transient decides a quantized build's whole-model peak (1.23 GiB for Q4_K), yet GGUF Q4_K_M reserved a
near constant 0.35 GiB more than the fp8 build at every measured size, about what dequantizing the largest block linears
(the MLP's, 6144 x 16384) costs beyond the fp8 build's cast of them. So the block phase prices the block layers'
transients, and `outer_layer_peak` adds only what the largest single layer anywhere needs beyond the whole block phase.
It is the one term that reads fitted values; `bounds.line_bounds` prices a point with the line deciding, where it is 0.

Resident through both phases, and priced on top of either: the caption excess, regional prompting's masks, the reference
K/V cache, the style reference's captured K/V, and the LoRA patches.

The analytic terms:

- The largest block layer's stored weights, for a transformer the model cache may load partially: a RAM-held layer is
  copied to the device for its forward, alive during its matmul. Whether the cache loads the transformer whole is
  decided by the lock this estimate sizes, so the node charges it on every forward (as the FLUX.2 denoise does); a
  fully resident model over-reserves one layer. The calibration script, whose models are resident, declares them so.
- The dequantization copy an int8_tensorwise, nvfp4 or GGUF build materializes per quantized block linear
  (`peak_dequant_transient_bytes`), and the compute-dtype copies of a block layer's float8-stored weights
  (`peak_fp8_cast_transient_bytes`); zero for every other build.
- `outer_layer_peak`: the largest single layer's streamed copy plus its transients, anywhere in the model, less the
  whole block phase where it exceeds it.
- Where SDPA has no fused kernel for these shapes it materializes the score matrices (`sdpa_score_matrix_bytes`), for
  the query heads of the transformer's config: the K/V heads are expanded before the call unless a fused kernel takes
  them grouped (`krea2/attention.py`). In K/V-cache mode each step's queries see the cached keys too, and the precompute
  is a square of the references' tokens; the larger area is priced as the smallest square that holds it. The probe
  answers for the process-wide policy, so a kernel forced through the `INVOKE_KREA2_SDPA_BACKEND` debugging switch is
  not priced. Known gap: a styled block's queries also see the reference's keys, a larger score matrix than priced.
- Regional prompting's masks, their construction scratch and SDPA's additive bias, as the node sizes them
  (`Krea2DenoiseInvocation._regional_attention_mask_bytes`). Not measured.
- The reference K/V cache, an exact size (`reference_edit.reference_kv_cache_bytes`): one key and one value per block at
  the K/V heads, plus one block's references expanded to the query heads. On the RTX 4090 the real model's K/V-cache
  denoises (1152x1536 with one and three references) followed that structure: the larger of the step's and the
  references' tokens on the line, plus the cache.
- Style reference's captured K/V, an exact size, and the extra activations of the styled attention and the reference
  pass: 0.35 times 0.5 MiB per step token plus 0.35 times 1.5 GiB, because at 1024 px (8 steps, cfg 1) the styled run's
  peak rose ~1.6 GiB over the unstyled one, 0.5 GiB of it the cache. Not measured against the line.
- The LoRA patches, at their files' size: a 0.03 GiB and a 0.21 GiB bf16 file of plain LoRA layers (`LoRALayer`) on the
  sidecar-patched GGUF Q4_K_M build added 0.02 and 0.22 GiB on the RTX 4090, at 1152x1536 and at 2048 px alike.
- The sidecar residual of the other patch types (LoKr, LoHa, DoRA, full layers and the rest that are no `LoRALayer`),
  wherever a layer is patched as a sidecar: a quantized or fp8-stored layer, or one partial loading leaves in RAM.
  `autocast_linear_forward_sidecar_patches` hands them to `_aggregate_patch_parameters`, which per forward copies the
  base weight to the device and dequantizes it again, and builds the full residual weight, cast to the input's dtype,
  beside the base output. For the largest layer such a patch targets, the layer's streamed copy (unless resident), its
  dequantization or fp8 cast transient, and its weight's elements at float32 plus the compute dtype. Derived from the
  code path, not measured; once, since the layers run one after another. A packed nvfp4 weight counts half its logical
  elements, so the residual reads low there; its dequantization transient covers that in practice.

Known and not priced: on a dense bf16 or fp16 build a LoRA on a resident layer is patched directly, once, before the
loop, which takes about 8 bytes per element of the largest patched layer (~1.7 GiB for `time_mod_proj`). It never
coincides with the block activations, and the `device_working_mem_gb` floor of every reservation covers it.

Not measured: Sage attention, flash attention (Linux), ROCm, int8 and bf16 builds of the real model, regional prompting,
CFG with captions of unequal length, and the patch types that are no `LoRALayer`.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Final, Optional, cast

import torch

from invokeai.backend.krea2.reference_edit import reference_kv_cache_bytes
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.calc_tensor_size import calc_tensor_size
from invokeai.backend.util.fp8 import peak_fp8_cast_transient_bytes
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import Line, off_line_reasons
from invokeai.backend.working_memory.platform import platform_for
from invokeai.backend.working_memory.spec import EstimatorSpec, register

_MIB: Final = 2**20
_GIB: Final = 2**30
FITTED_CAPTION_TOKENS: Final = 512
"""The caption the fit grid runs; the line covers captions up to it."""
_CAPTION_EXCESS_BYTES_PER_TOKEN: Final[Mapping[str, int]] = {
    "bf16": int(0.34 * _MIB),
    "fp16": int(0.34 * _MIB),
    "fp32": int(0.52 * _MIB),
}
"""Per caption token beyond `FITTED_CAPTION_TOKENS`: the caption checks' largest demand above the line's slope, times
`bounds.MAX_OVER_RESERVATION`, rounded up to the next 0.01 MiB (see the module docstring)."""
_STYLE_ACTIVATION_SHARE: Final = 0.35
_STYLE_ACTIVATION_BYTES_PER_TOKEN: Final = _MIB // 2
_STYLE_ACTIVATION_BASE_BYTES: Final = 3 * _GIB // 2
_RESIDUAL_BYTES_PER_ELEMENT: Final = torch.float32.itemsize
"""A sidecar residual weight is built in at most float32, then cast to the input's dtype."""
_DTYPE_NAMES: Final[Mapping[torch.dtype, str]] = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
"""The dtypes `TorchDevice.choose_bfloat16_safe_dtype` runs the transformer in."""
_COEFFICIENTS: Final = frozenset(f"{part}_denoise_{dtype}" for part in ("k", "b") for dtype in _DTYPE_NAMES.values())


def _line(features: Mapping[str, FeatureValue]) -> Optional[Line]:
    """The forward's line, `denoise_<dtype>` over the attended tokens. Only the plain forward fits it, joint references
    included: no attention mask, and every analytic term of `_terms` zero (see the module docstring). The label names
    what puts a forward off the line: the mask, and each analytic term that is not zero, in bytes. A dtype without a
    line names none."""
    dtype, tokens = features.get("dtype"), features.get("tokens")
    if not isinstance(dtype, str) or f"k_denoise_{dtype}" not in _COEFFICIENTS or not isinstance(tokens, int):
        return None
    off_line = off_line_reasons(_terms, features, _COEFFICIENTS)
    references, cached = features.get("reference_tokens"), features.get("cached_reference_tokens")
    shape = "+".join(
        (
            str(features.get("image_tokens")),
            *([f"{references}ref"] if references else []),
            *([f"{cached}kvref"] if cached else []),
            f"{features.get('text_tokens')}tok",
        )
    )
    return Line(f"denoise_{dtype}", tokens, not off_line, "-".join(("denoise", dtype, shape, *off_line)))


def _step_tokens(features: Mapping[str, FeatureValue]) -> int:
    """The tokens of one target pass: the image, the joint references and the longer caption of a CFG pair."""
    return sum(cast(int, features[name]) for name in ("image_tokens", "reference_tokens", "text_tokens"))


def _terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    dtype = cast(str, features["dtype"])
    style_kv_bytes = cast(int, features["style_kv_bytes"])
    style_activation_bytes = 0
    if style_kv_bytes:
        # Whole bytes, so that a legacy-priced reservation sums to its formula exactly.
        style_activation_bytes = int(
            _STYLE_ACTIVATION_SHARE
            * (_STYLE_ACTIVATION_BYTES_PER_TOKEN * _step_tokens(features) + _STYLE_ACTIVATION_BASE_BYTES)
        )
    caption = max(cast(int, features["text_tokens"]), cast(int, features["style_text_tokens"]))
    caption_excess = max(0, caption - FITTED_CAPTION_TOKENS) * _CAPTION_EXCESS_BYTES_PER_TOKEN[dtype]
    blocks = (
        WorkingMemoryTerm.fitted("activations", f"k_denoise_{dtype}", cast(int, features["tokens"]), values),
        WorkingMemoryTerm.fitted("base", f"b_denoise_{dtype}", 1, values),
        WorkingMemoryTerm("attention_scores", cast(int, features["score_matrix_bytes"]), True),
        WorkingMemoryTerm("style_reference_activations", style_activation_bytes, True),
        WorkingMemoryTerm("streamed_layer", cast(int, features["streamed_layer_bytes"]), True),
        WorkingMemoryTerm("dequant_transient", cast(int, features["dequant_transient_bytes"]), True),
        WorkingMemoryTerm("fp8_cast_transient", cast(int, features["fp8_cast_transient_bytes"]), True),
        WorkingMemoryTerm("sidecar_patch_residual", cast(int, features["sidecar_residual_bytes"]), True),
    )
    block_phase = sum(term.bytes for term in blocks)
    return (
        *blocks,
        WorkingMemoryTerm("outer_layer_peak", max(0.0, cast(int, features["peak_layer_bytes"]) - block_phase), True),
        WorkingMemoryTerm("caption_excess", caption_excess, True),
        WorkingMemoryTerm("regional_attention_masks", cast(int, features["regional_mask_bytes"]), True),
        WorkingMemoryTerm("reference_kv_cache", cast(int, features["reference_kv_bytes"]), True),
        WorkingMemoryTerm("style_reference_kv_cache", style_kv_bytes, True),
        WorkingMemoryTerm("lora_patches", cast(int, features["lora_bytes"]), True),
    )


def _legacy_formula(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> float:
    """What the node reserved by hand before the lines were measured, with the legacy-priced values in place of its
    0.5 MiB per token and 1.5 GiB base, plus the terms it lacked: the streamed layer, the fp8 cast and the sidecar
    residual. It priced neither the style reference's caption nor captions beyond the fit grid's apart."""
    dtype = cast(str, features["dtype"])
    tokens = max(_step_tokens(features), cast(int, features["cached_reference_tokens"]))
    estimated = tokens * values[f"k_denoise_{dtype}"] + values[f"b_denoise_{dtype}"]
    if features["cfg"]:
        estimated = int(estimated * 1.1)
    estimated += cast(int, features["regional_mask_bytes"])
    if features["style_kv_bytes"]:
        estimated = int(estimated * 1.35) + cast(int, features["style_kv_bytes"])
    estimated += cast(int, features["reference_kv_bytes"])
    estimated += int(0.5 * cast(int, features["lora_count"]) * _GIB)
    estimated += cast(int, features["score_matrix_bytes"]) + cast(int, features["model_dequant_transient_bytes"])
    lacked = ("streamed_layer_bytes", "fp8_cast_transient_bytes", "sidecar_residual_bytes")
    return estimated + sum(cast(int, features[name]) for name in lacked)


KREA2_DENOISE_SPEC = register(
    EstimatorSpec(
        id="denoise.krea2",
        version=1,
        coefficients=_COEFFICIENTS,
        platform_keys=("conv_backend", "allocator"),
        terms=_terms,
        line=_line,
        legacy_formula=_legacy_formula,
    )
)


def _stored_bytes(module: torch.nn.Module) -> int:
    """A module's own parameters and buffers as stored: what partial loading copies to the device for its forward."""
    return sum(calc_tensor_size(t) for t in (*module.parameters(recurse=False), *module.buffers(recurse=False)))


def _transient_bytes(module: torch.nn.Module, dtype: torch.dtype) -> int:
    """The dequantization and fp8 cast transients of one leaf layer's forward."""
    return peak_dequant_transient_bytes(module, dtype) + peak_fp8_cast_transient_bytes(module, dtype)


def _peak_layer_bytes(transformer: torch.nn.Module, dtype: torch.dtype, weights_resident: bool) -> int:
    """The most one layer of `transformer` needs while it runs: its streamed copy (none for resident weights) plus,
    for a leaf layer, its dequantization and fp8 cast transients."""
    peak = 0
    for module in transformer.modules():
        streamed = 0 if weights_resident else _stored_bytes(module)
        transient = _transient_bytes(module, dtype) if next(module.children(), None) is None else 0
        peak = max(peak, streamed + transient)
    return peak


def _sidecar_residual_bytes(
    transformer: torch.nn.Module, patches: Sequence[ModelPatchRaw], dtype: torch.dtype, weights_resident: bool
) -> int:
    """The largest per-forward residual a patch that is no `LoRALayer` costs on a layer patched as a sidecar (see the
    module docstring); 0 without such a patch. Layer keys are resolved as `LayerPatcher` resolves them."""
    peak = 0
    for patch in patches:
        flattened = not any("." in key for key in patch.layers)
        for key, layer in patch.layers.items():
            if isinstance(layer, LoRALayer) or not key.startswith(KREA2_LORA_TRANSFORMER_PREFIX):
                continue
            try:
                _, module = LayerPatcher._get_submodule(
                    transformer, key[len(KREA2_LORA_TRANSFORMER_PREFIX) :], layer_key_is_flattened=flattened
                )
            except AttributeError:
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor):
                continue
            transient = _transient_bytes(module, dtype)
            if weights_resident and not transient:
                # A resident dense layer is patched directly: its weight changes once, not per forward.
                continue
            streamed = 0 if weights_resident else _stored_bytes(module)
            residual = math.prod(weight.shape) * (_RESIDUAL_BYTES_PER_ELEMENT + dtype.itemsize)
            peak = max(peak, streamed + transient + residual)
    return peak


def estimate_krea2_denoise_working_memory(
    transformer: torch.nn.Module,
    *,
    image_seq_len: int,
    text_seq_len: int,
    style_text_seq_len: int = 0,
    reference_seq_len: int = 0,
    cache_references: bool = False,
    cfg: bool = False,
    loras: Sequence[tuple[ModelPatchRaw, int]] = (),
    regional_mask_bytes: int = 0,
    style_kv_bytes: int = 0,
    weights_resident: bool = False,
    device: torch.device,
    dtype: torch.dtype,
) -> WorkingMemoryEstimate:
    """The working memory the denoise needs on `device` in `dtype` while it holds `transformer`'s lock.

    `transformer` is the Krea-2 transformer about to be locked, read unlocked for its geometry and its layers' streamed
    copies and transients. `text_seq_len` is the longer caption of a CFG pair, `style_text_seq_len` the caption of a
    style reference's pass (0 without one), `reference_seq_len` every edit reference's tokens, which run in the
    per-step sequence or, with `cache_references`, once alone into a K/V cache. `cfg` is whether a negative pass runs.
    `loras` are the patches applied under the lock, each with its file's size. `regional_mask_bytes` is regional
    prompting's masks and bias (a mask is then passed to the attention), `style_kv_bytes` a style reference's captured
    K/V (0 without one). `weights_resident` is True only where the caller knows every weight stays on the device;
    otherwise the largest layer is reserved for streaming, and every patched layer may be a sidecar.
    """
    dtype_name = _DTYPE_NAMES.get(dtype)
    if dtype_name is None:
        raise ValueError(
            f"Working-memory estimator '{KREA2_DENOISE_SPEC.id}' has no {dtype} line on {platform_for(device)}: the "
            "transformer runs in a dtype it is not measured for."
        )
    config = transformer.config
    joint_references = 0 if cache_references else reference_seq_len
    cached_references = reference_seq_len if cache_references else 0
    step_tokens = image_seq_len + text_seq_len + joint_references
    style_tokens = image_seq_len + style_text_seq_len if style_text_seq_len else 0
    score_side = max(step_tokens, style_tokens)
    if cached_references:
        area = max(step_tokens * (step_tokens + cached_references), cached_references**2)
        score_side = math.isqrt(area - 1) + 1
    has_attention_mask = regional_mask_bytes > 0
    blocks = cast(torch.nn.ModuleList, transformer.transformer_blocks)
    patches = [patch for patch, _ in loras]
    features: dict[str, FeatureValue] = {
        "image_tokens": image_seq_len,
        "text_tokens": text_seq_len,
        "style_text_tokens": style_text_seq_len,
        "reference_tokens": joint_references,
        "cached_reference_tokens": cached_references,
        "tokens": max(step_tokens, style_tokens, cached_references),
        "dtype": dtype_name,
        "has_attention_mask": has_attention_mask,
        "cfg": cfg,
        "score_matrix_bytes": sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            num_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
            seq_len=score_side,
            has_attn_mask=has_attention_mask,
        ),
        "streamed_layer_bytes": 0
        if weights_resident
        else max((_stored_bytes(module) for module in blocks.modules()), default=0),
        "dequant_transient_bytes": peak_dequant_transient_bytes(blocks, dtype),
        "fp8_cast_transient_bytes": peak_fp8_cast_transient_bytes(blocks, dtype),
        "peak_layer_bytes": _peak_layer_bytes(transformer, dtype, weights_resident),
        "model_dequant_transient_bytes": peak_dequant_transient_bytes(transformer, dtype),
        "sidecar_residual_bytes": _sidecar_residual_bytes(transformer, patches, dtype, weights_resident),
        "regional_mask_bytes": regional_mask_bytes,
        "reference_kv_bytes": reference_kv_cache_bytes(
            num_layers=len(blocks),
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.attention_head_dim,
            reference_seq_len=cached_references,
            element_size=dtype.itemsize,
        ),
        "style_kv_bytes": style_kv_bytes,
        "lora_count": len(loras),
        "lora_bytes": sum(size for _, size in loras),
    }
    return KREA2_DENOISE_SPEC.estimate(features, device)
