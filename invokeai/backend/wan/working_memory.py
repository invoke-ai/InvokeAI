"""The working memory one Wan 2.2 transformer forward needs on its compute device, as the denoise nodes reserve it.

The fitted part is the activations of one forward, which scale with the tokens it attends over: the latent's after the
1x2x2 patchify, ``frames * (height // 2) * (width // 2)``. The caption runs through cross-attention only, a fixed 512
tokens. Conditional and unconditional passes run one after the other, so CFG adds nothing. The lines are fitted per
geometry and dtype (`_line`): the A14B experts (width 5120, FFN 13824), TI2V-5B (3072, 14336) with one timestep per
sample, and TI2V-5B with one timestep per token, which its image-to-video runs pass: diffusers then keeps the timestep
projection as ``[1, tokens, 6, 3072]`` for the whole forward and each block builds a float32 modulation table of that
shape. The coefficients and the points they were fitted on ship in `invokeai/backend/working_memory/data/wan.json`,
measured by `scripts/calibrate_wan_working_memory.py`; their provenance is in the data.

The entry is keyed by no platform trait (as the MOSS-Image denoise's): the fitted term is the activations of GEMMs and
fused attention, not convolution workspaces. A build without a fused SDPA kernel for these shapes is what moves it, and
the analytic score-matrix term prices that. Measured only on a W7900 under ROCm on Linux with the native allocator.

SageAttention (`attention_backend: sage`) is priced as SDPA: holding its quantized copies of Q, K and V over every
attention call of the fit grid's largest forwards did not raise the forward's peak, which the feed-forward sets (see the
`sage` check of the calibration script).

The analytic terms:

- Where SDPA has no fused kernel for these shapes it materializes the score matrices (`sdpa_score_matrix_bytes`).
- The dequantization copy a GGUF or int8 build materializes per quantized linear and forward
  (`peak_dequant_transient_bytes`), and the compute-dtype copies of float8-stored weights
  (`peak_fp8_cast_transient_bytes`); zero for a dense build.
- The LoRAs' tensors and the update their sidecars form per forward (`lora_sidecar_bytes`), one row per token except
  for the linears that read the timestep embedding, which see one per sample unless the timesteps are per token.
"""

from collections.abc import Mapping, Sequence
from typing import Final, Optional, cast

import torch

from invokeai.backend.model_manager.taxonomy import ModelFormat, WanVariantType
from invokeai.backend.patches.lora_conversions.wan_lora_constants import WAN_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.fp8 import peak_fp8_cast_transient_bytes
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import Line, off_line_reasons
from invokeai.backend.working_memory.spec import EstimatorSpec, register

WAN_ATTENTION_HEAD_DIM: Final = 128
WAN_ATTENTION_HEADS: Final[Mapping[str, int]] = {"a14b": 40, "ti2v": 24}
"""Heads of 128 per geometry: widths 5120 and 3072."""
_GEOMETRIES: Final = ("a14b", "ti2v", "ti2v_token_timesteps")
_DTYPE_NAMES: Final[Mapping[torch.dtype, str]] = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
"""The dtypes `TorchDevice.choose_bfloat16_safe_dtype` runs the transformer in."""
_COEFFICIENTS: Final = frozenset(
    f"{part}_{geometry}_{dtype}" for part in ("k", "b") for geometry in _GEOMETRIES for dtype in _DTYPE_NAMES.values()
)
# The linears that read the timestep embedding: one row per sample, not per token, unless the timesteps are per token.
_PER_SAMPLE_LINEARS: Final = dict.fromkeys(
    (
        "condition_embedder.time_embedder.linear_1",
        "condition_embedder.time_embedder.linear_2",
        "condition_embedder.time_proj",
    ),
    1,
)


def wan_tokens(latent_shape: torch.Size) -> int:
    """Transformer tokens of a 5D latent ``[B, C, T, H, W]`` after the 1x2x2 patchify."""
    batch, _, frames, height, width = latent_shape
    return batch * frames * (height // 2) * (width // 2)


def _line(features: Mapping[str, FeatureValue]) -> Optional[Line]:
    """The forward's line, `<geometry>_<dtype>` over the attended tokens. Only the plain forward fits it: every
    analytic term of `_terms` zero. The label names the latent grid and each analytic term that is not zero, in bytes."""
    geometry, dtype, tokens = features.get("geometry"), features.get("dtype"), features.get("tokens")
    if not isinstance(geometry, str) or not isinstance(dtype, str) or not isinstance(tokens, int):
        return None
    if f"k_{geometry}_{dtype}" not in _COEFFICIENTS:
        return None
    off_line = off_line_reasons(_terms, features, _COEFFICIENTS)
    shape = f"{features.get('frames')}x{features.get('latent_height')}x{features.get('latent_width')}"
    return Line(f"{geometry}_{dtype}", tokens, not off_line, "-".join((geometry, dtype, shape, *off_line)))


def _terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    line = f"{features['geometry']}_{features['dtype']}"
    return (
        WorkingMemoryTerm.fitted("activations", f"k_{line}", cast(int, features["tokens"]), values),
        WorkingMemoryTerm.fitted("base", f"b_{line}", 1, values),
        WorkingMemoryTerm("attention_scores", cast(int, features["score_matrix_bytes"]), True),
        WorkingMemoryTerm("dequant_transient", cast(int, features["dequant_bytes"]), True),
        WorkingMemoryTerm("fp8_cast_transient", cast(int, features["fp8_cast_bytes"]), True),
        WorkingMemoryTerm("lora_sidecar", cast(int, features["lora_sidecar_bytes"]), True),
    )


WAN_DENOISE_SPEC = register(
    EstimatorSpec(
        id="wan.denoise",
        version=1,
        coefficients=_COEFFICIENTS,
        platform_keys=(),
        terms=_terms,
        line=_line,
    )
)


def estimate_wan_denoise_working_memory(
    transformer: torch.nn.Module,
    *,
    variant: WanVariantType,
    latent_shape: torch.Size,
    token_timesteps: bool = False,
    loras: Sequence[ModelPatchRaw] = (),
    model_format: ModelFormat,
    device: torch.device,
    dtype: torch.dtype,
) -> WorkingMemoryEstimate:
    """What one forward of the Wan expert `transformer` over a 5D latent of `latent_shape` needs on `device`.

    `transformer` is read unlocked, before the lock this reservation applies to. `token_timesteps` is whether the
    forward passes one timestep per token (TI2V-5B image-to-video). `loras` are the patches applied to this expert,
    with its `model_format`, which decides whether they run as sidecars.
    """
    dtype_name = _DTYPE_NAMES.get(dtype)
    if dtype_name is None:
        raise ValueError(f"Working-memory estimator '{WAN_DENOISE_SPEC.id}' has no {dtype} line.")
    ti2v = variant == WanVariantType.TI2V_5B
    geometry = ("ti2v_token_timesteps" if token_timesteps else "ti2v") if ti2v else "a14b"
    tokens = wan_tokens(latent_shape)
    features: dict[str, FeatureValue] = {
        "geometry": geometry,
        "dtype": dtype_name,
        "tokens": tokens,
        "frames": int(latent_shape[2]),
        "latent_height": int(latent_shape[3]),
        "latent_width": int(latent_shape[4]),
        "score_matrix_bytes": sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            num_heads=WAN_ATTENTION_HEADS["ti2v" if ti2v else "a14b"],
            head_dim=WAN_ATTENTION_HEAD_DIM,
            seq_len=tokens,
            via_diffusers_dispatch=True,
        ),
        "dequant_bytes": peak_dequant_transient_bytes(transformer, dtype),
        "fp8_cast_bytes": peak_fp8_cast_transient_bytes(transformer, dtype),
        "lora_sidecar_bytes": _lora_sidecar_bytes(
            transformer, loras, model_format, tokens, dtype, per_sample_timesteps=not token_timesteps
        ),
    }
    return WAN_DENOISE_SPEC.estimate(features, device)


def _lora_sidecar_bytes(
    transformer: torch.nn.Module,
    loras: Sequence[ModelPatchRaw],
    model_format: ModelFormat,
    tokens: int,
    dtype: torch.dtype,
    *,
    per_sample_timesteps: bool,
) -> int:
    if not loras:
        return 0
    # Imported here: the patch layers import the model manager's loaders, and `import_estimator_modules` imports this
    # module on its own.
    from invokeai.backend.patches.sidecar_working_memory import lora_sidecar_bytes

    return lora_sidecar_bytes(
        loras,
        transformer,
        model_format,
        prefix=WAN_LORA_TRANSFORMER_PREFIX,
        rows=tokens,
        per_sample_rows=_PER_SAMPLE_LINEARS if per_sample_timesteps else {},
        dtype=dtype,
    )
