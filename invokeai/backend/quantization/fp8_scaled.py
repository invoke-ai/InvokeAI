"""Shared handling for ComfyUI-style "scaled fp8" checkpoints.

These checkpoints store each quantized Linear as:

    <path>.weight         float8_e4m3fn
    <path>.weight_scale   float32   (usually a scalar; per-output-channel also occurs)
    <path>.input_scale    float32   (optional; a calibrated *static* activation scale)

so that ``w_real ≈ weight.to(float) * weight_scale``. Some producers use ``.scale_weight`` instead
of ``.weight_scale``. A ``_quantization_metadata`` entry in the safetensors header may additionally
mark individual layers with ``full_precision_matrix_mult``, meaning the producer determined that
this layer must not be multiplied in fp8.

Historically InvokeAI dequantized these to bf16 at load time (three near-identical implementations
in the FLUX.2, Z-Image and Qwen-Image loaders). That throws away both the VRAM saving and the
ability to run the matmul on the fp8 tensor cores. This module keeps the quantization intact so
that :class:`CustomLinear` can decide per forward what to do with it.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

FP8_DTYPE = torch.float8_e4m3fn

WEIGHT_SCALE_SUFFIXES = (".weight_scale", ".scale_weight")
INPUT_SCALE_SUFFIX = ".input_scale"

QUANT_METADATA_KEY = "_quantization_metadata"

# Per-layer marker tensor written by ComfyUI exports that do not use the header entry above.
COMFY_QUANT_SUFFIX = ".comfy_quant"

# Standalone marker keys some producers emit alongside the tensors. They carry no per-layer data.
STRAY_METADATA_KEYS = ("scaled_fp8",)


@dataclass
class Fp8ScaledLayer:
    """The quantization parameters recovered for a single Linear."""

    weight_scale: torch.Tensor
    """Scalar, or shape ``(out_features,)`` for per-output-channel quantization."""

    input_scale: torch.Tensor | None = None
    """Calibrated static activation scale, if the checkpoint ships one."""

    full_precision_matmul: bool = False
    """The producer marked this layer as unsafe for an fp8 matmul; only dequantized use is allowed."""

    def is_per_tensor(self) -> bool:
        return self.weight_scale.numel() == 1


def parse_quantization_metadata(metadata: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Parse the safetensors ``_quantization_metadata`` header entry into a per-layer mapping.

    Returns an empty dict when the entry is absent or unparseable - the scales themselves are the
    source of truth, the metadata only adds per-layer hints.
    """
    if not metadata:
        return {}
    raw = metadata.get(QUANT_METADATA_KEY)
    if not raw:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    if not isinstance(raw, dict):
        return {}
    layers = raw.get("layers")
    return layers if isinstance(layers, dict) else {}


def extract_comfy_quant_hints(sd: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Decode per-layer ``<path>.comfy_quant`` markers into the same mapping
    :func:`parse_quantization_metadata` produces, popping them out of ``sd``.

    ComfyUI writes the per-layer quantization flags in one of two places. Newer exports use the
    safetensors *header* (``_quantization_metadata``); others store one uint8 JSON blob per
    quantized layer as an ordinary tensor, e.g.::

        model.layers.0.mlp.down_proj.comfy_quant
            -> {"format": "float8_e4m3fn", "full_precision_matrix_mult": false}

    Both forms carry ``full_precision_matrix_mult`` - the producer's instruction that a layer must
    not be multiplied in fp8. A loader that reads only the header therefore *silently ignores that
    instruction* on a checkpoint using the per-tensor form, and runs an fp8 matmul the producer
    measured as unsafe. Checkpoints using the per-tensor form and marking layers this way exist in
    the wild, so both forms must be read.

    Malformed blobs are skipped rather than raised on: the marker is an optimization hint, and the
    scales themselves remain the source of truth.
    """
    import json

    hints: dict[str, dict[str, Any]] = {}
    for key in list(sd.keys()):
        if not isinstance(key, str) or not key.endswith(COMFY_QUANT_SUFFIX):
            continue
        raw = sd.pop(key)
        path = key[: -len(COMFY_QUANT_SUFFIX)]
        try:
            # A uint8 tensor holding UTF-8 JSON, sometimes NUL-padded to a fixed width.
            blob = bytes(raw.flatten().tolist()).decode("utf-8", errors="replace").rstrip("\x00")
            parsed = json.loads(blob)
        except Exception:
            continue
        if isinstance(parsed, dict):
            hints[path] = parsed
    return hints


def _strip_scale_suffix(key: str) -> tuple[str, str] | None:
    for suffix in WEIGHT_SCALE_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    if key.endswith(INPUT_SCALE_SUFFIX):
        return key[: -len(INPUT_SCALE_SUFFIX)], INPUT_SCALE_SUFFIX
    return None


def extract_fp8_scaled_layers(
    sd: dict[str, Any],
    metadata: Mapping[str, Any] | None = None,
    layer_hints: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Fp8ScaledLayer]:
    """Pop the quantization side-channel out of ``sd`` and return it keyed by module path.

    ``sd`` is modified in place: scale and marker keys are removed so the remaining state dict
    loads cleanly into a model that knows nothing about fp8. The ``.weight`` tensors are left as
    float8 - the caller decides whether to keep or dequantize them.

    Only layers whose ``.weight`` is actually float8 are reported; a stray scale without a matching
    fp8 weight is dropped rather than silently mis-scaling a bf16 weight.

    ``layer_hints`` overrides the parsed metadata. Loaders that rename checkpoint keys (native →
    diffusers) must pass hints keyed by the *renamed* paths, otherwise the per-layer flags -
    including ``full_precision_matrix_mult`` - silently match nothing.
    """
    layer_meta = dict(layer_hints) if layer_hints is not None else parse_quantization_metadata(metadata)

    weight_scales: dict[str, torch.Tensor] = {}
    input_scales: dict[str, torch.Tensor] = {}
    for key in list(sd.keys()):
        if not isinstance(key, str):
            continue
        parsed = _strip_scale_suffix(key)
        if parsed is None:
            continue
        path, suffix = parsed
        if suffix == INPUT_SCALE_SUFFIX:
            input_scales[path] = sd.pop(key)
        else:
            weight_scales[path] = sd.pop(key)

    for key in list(sd.keys()):
        if isinstance(key, str) and (key in STRAY_METADATA_KEYS or "comfy_quant" in key):
            del sd[key]

    layers: dict[str, Fp8ScaledLayer] = {}
    for path, scale in weight_scales.items():
        weight = sd.get(f"{path}.weight")
        if weight is None or getattr(weight, "dtype", None) != FP8_DTYPE:
            # A scale without an fp8 weight means the weight was already dequantized (or the key
            # naming does not line up). Applying the scale later would corrupt it, so drop it.
            continue
        hints = layer_meta.get(path, {})
        layers[path] = Fp8ScaledLayer(
            weight_scale=scale.float().reshape(()) if scale.numel() == 1 else scale.float().flatten(),
            input_scale=input_scales.get(path).float().reshape(()) if path in input_scales else None,
            full_precision_matmul=bool(hints.get("full_precision_matrix_mult", False)),
        )
    return layers


def dequantize_fp8_scaled(
    sd: dict[str, Any],
    layers: Mapping[str, Fp8ScaledLayer],
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Fold the scales back into the weights, producing a plain ``dtype`` state dict.

    This is the legacy behavior, kept as the fallback for models/devices that cannot use the fp8
    path. The multiply runs in float32 for precision but the result is stored as ``dtype``
    immediately, so a cold load never holds the whole model in float32.
    """
    for path, layer in layers.items():
        key = f"{path}.weight"
        weight = sd.get(key)
        if weight is None:
            continue
        scale = layer.weight_scale
        if scale.numel() > 1:
            scale = scale.reshape(-1, *([1] * (weight.dim() - 1)))
        sd[key] = (weight.float() * scale).to(dtype)
    return sd


def attach_fp8_scales(
    model: torch.nn.Module,
    layers: Mapping[str, Fp8ScaledLayer],
    module_paths: Iterable[str] | None = None,
) -> int:
    """Register the recovered scales as buffers on the matching modules.

    ``CustomLinear`` looks for ``weight_scale`` / ``input_scale`` buffers and a
    ``_fp8_full_precision_matmul`` flag. The buffers are non-persistent: they must never end up in
    ``state_dict()`` output, or a re-save would produce a checkpoint that gets scaled twice.

    The producer's ``full_precision_matrix_mult`` markers are applied here rather than dropped at
    parse time, so the recovered metadata stays inspectable, and are suppressed when the user has
    turned the hints off (see :func:`full_precision_hints_respected`).

    Returns the number of modules that were annotated.
    """
    wanted = set(module_paths) if module_paths is not None else None
    respect_hints = full_precision_hints_respected()
    count = 0
    for path, layer in layers.items():
        if wanted is not None and path not in wanted:
            continue
        try:
            module = model.get_submodule(path) if path else model
        except AttributeError:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype != FP8_DTYPE:
            continue
        module.register_buffer("weight_scale", layer.weight_scale.to(weight.device), persistent=False)
        if layer.input_scale is not None:
            module.register_buffer("input_scale", layer.input_scale.to(weight.device), persistent=False)
        module._fp8_full_precision_matmul = layer.full_precision_matmul and respect_hints
        count += 1
    return count


# ----------------------------------------------------------------------------------- runtime path

FP8_MAX = torch.finfo(FP8_DTYPE).max

# torch._scaled_mm requires every GEMM dimension to be a multiple of 16.
_MM_ALIGNMENT = 16

_fp8_mm_supported: dict[int, bool] = {}

# fp8 compute quantizes the *activations* as well, so it changes numerics: an existing install would
# start producing different images at the same seed. It is therefore opt-in (`fp8_compute` in
# invokeai.yaml) for one release before becoming the default.
#
# The same flag also decides whether scaled fp8 checkpoints stay quantized at load. Keeping them
# quantized without the fp8 matmul would halve VRAM but make generation *slower* (the dequantize
# round trip costs more than it saves), so the two must be switched together.
_fp8_matmul_override: bool | None = None


def set_fp8_matmul_enabled(enabled: bool | None) -> None:
    """Override the configured setting process-wide. Pass ``None`` to revert to the config value."""
    global _fp8_matmul_override
    _fp8_matmul_override = enabled


def is_fp8_matmul_enabled() -> bool:
    if _fp8_matmul_override is not None:
        return _fp8_matmul_override
    try:
        from invokeai.app.services.config.config_default import get_config

        return bool(get_config().fp8_compute)
    except Exception:
        # Backend code may run outside a configured app (scripts, tests). Default to the safe path.
        return False


_full_precision_hints_override: bool | None = None


def set_full_precision_hints_respected(respected: bool | None) -> None:
    """Override the configured setting process-wide. Pass ``None`` to revert to the config value."""
    global _full_precision_hints_override
    _full_precision_hints_override = respected


def full_precision_hints_respected() -> bool:
    """Whether ``full_precision_matrix_mult`` markers are obeyed.

    Honoring a marker means that layer dequantizes on every forward instead of using the fp8 tensor
    cores, and that is not cheap: on a checkpoint that marks the attention output, gate and FFN-down
    projections (a common choice) the marked layers can be ~40% of the quantized weights, and
    measurably erase most of the fp8_compute speedup. Whether that trade is worth it depends on how
    much the producer's flags actually buy in a given checkpoint, so it is a user-facing setting
    rather than a hard-coded policy.
    """
    if _full_precision_hints_override is not None:
        return _full_precision_hints_override
    try:
        from invokeai.app.services.config.config_default import get_config

        return bool(get_config().fp8_compute_full_precision_hints)
    except Exception:
        # Backend code may run outside a configured app (scripts, tests). Default to obeying them.
        return True


def device_supports_fp8_matmul(device: torch.device) -> bool:
    """Whether ``torch._scaled_mm`` can run on this device (Ada/SM 8.9 and newer)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _fp8_mm_supported.get(index)
    if cached is None:
        cached = torch.cuda.get_device_capability(index) >= (8, 9)
        _fp8_mm_supported[index] = cached
    return cached


def dequantize_weight(weight: torch.Tensor, weight_scale: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """Cast an fp8 weight up to ``dtype``, applying its scale if it has one.

    Used by the fallback path. Casting *without* the scale - which is what a plain ``.to(dtype)``
    does - silently produces a wrongly-scaled weight, so every dequantization of a scaled fp8
    weight must go through here.
    """
    out = weight.to(dtype)
    if weight_scale is None:
        return out
    scale = weight_scale.to(device=out.device, dtype=dtype)
    if scale.numel() > 1:
        scale = scale.reshape(-1, *([1] * (out.dim() - 1)))
    return out * scale


def scaled_mm_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``F.linear`` executed on the fp8 tensor cores via ``torch._scaled_mm``.

    ``weight`` stays float8 and is never materialized in a wider dtype - that is where both the
    VRAM and the speed come from.

    Scaling notes:
    - The activation scale is per-tensor. Ada rejects per-row activation scaling outright, and
      a static ``input_scale`` from the checkpoint is used when available (calibrated, and it saves
      the per-forward ``amax`` reduction).
    - A per-output-channel weight scale cannot be handed to ``_scaled_mm`` on Ada either, but it is
      separable: scaling row ``j`` of the weight scales output column ``j``, so it is applied to the
      result instead. Per-tensor scales go straight into the kernel.
    """
    orig_shape = input.shape
    x = input.reshape(-1, orig_shape[-1])

    # The transpose must be produced here rather than cached: a stored transposed view stops being a
    # view the moment the tensor is moved between devices (which partial loading does constantly),
    # silently doubling the weight memory.
    weight_t = weight.t()

    pad = (-x.shape[0]) % _MM_ALIGNMENT
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))

    if input_scale is not None:
        x_scale = input_scale.to(device=x.device, dtype=torch.float32).reshape(1, 1)
        x_fp8 = (x / x_scale.to(x.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    else:
        amax = x.abs().amax().clamp(min=1e-12)
        x_scale = (amax / FP8_MAX).float().reshape(1, 1)
        x_fp8 = (x / x_scale.to(x.dtype)).to(FP8_DTYPE)

    per_tensor = weight_scale is not None and weight_scale.numel() == 1
    if per_tensor:
        w_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1, 1)
    else:
        w_scale = torch.ones(1, 1, device=x.device, dtype=torch.float32)

    out = torch._scaled_mm(x_fp8.contiguous(), weight_t, x_scale, w_scale, out_dtype=input.dtype)

    if pad:
        out = out[: x.shape[0] - pad]
    if weight_scale is not None and not per_tensor:
        out = out * weight_scale.to(device=out.device, dtype=out.dtype).reshape(1, -1)

    out = out.reshape(*orig_shape[:-1], weight.shape[0])
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out
