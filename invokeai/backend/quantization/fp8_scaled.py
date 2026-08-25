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
from logging import Logger
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

FP8_DTYPE = torch.float8_e4m3fn

# Every float8 dtype a checkpoint may store weights in. Scale *recovery* must cover all of them:
# only `float8_e4m3fn` can stay quantized (see `can_stay_quantized`), but an `e5m2` weight still
# needs its `weight_scale` folded in on the way to bf16. Gating extraction on e4m3fn alone dropped
# the scale key and then cast the weight unscaled — off by `1/weight_scale`, silently.
FP8_WEIGHT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

WEIGHT_SCALE_SUFFIXES = (".weight_scale", ".scale_weight")
# Both spellings occur, exactly as for the weight scale. ComfyUI normalizes `.scale_input` to
# `.input_scale` on load (comfy/utils.py, convert_old_quants); reading only one of them means a
# calibrated activation scale is silently discarded and every forward pays the amax reduction.
INPUT_SCALE_SUFFIXES = (".input_scale", ".scale_input")

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


def read_safetensors_metadata(path: Path, logger: Logger | None = None) -> dict[str, str] | None:
    """Read the safetensors header metadata, or None if it cannot be read.

    Only used to enrich fp8 handling (per-layer ``full_precision_matrix_mult`` hints), so an
    unreadable header must not fail the model load. It is warned about rather than swallowed: without
    the hints, layers the quantizer marked as unsafe would silently be multiplied in fp8.
    """
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            return f.metadata()
    except Exception as e:
        if logger is not None:
            logger.warning(f"Could not read safetensors metadata from {path.name} ({e}); fp8 layer hints unavailable.")
        return None


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


# Key prefixes redistributors wrap a transformer in. Loaders strip these off the state dict before
# anything else, so `_quantization_metadata` — which is read from the file and still carries them —
# has to be stripped the same way.
TRANSFORMER_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "net.")


def strip_layer_path_prefix(
    layer_hints: Mapping[str, Any],
    prefixes: Iterable[str] = TRANSFORMER_KEY_PREFIXES,
) -> dict[str, Any]:
    """Re-key ``layer_hints`` as if the checkpoint prefix had been stripped from their names.

    ``_quantization_metadata`` lives in the safetensors header, so its layer names are in the
    file's own scheme — ``model.diffusion_model.blocks.0.attn.wq`` — while the state dict has had
    that prefix removed before the scales are extracted. A hint whose name still carries the prefix
    matches no layer, so ``full_precision_matrix_mult`` is silently ignored and the producer's
    "do not multiply this one in fp8" instruction is disregarded: exactly the failure the hint
    plumbing exists to prevent.

    Names that carry none of ``prefixes`` are passed through unchanged. Dropping them instead — as
    running the names through a strip function that filters by prefix would — turns a
    partially-prefixed header into a silently truncated one, and can abort the load.
    """
    out: dict[str, Any] = {}
    for name, hints in layer_hints.items():
        if isinstance(name, str):
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
        out[name] = hints
    return out


# Every per-layer side-channel suffix that belongs to a module rather than being a tensor of its
# own. Used by the detach/reattach pair below.
LAYER_SIDECHANNEL_SUFFIXES = WEIGHT_SCALE_SUFFIXES + INPUT_SCALE_SUFFIXES + (COMFY_QUANT_SUFFIX,)


def detach_layer_sidechannel(sd: dict[str, Any]) -> dict[str, list[tuple[str, Any]]]:
    """Pop every per-layer quantization side-channel entry, keyed by the module path it belongs to.

    For loaders that rename checkpoint keys. Key converters are written against ``.weight`` — they
    match it as a substring, or test whole keys for equality — so a sibling ``.scale_weight`` or
    ``.input_scale`` is *not* carried along, and neither is any scale on a key the converter renames
    by equality. The scale is then orphaned under its old path while its weight moves, and
    :func:`extract_fp8_scaled_layers` drops it because no fp8 weight sits at the old path any more.
    The weight stays quantized with no scale attached and is off by ``1/weight_scale``, in silence:
    the layer never enters ``fp8_layers``, so :func:`warn_on_unattached_scales` cannot see it either.

    Take the side channel out of the way, convert, then :func:`reattach_layer_sidechannel`.
    """
    detached: dict[str, list[tuple[str, Any]]] = {}
    for key in list(sd.keys()):
        if not isinstance(key, str):
            continue
        for suffix in LAYER_SIDECHANNEL_SUFFIXES:
            if key.endswith(suffix):
                detached.setdefault(key[: -len(suffix)], []).append((suffix, sd.pop(key)))
                break
    return detached


def reattach_layer_sidechannel(
    sd: dict[str, Any],
    detached: Mapping[str, list[tuple[str, Any]]],
    path_map: Mapping[str, str],
) -> list[str]:
    """Put detached side-channel entries back under their renamed module paths.

    Returns the module paths that could not be placed. A path with no entry in ``path_map`` had no
    destination in the converted state dict — usually because the converter drops that module
    outright — so its scale is dropped with it. Returning them rather than swallowing them lets the
    caller say so: a *silently* dropped scale is exactly the failure this pair exists to prevent.
    """
    orphaned: list[str] = []
    for path, entries in detached.items():
        destination = path_map.get(path, path)
        if f"{destination}.weight" not in sd:
            orphaned.append(path)
            continue
        for suffix, value in entries:
            sd[f"{destination}{suffix}"] = value
    return orphaned


def iter_weight_scale_pairs(sd: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield ``(weight_key, scale_key)`` for every weight scale in ``sd``, in either spelling.

    For loaders that fold the scale into the weight themselves instead of going through
    :func:`extract_fp8_scaled_layers`. Matching only ``.weight_scale`` is the failure that keeps
    recurring: a ``.scale_weight`` checkpoint then either loses its scales silently (if the loader
    strips both spellings afterwards, leaving the weight off by ``1/weight_scale``) or trips
    ``load_state_dict(..., strict=True)`` on the leftover key. Pairs whose ``.weight`` is absent are
    skipped, so a stray scale cannot invent one.
    """
    for key in list(sd.keys()):
        if not isinstance(key, str):
            continue
        for suffix in WEIGHT_SCALE_SUFFIXES:
            if key.endswith(suffix):
                weight_key = f"{key[: -len(suffix)]}.weight"
                if weight_key in sd:
                    yield weight_key, key
                break


# Per-layer quantization side-channel entries that sit next to a fused `qkv.weight` and therefore
# have to be carried through a split of it. The scale spellings are the ones this module accepts;
# the marker is a JSON blob describing the layer, identical for all three parts of the split.
QKV_SPLIT_SIDECHANNEL_SUFFIXES = ("weight_scale", "scale_weight", "input_scale", "scale_input", "comfy_quant")


def split_qkv_sidechannel(key: str, value: Any) -> tuple[Any, Any, Any]:
    """Split a fused-QKV scale/marker into the parts belonging to Q, K and V.

    A per-tensor scale (and any marker blob) describes the whole fused tensor, so each third
    inherits it unchanged. A per-output-channel scale has one entry per row and is split exactly
    like the weight.

    Getting this wrong is silent: a scale left on the fused path is keyed on a module the split
    model does not have, so `attach_fp8_scales` finds nothing and the three weights stay quantized
    but *unscaled* -- off by 1/weight_scale, with no error anywhere.
    """
    tensor = torch.as_tensor(value) if hasattr(value, "shape") else value
    if not hasattr(tensor, "shape") or tensor.dim() == 0 or tensor.shape[0] == 1:
        return (tensor, tensor, tensor)
    if tensor.numel() == 1 or key.endswith(("comfy_quant", "input_scale", "scale_input")):
        # A marker blob is a 1-D byte string, not a per-channel vector -- never split it.
        return (tensor, tensor, tensor)
    if tensor.shape[0] % 3 != 0:
        raise ValueError(
            f"Cannot split fused QKV quantization data '{key}': first dimension ({tensor.shape[0]}) is "
            "neither 1 nor divisible by 3, so it matches neither a per-tensor nor a per-channel scale."
        )
    third = tensor.shape[0] // 3
    return (tensor[:third], tensor[third : 2 * third], tensor[2 * third :])


def is_scale_metadata_key(key: Any) -> bool:
    """Whether ``key`` is fp8 scale/quantization metadata rather than a model tensor.

    Covers both spellings of the weight and input scales plus the marker keys producers emit, so
    callers strip exactly what they were able to interpret.
    """
    if not isinstance(key, str):
        return False
    return (
        key.endswith(WEIGHT_SCALE_SUFFIXES)
        or key.endswith(INPUT_SCALE_SUFFIXES)
        or COMFY_QUANT_SUFFIX.strip(".") in key
        or key in STRAY_METADATA_KEYS
    )


def _strip_scale_suffix(key: str) -> tuple[str, bool] | None:
    """Return ``(module path, is_input_scale)``, or None if ``key`` is not a scale key."""
    for suffix in WEIGHT_SCALE_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], False
    for suffix in INPUT_SCALE_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], True
    return None


def _usable_input_scale(scale: torch.Tensor | None) -> torch.Tensor | None:
    """A calibrated static activation scale, or None to fall back to per-forward ``amax`` scaling.

    An ``input_scale`` of exactly 1.0 is a placeholder: the producer wrote the field without
    calibrating it. Taking it at face value replaces the dynamic scale with *no scaling at all*, so
    every activation above the fp8 maximum saturates — far worse than the dynamic path it
    suppresses. ComfyUI drops the key in exactly this case (comfy/utils.py, convert_old_quants).

    Non-finite and non-positive scales are rejected for the same reason: they cannot be a valid
    divisor, and using one would produce inf/NaN activations instead of a slightly worse image.
    """
    if scale is None:
        return None
    scale = scale.float().reshape(())
    if not torch.isfinite(scale) or scale <= 0 or scale.item() == 1.0:
        return None
    return scale


def _normalize_weight_scale(scale: torch.Tensor) -> torch.Tensor:
    """Canonical float32 form of a weight scale, preserving its layout.

    Per-tensor scales become 0-d and per-output-channel scales 1-D, so downstream code can branch on
    ``numel()``. A scale with more than one dimension is *block-wise* — one entry per block of
    weight elements — and is returned with its shape intact: flattening it destroys the block
    geometry that :func:`expand_weight_scale` needs to line it back up with the weight.
    """
    scale = scale.float()
    if scale.numel() == 1:
        return scale.reshape(())
    if scale.dim() > 1:
        return scale
    return scale.flatten()


def expand_weight_scale(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Broadcast ``scale`` to line up with ``weight`` for an elementwise multiply.

    Handles the three layouts producers emit:

    - per-tensor (0-d / single element) — returned unchanged, broadcasting handles it;
    - per-output-channel (one entry per row) — reshaped to ``(rows, 1, ...)``;
    - block-wise (one entry per block along one or more dims) — each axis is
      ``repeat_interleave``d by that axis' block size.

    Without the block-wise case a 2-D scale reaches the multiply as-is and raises a shape error, so
    a checkpoint using that layout fails to load outright. That is the layout ComfyUI's own
    dequantizer expands, and the FLUX.2 loader used to expand before this module centralized the
    logic.
    """
    if scale.numel() == 1:
        return scale
    if scale.dim() <= 1:
        return scale.reshape(-1, *([1] * (weight.dim() - 1)))
    for dim in range(weight.dim()):
        if dim < scale.dim() and scale.shape[dim] != weight.shape[dim]:
            block = weight.shape[dim] // scale.shape[dim]
            if block > 1:
                scale = scale.repeat_interleave(block, dim=dim)
    return scale


def is_matmul_usable_scale(weight: Any, scale: torch.Tensor) -> bool:
    """Whether ``scaled_mm_linear`` can apply ``scale`` without materializing the weight.

    It handles exactly two layouts: a per-tensor scalar, which goes into the ``_scaled_mm`` call,
    and a per-output-channel vector, which is applied to the *result* (scaling weight row ``j``
    scales output column ``j``). A block-wise scale is separable in neither sense, so such a layer
    has to be dequantized up front instead of failing mid-generation inside the kernel.
    """
    if scale.numel() == 1:
        return True
    rows = getattr(weight, "shape", (None,))[0]
    return scale.dim() == 1 and scale.numel() == rows


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
        path, is_input_scale = parsed
        if is_input_scale:
            input_scales[path] = sd.pop(key)
        else:
            weight_scales[path] = sd.pop(key)

    for key in list(sd.keys()):
        if isinstance(key, str) and (key in STRAY_METADATA_KEYS or "comfy_quant" in key):
            del sd[key]

    layers: dict[str, Fp8ScaledLayer] = {}
    for path, scale in weight_scales.items():
        weight = sd.get(f"{path}.weight")
        if weight is None or getattr(weight, "dtype", None) not in FP8_WEIGHT_DTYPES:
            # A scale without an fp8 weight means the weight was already dequantized (or the key
            # naming does not line up). Applying the scale later would corrupt it, so drop it.
            continue
        hints = layer_meta.get(path, {})
        layers[path] = Fp8ScaledLayer(
            weight_scale=_normalize_weight_scale(scale),
            input_scale=_usable_input_scale(input_scales.get(path)),
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
        weight = weight.float()
        sd[key] = (weight * expand_weight_scale(weight, layer.weight_scale)).to(dtype)
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

    Returns the number of modules that were annotated. A caller that expects *every* recovered
    layer to be annotated should check that count with :func:`warn_on_unattached_scales`.
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


def warn_on_unattached_scales(logger: Logger, what: str, attached: int, layers: Mapping[str, Any]) -> None:
    """Complain when :func:`attach_fp8_scales` annotated fewer modules than there were layers.

    Every recovered layer should reach a module: :func:`split_fp8_scaled_layers` has already folded
    the ones that cannot stay quantized. A shortfall therefore means a scale went nowhere, and a
    weight is now off by ``1/weight_scale`` — visually a broken or washed-out generation, with
    nothing in the log to point at it. Loaders otherwise report ``attached`` as if it were the whole
    story, which reads as success.
    """
    missing = len(layers) - attached
    if missing > 0:
        logger.warning(
            f"{what}: {missing} of {len(layers)} scaled fp8 layer(s) did not receive their weight_scale. "
            "Those weights are quantized but unscaled, which will degrade output. This is a bug — "
            "please report the checkpoint."
        )


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


def _probe_fp8_matmul(index: int) -> bool | None:
    """Run one minimal ``_scaled_mm`` on device ``index`` and report whether it worked.

    Asking the device is the only reliable test. ``get_device_capability`` reports the *gfx arch*
    on ROCm, not an SM version, so an RDNA3 card (gfx1100) answers ``(11, 0)`` and sails past a
    ``>= (8, 9)`` check — then every forward raises ``torch._scaled_mm is only supported on CUDA
    devices with compute capability >= 9.0 or 8.9, or ROCm MI300+``. The whole point of the
    capability gate is to *fall back* rather than raise mid-generation, so it must not itself be a
    guess. The probe is one 16x16 matmul, run once per device and cached.

    Returns ``None`` when the probe could not be *carried out* — an allocation failure rather than
    an unsupported operation. The probe runs during a model load, i.e. under real VRAM pressure, and
    caching a momentary OOM as "this GPU cannot do fp8" would disable the fp8 matmul for the rest of
    the process. Same reasoning as `_device_supports_fp8_storage`, which also refuses to cache a
    transient failure.
    """
    try:
        device = torch.device("cuda", index)
        # Column-major right operand, i.e. exactly the layout `scaled_mm_linear` feeds it.
        lhs = torch.zeros((_MM_ALIGNMENT, _MM_ALIGNMENT), device=device, dtype=FP8_DTYPE)
        rhs = torch.zeros((_MM_ALIGNMENT, _MM_ALIGNMENT), device=device, dtype=FP8_DTYPE).t()
        scale = torch.ones((1, 1), device=device, dtype=torch.float32)
        torch._scaled_mm(lhs, rhs, scale, scale, out_dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        return None
    except RuntimeError as e:
        # cuBLAS reports a failed workspace allocation as a plain RuntimeError, not an OOM.
        if "out of memory" in str(e).lower() or "ALLOC_FAILED" in str(e):
            return None
        return False
    except Exception:
        return False
    return True


def device_supports_fp8_matmul(device: torch.device) -> bool:
    """Whether ``torch._scaled_mm`` can run on this device (Ada/SM 8.9 and newer, or MI300+)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _fp8_mm_supported.get(index)
    if cached is not None:
        return cached
    # The capability check is only a cheap pre-filter that spares older CUDA cards the probe;
    # it is deliberately not trusted on its own (see `_probe_fp8_matmul`).
    if not (torch.version.hip is not None or torch.cuda.get_device_capability(index) >= (8, 9)):
        _fp8_mm_supported[index] = False
        return False
    probed = _probe_fp8_matmul(index)
    if probed is None:
        # Inconclusive: answer this call conservatively but leave the cache empty so the next load
        # re-probes instead of the process being stuck without fp8 after one transient OOM.
        return False
    _fp8_mm_supported[index] = probed
    return probed


def reset_fp8_matmul_support_cache() -> None:
    """Forget the probed per-device support. For tests; the answer cannot change at runtime."""
    _fp8_mm_supported.clear()


def should_keep_fp8_weights(device: torch.device) -> bool:
    """Whether fp8 weights in a checkpoint should survive the load instead of being dequantized.

    Only true when the fp8 matmul is both enabled and usable, because keeping weights quantized
    without it is the worst of both worlds: the same VRAM as fp8 but a dequantize round trip on
    every forward (measured slower than plain bf16).
    """
    return is_fp8_matmul_enabled() and device_supports_fp8_matmul(device)


def _is_fp8_matmul_weight(key: str, tensor: Any, model: torch.nn.Module | None) -> bool:
    """Whether this state-dict entry is a weight `scaled_mm_linear` can actually consume.

    Only the ``.weight`` of an ``nn.Linear`` qualifies. This matters because checkpoints exist that
    quantize *everything* — biases, norm weights, even learned pad tokens. Keeping those in fp8 does
    not save anything worth having and actively breaks inference: an fp8 norm or pad token flows
    into the activations, and the next Linear then receives an fp8 *input*, which dies in
    ``x.abs()`` with ``"abs_cuda" not implemented for 'Float8_e4m3fn'``. Observed on a Z-Image
    checkpoint where 243 of 453 fp8 tensors were 1-D.
    """
    if not key.endswith(".weight") or getattr(tensor, "dim", None) is None or tensor.dim() < 2:
        return False
    if model is None:
        # No model to resolve against: the 2-D + `.weight` shape test above is the safe subset.
        return True
    try:
        module = model.get_submodule(key[: -len(".weight")])
    except AttributeError:
        return False
    return isinstance(module, torch.nn.Linear)


def can_stay_quantized(
    key: str,
    tensor: Any,
    model: torch.nn.Module | None,
    skip_patterns: Iterable[str] = (),
) -> bool:
    """Whether this state-dict entry may be left in fp8 by :func:`cast_state_dict`.

    Single source of truth for that decision: the loaders reserve RAM against it
    (:func:`predict_cast_state_dict_size`) and decide which scaled layers survive with it
    (:func:`split_fp8_scaled_layers`), so the three must never drift apart.

    Only ``float8_e4m3fn`` qualifies. ``float8_e5m2`` is always cast because
    :func:`scaled_mm_linear` cannot use it as the weight operand on Ada, so keeping it quantized
    would buy VRAM at the cost of a per-forward dequantize.

    ``skip_patterns`` are substrings of the state-dict key whose weights must be dequantized even
    when the rest stays fp8. Pass the model's ``_skip_layerwise_casting_patterns``: diffusers uses
    it to mark precision-sensitive modules, and some of them *read their own weight's dtype and cast
    their activations to it*. Z-Image's ``TimestepEmbedder.forward`` does exactly that
    (``t_freq.to(self.mlp[0].weight.dtype)``), so leaving its weight in fp8 hands the next Linear an
    fp8 activation and the forward dies in ``x.abs()`` with
    ``"abs_cuda" not implemented for 'Float8_e4m3fn'``.
    """
    return (
        getattr(tensor, "dtype", None) is FP8_DTYPE
        and _is_fp8_matmul_weight(key, tensor, model)
        and not any(pattern in key for pattern in skip_patterns)
    )


def _is_castable_float(tensor: Any) -> bool:
    """Whether ``tensor`` is a floating-point payload that may be cast to the compute dtype."""
    is_floating_point = getattr(tensor, "is_floating_point", None)
    if not callable(is_floating_point):
        return False
    try:
        return bool(is_floating_point())
    except Exception:
        return False


def cast_state_dict(
    sd: dict[str, Any],
    dtype: torch.dtype,
    *,
    keep_fp8: bool,
    model: torch.nn.Module | None = None,
    skip_patterns: Iterable[str] = (),
) -> int:
    """Cast every tensor in ``sd`` to ``dtype`` in place, optionally leaving fp8 weights quantized.

    Loaders historically cast the whole state dict unconditionally, which silently dequantizes a
    checkpoint that ships raw fp8 weights (fp8 tensors with no ``weight_scale`` alongside them) —
    the VRAM saving and the tensor cores are both thrown away before the model is ever built.

    A *scaled* fp8 weight must never reach this function still carrying an unapplied scale: the
    plain ``tensor.to(dtype)`` below drops the scale silently, leaving the weight off by
    ``1/weight_scale``. Run :func:`split_fp8_scaled_layers` first — it folds the scale into exactly
    those layers this function would cast.

    Returns the number of tensors left in fp8.
    """
    patterns = tuple(skip_patterns)
    kept = 0
    for key in sd:
        tensor = sd[key]
        if keep_fp8 and can_stay_quantized(key, tensor, model, patterns):
            kept += 1
            continue
        if not _is_castable_float(tensor):
            # Integer payloads (embedding indices, packed buffers) are not weights and must keep
            # their dtype. Loaders used to guard this themselves; centralizing it here means a
            # loader that switches to `cast_state_dict` does not silently lose the guard.
            continue
        sd[key] = tensor.to(dtype)
    return kept


def predict_cast_state_dict_size(
    sd: Mapping[str, Any],
    dtype: torch.dtype,
    *,
    keep_fp8: bool,
    model: torch.nn.Module | None = None,
    skip_patterns: Iterable[str] = (),
) -> int:
    """Bytes the state dict will occupy once :func:`cast_state_dict` has run over it.

    Loaders call this to size their ``make_room()`` reservation. Charging 1 byte/element for every
    fp8 tensor is wrong in the direction that hurts: only 2-D ``nn.Linear`` weights outside the skip
    patterns stay quantized, and everything else — biases, norms, learned pad tokens, the
    deliberately-dequantized precision-sensitive Linears — lands at ``dtype.itemsize``. On a
    checkpoint that quantized all 453 of its tensors that under-count is most of the difference.
    """
    patterns = tuple(skip_patterns)
    total = 0
    for key, tensor in sd.items():
        if (keep_fp8 and can_stay_quantized(key, tensor, model, patterns)) or not _is_castable_float(tensor):
            total += tensor.nelement() * tensor.element_size()
        else:
            total += tensor.nelement() * dtype.itemsize
    return total


def split_fp8_scaled_layers(
    sd: dict[str, Any],
    layers: Mapping[str, Fp8ScaledLayer],
    dtype: torch.dtype,
    *,
    model: torch.nn.Module | None = None,
    skip_patterns: Iterable[str] = (),
) -> dict[str, Fp8ScaledLayer]:
    """Dequantize the scaled layers that cannot stay quantized; return the ones that can.

    Every filter that keeps a weight *out* of fp8 — a skip pattern, a weight that is not a 2-D
    ``nn.Linear.weight`` — is a filter that would otherwise let :func:`cast_state_dict` do a plain
    ``.to(dtype)`` on a scaled weight, i.e. drop its ``weight_scale`` and leave the weight off by
    ``1/weight_scale``. :func:`attach_fp8_scales` cannot repair that afterwards: it skips any module
    whose weight is no longer fp8, so the scale is lost for good. Krea-2 hits this on ordinary
    ComfyUI exports, where ``time_embed.linear_1/linear_2`` are quantized like any other Linear and
    match the model's ``time_embed`` skip pattern.

    So the filters are applied here instead, *before* the cast, and the affected layers go through
    :func:`dequantize_fp8_scaled`, which applies the scale properly. They are then dropped from the
    returned mapping — they are no longer fp8, so there is nothing left to attach.

    A scale layout :func:`scaled_mm_linear` cannot apply — block-wise, or a vector that does not
    match the weight's row count — is dequantized here too. Left quantized it would fail inside the
    kernel mid-generation instead. Doing it here rather than in :func:`can_stay_quantized` keeps the
    RAM accounting honest: every caller runs this before
    :func:`predict_cast_state_dict_size`, so the prediction sees the already-widened tensor.

    ``float8_e5m2`` layers land here by way of :func:`can_stay_quantized`, which admits only
    ``float8_e4m3fn`` — they are dequantized *with* their scale applied rather than losing it.
    """
    patterns = tuple(skip_patterns)
    usable: dict[str, Fp8ScaledLayer] = {}
    unusable: dict[str, Fp8ScaledLayer] = {}
    for path, layer in layers.items():
        key = f"{path}.weight"
        tensor = sd.get(key)
        if (
            tensor is not None
            and can_stay_quantized(key, tensor, model, patterns)
            and is_matmul_usable_scale(tensor, layer.weight_scale)
        ):
            usable[path] = layer
        else:
            unusable[path] = layer
    if unusable:
        dequantize_fp8_scaled(sd, unusable, dtype)
    return usable


def count_fp8_weights(model: torch.nn.Module) -> int:
    """Number of parameters already stored as ``float8_e4m3fn``.

    Used to tell a checkpoint that arrived quantized apart from one this loader is about to
    quantize itself — the two must not both happen (see `ModelLoader._apply_fp8_layerwise_casting`).
    """
    return sum(1 for p in model.parameters() if p.dtype is FP8_DTYPE)


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
    return out * expand_weight_scale(out, scale)


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
