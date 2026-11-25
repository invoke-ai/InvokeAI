"""Reserve for what a load is holding, not for half of it.

A single-file load recovers a quantization side channel out of its state dict -- scaled-fp8 scales
via `extract_fp8_scaled_layers`, nvfp4 packed payloads via `pop_nvfp4_layers` -- and then asks the
RAM cache for room. Those were separate steps, and the number was assembled at each seam:
`predict_cast_state_dict_size(...)` plus, where the loader remembered, `predict_nvfp4_install_size(...)`.
Fourteen call sites of the first and seven of the second, across eight loaders.

The gap is structural rather than careless. `predict_cast_state_dict_size` answers about the *state
dict*, exactly and by contract -- bytes it will occupy once the split and the cast have run. The
extraction has already popped the scale keys out of that dict, so that function cannot count them and
must not: tying the two together breaks the equality its own cells pin, because whether a scale
survives the split depends on whether its layer stayed fp8.

So the composition belongs to one function that is handed both halves. On Comfy-Org's released MXFP8
Krea-2 build the term nobody could add is 1.455 GiB: `decode_mx_block_scales` replaces each stored
`uint8` exponent grid with a float32 one four times the size, one entry per 32 weight elements, and
they stay resident from the extraction until the split consumes them.

A function rather than a value object, deliberately. An earlier draft was a frozen dataclass, and
each of its extra affordances turned into a defect: the generated constructor defaulted the two
mappings and so silently under-reserved, `frozen` generated `__eq__`/`__hash__` over dicts of tensors
that raise when called, and an instance bound to a local outlives the fold that frees the grids. No
caller wanted the decomposition -- they all want the reservation made.

`reserve` is a callable rather than the cache itself, matching `install_int8_convrot_layers(...,
reserve=...)` next door. That is not only for tests: it keeps `backend/quantization` from acquiring an
import edge into `backend/model_manager/load/model_cache`, which nothing in this package has today.
"""

import inspect
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any

import torch

from invokeai.backend.quantization.fp8_scaled import (
    Fp8ScaledLayer,
    fold_transient_bytes,
    is_castable_float,
    predict_cast_state_dict_size,
    scaled_layer_for,
    survives_split_and_cast,
)
from invokeai.backend.quantization.nvfp4 import (
    NVFP4Payload,
    predict_modelopt_nvfp4_relayout_host_size,
    predict_nvfp4_install_host_size,
    predict_nvfp4_install_size,
)
from invokeai.backend.util.ram import _source_is_mapped, require_ram_headroom, state_dict_cast_bytes


def side_channel_bytes(fp8_layers: Mapping[str, Fp8ScaledLayer]) -> int:
    """The recovered scales, resident from extraction until the split consumes them.

    `extract_fp8_scaled_layers` pops each scale key out of the state dict, so from that point on no
    prediction over the dict can see them. Small for a per-tensor scale, and not small for an MXFP8
    grid -- 1.455 GiB across 256 layers on the released 12.6 GiB Krea-2 build.

    `input_scale` is `None` where the producer shipped none, or where `_usable_input_scale` rejected an
    uncalibrated placeholder; that is the only absence counted as nothing. Anything else lacking the
    tensor interface raises rather than being charged zero, because a silent zero here is the failure
    this module exists to remove.
    """
    total = 0
    for layer in fp8_layers.values():
        for scale in (layer.weight_scale, layer.input_scale):
            if scale is not None:
                total += scale.nelement() * scale.element_size()
    return total


def _reserve(reserve: Callable[[int], Any], needed: int, host_bytes_needed: int) -> Any:
    """Pass incremental host demand while preserving one-argument reserve callbacks."""
    try:
        parameters = inspect.signature(reserve).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters) or any(
        parameter.name == "host_bytes_needed" for parameter in parameters
    ):
        return reserve(needed, host_bytes_needed=host_bytes_needed)  # type: ignore[call-arg]
    return reserve(needed)


def reserve_for_load(
    reserve: Callable[[int], Any],
    state_dict: Mapping[str, Any],
    dtype: torch.dtype,
    *,
    keep_fp8: bool,
    model: torch.nn.Module | None,
    fp8_layers: Mapping[str, Fp8ScaledLayer],
    nvfp4_payloads: Mapping[str, NVFP4Payload],
    skip_patterns: Iterable[str] = (),
    source_tensors_retained: bool = False,
    source_tensors_mapped: bool | Collection[int] | None = False,
    source_tensors_cloned: bool = False,
    modelopt_nvfp4_relayout: bool = False,
) -> int:
    """Ask for room for the state dict, the payloads it will install, and the scales it took out.

    Both side-channel mappings are required rather than defaulted, and so is `model`. A seam that
    recovered nothing passes `{}`, and one with no model yet passes `None`, and both say so; a default
    would let a seam that *did* have something omit it by silence and under-reserve. That matters for
    `model` too: with none, `_is_fp8_matmul_weight` answers for the safe subset, so under `keep_fp8`
    every 2-D weight predicts as staying quantized at 1 B/element and a quantized non-Linear then
    arrives at `dtype.itemsize`.

    This is weaker than making the ordering unrepresentable -- nothing here can stop a caller
    reserving before it has recovered -- and it is what a function taking the results can honestly
    give. `skip_patterns` keeps its default: omitting it is legible beside the `cast_state_dict` that
    omits it too, which is the only seam that does.

    Returns the bytes asked for, so a caller can log or assert on them.
    """
    patterns = tuple(skip_patterns)
    nvfp4_bytes = 0
    if nvfp4_payloads:
        if model is None:
            # `predict_nvfp4_install_size` needs the model to decide which payloads stay packed.
            # Absorbing that into a zero would drop a GiB-scale term without a word, which is this
            # module's subject.
            raise ValueError(
                f"Cannot size {len(nvfp4_payloads)} nvfp4 payload(s) without the model they are "
                "installed into; reserve after the model is built, or pass no payloads."
            )
        nvfp4_bytes = predict_nvfp4_install_size(model, nvfp4_payloads, dtype, patterns)

    needed = (
        predict_cast_state_dict_size(
            state_dict,
            dtype,
            keep_fp8=keep_fp8,
            model=model,
            skip_patterns=patterns,
            scaled_layers=fp8_layers,
        )
        + nvfp4_bytes
        + side_channel_bytes(fp8_layers)
    )
    # `dequantize_fp8_scaled` consumes this same mapping in insertion order. The state dict's key order is
    # independent (and often differs after extraction), so use the mapping order for the cumulative fold peak.
    fold_sd = {}
    for path, layer in fp8_layers.items():
        key = f"{path}.weight"
        tensor = state_dict.get(key)
        if (
            tensor is not None
            and is_castable_float(tensor)
            and not (keep_fp8 and survives_split_and_cast(key, tensor, model, patterns, layer))
        ):
            fold_sd[key] = tensor
    fold_delta = 0
    fold_output_total = 0
    fold_peak = 0
    fold_source_counts: dict[int, int] = {}
    fold_source_bytes: dict[int, int] = {}
    fold_source_mapped: dict[int, bool] = {}
    for tensor in state_dict.values():
        if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
            continue
        storage = tensor.untyped_storage()
        storage_id = storage.data_ptr()
        fold_source_counts[storage_id] = fold_source_counts.get(storage_id, 0) + 1
        fold_source_bytes[storage_id] = storage.nbytes()
        fold_source_mapped[storage_id] = _source_is_mapped(tensor, source_tensors_mapped)
    for key, tensor in fold_sd.items():
        layer = scaled_layer_for(key, fp8_layers)
        if layer is None:
            continue
        output_bytes = tensor.nelement() * dtype.itemsize
        fold_peak = max(
            fold_peak,
            fold_delta + output_bytes + fold_transient_bytes(tensor, layer.weight_scale),
        )
        fold_delta += output_bytes
        fold_output_total += output_bytes
        # Fused checkpoints can expose several views of one source storage; release it only after the last view folds.
        storage_id = tensor.untyped_storage().data_ptr()
        fold_source_counts[storage_id] -= 1
        if fold_source_counts[storage_id] == 0 and not fold_source_mapped[storage_id]:
            fold_delta -= fold_source_bytes[storage_id]
    cast_sd = {
        key: tensor
        for key, tensor in state_dict.items()
        if is_castable_float(tensor)
        and not (keep_fp8 and survives_split_and_cast(key, tensor, model, patterns, scaled_layer_for(key, fp8_layers)))
        and key not in fold_sd
    }
    cast_target_bytes, cast_peak = state_dict_cast_bytes(
        cast_sd,
        dtype,
        in_place=True,
        source_tensors_mapped=True if source_tensors_retained else source_tensors_mapped,
    )
    source_bytes = sum(
        tensor.nelement() * tensor.element_size()
        for key, tensor in state_dict.items()
        if not tensor.is_meta
        and (
            source_tensors_retained
            or not _source_is_mapped(tensor, source_tensors_mapped)
            or (key not in fold_sd and key not in cast_sd)
            or getattr(tensor, "dtype", None) == dtype
        )
    )
    source_bytes += sum(
        scale.nelement() * scale.element_size()
        for layer in fp8_layers.values()
        for scale in (layer.weight_scale, layer.input_scale)
        if scale is not None and (source_tensors_retained or not _source_is_mapped(scale, source_tensors_mapped))
    )
    source_bytes += sum(
        payload.nbytes()
        for payload in nvfp4_payloads.values()
        if source_tensors_retained or not _source_is_mapped(payload.weight, source_tensors_mapped)
    )
    mapped_nvfp4_source_bytes = sum(
        payload.nbytes()
        for payload in nvfp4_payloads.values()
        if not source_tensors_retained and _source_is_mapped(payload.weight, source_tensors_mapped)
    )
    cast_source_bytes = sum(tensor.nelement() * tensor.element_size() for tensor in cast_sd.values())
    cast_delta = cast_target_bytes - cast_source_bytes
    # A mapped source at the destination dtype is reused by `cast_state_dict`; only differing dtypes allocate.
    mapped_cast_bytes = sum(
        tensor.nelement() * dtype.itemsize
        for tensor in cast_sd.values()
        if is_castable_float(tensor)
        and getattr(tensor, "dtype", None) != dtype
        and _source_is_mapped(tensor, source_tensors_mapped)
    )
    cloned_mapped_bytes = (
        sum(
            tensor.nelement() * tensor.element_size()
            for key, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor)
            and not tensor.is_meta
            and key not in fold_sd
            and (not is_castable_float(tensor) or tensor.dtype == dtype)
        )
        if source_tensors_cloned
        else 0
    )
    released_side_channel_bytes = 0
    for path, layer in fp8_layers.items():
        weight = state_dict.get(f"{path}.weight")
        if weight is None:
            continue
        if not keep_fp8 or not survives_split_and_cast(f"{path}.weight", weight, model, patterns, layer):
            released_side_channel_bytes += sum(
                scale.nelement() * scale.element_size()
                for scale in (layer.weight_scale, layer.input_scale)
                if scale is not None and not _source_is_mapped(scale, source_tensors_mapped)
            )
    nvfp4_host_bytes = (
        predict_nvfp4_install_host_size(
            model,
            nvfp4_payloads,
            dtype,
            patterns,
            payloads_relaid_out=modelopt_nvfp4_relayout and bool(source_tensors_mapped),
        )
        if model is not None
        else 0
    )
    mapped_sources = bool(source_tensors_mapped)
    all_sources_mapped = mapped_sources and all(
        _source_is_mapped(tensor, source_tensors_mapped)
        for tensor in state_dict.values()
        if isinstance(tensor, torch.Tensor) and not tensor.is_meta
    )
    if all_sources_mapped:
        cast_phase_peak = max(fold_peak, fold_output_total + mapped_cast_bytes + cloned_mapped_bytes)
        cast_phase_delta = fold_output_total + mapped_cast_bytes + cloned_mapped_bytes
    elif mapped_sources or source_tensors_retained:
        cast_delta = sum(
            tensor.nelement() * dtype.itemsize - tensor.nelement() * tensor.element_size()
            for tensor in cast_sd.values()
            if is_castable_float(tensor)
            and getattr(tensor, "dtype", None) != dtype
            and not _source_is_mapped(tensor, source_tensors_mapped)
        )
        cast_phase_peak = max(fold_peak, fold_delta + cast_peak)
        cast_phase_delta = fold_delta + cast_delta + mapped_cast_bytes - released_side_channel_bytes
    else:
        cast_phase_peak = fold_delta + cast_peak
        cast_phase_delta = fold_delta + cast_delta - released_side_channel_bytes
    later_install_peak = max(
        nvfp4_host_bytes,
        predict_modelopt_nvfp4_relayout_host_size(nvfp4_payloads, source_tensors_mapped=source_tensors_mapped)
        if modelopt_nvfp4_relayout
        else 0,
    )
    host_bytes_needed = max(
        0,
        needed - source_bytes - mapped_nvfp4_source_bytes,
        fold_peak,
        cast_phase_peak,
        cast_phase_delta + later_install_peak,
    )
    if source_tensors_retained and not bool(source_tensors_mapped):
        # Pread-backed source tensors stay live while the model receives its assigned tensors. The final state dict
        # therefore has to coexist with the retained source, including values already at the destination dtype.
        retained_output_bytes = sum(
            tensor.nelement() * (dtype.itemsize if is_castable_float(tensor) else tensor.element_size())
            for tensor in state_dict.values()
            if isinstance(tensor, torch.Tensor) and not tensor.is_meta
        )
        host_bytes_needed = max(host_bytes_needed, retained_output_bytes)
    require_ram_headroom(_reserve(reserve, needed, host_bytes_needed), host_bytes_needed)
    return needed
