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

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from invokeai.backend.quantization.fp8_scaled import Fp8ScaledLayer, predict_cast_state_dict_size
from invokeai.backend.quantization.nvfp4 import NVFP4Payload, predict_nvfp4_install_size


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


def reserve_for_load(
    reserve: Callable[[int], None],
    state_dict: Mapping[str, Any],
    dtype: torch.dtype,
    *,
    keep_fp8: bool,
    model: torch.nn.Module | None,
    fp8_layers: Mapping[str, Fp8ScaledLayer],
    nvfp4_payloads: Mapping[str, NVFP4Payload],
    skip_patterns: Iterable[str] = (),
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
    reserve(needed)
    return needed
