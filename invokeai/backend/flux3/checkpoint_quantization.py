"""Which quantized FLUX.3 single files load, decided once for identification and for the loader.

Identification sees a header -- the tensor names, dtypes and shapes on the meta device, the safetensors metadata, and
the ``.comfy_quant`` markers read from their bytes -- while the loader sees the tensors. Both hand this module the same
picture and get the same answer, so a file installs exactly when the loader can build it, and one it cannot is refused
at install with the reason rather than failing at the first render.

The builds that load are Black Forest Labs' bf16 file with its ``variants/fp8r`` (fp8 with one scale per row) and
``variants/nvfp4`` (NVIDIA ModelOpt's layout), and Comfy-Org's bf16, ``int8_convrot`` and nvfp4 repackages. None of
them mixes two schemes. A file mixing int8 with nvfp4 or with scaled fp8 is refused rather than supported on
speculation; nvfp4 beside scaled fp8 loads, since the loader takes each of them its own way.
"""

from collections.abc import Collection, Mapping
from typing import Any

import torch

from invokeai.backend.quantization.fp8_scaled import (
    FP8_WEIGHT_DTYPES,
    iter_weight_scale_pairs,
    parse_quantization_metadata,
    strip_layer_path_prefix,
    unparsed_fp8_scale_keys,
)
from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    INT8_TENSORWISE_FORMAT,
    check_hadamard_size,
    reject_unmarked_int8_weights,
)
from invokeai.backend.quantization.nvfp4 import find_nvfp4_layers, modelopt_nvfp4_header_layers

_INSTEAD = (
    "Install the bf16 file, Comfy-Org's int8 or nvfp4 build, Black Forest Labs' fp8r or nvfp4 variant, or a GGUF "
    "instead."
)

# The weights `flux3_transformer_config` reads the model's input widths from. Packed two codes to a byte, an nvfp4
# weight stores its width halved, and the transformer would be built for that.
_INPUT_WIDTH_WEIGHTS = ("emb_in.image.weight", "emb_in.image_cond.weight", "txt_in.weight", "time_in.in_layer.weight")

# bitsandbytes keeps a 4-bit layer's quantization state beside its packed `uint8` weight.
_BITSANDBYTES_STATE = (".absmax", ".quant_map", ".quant_state")

_TORCHAO_PAYLOAD_SUFFIX = "._weight_qdata"


def flux3_header_layers(metadata: Mapping[str, Any] | None, keys: Collection[Any]) -> dict[str, Any]:
    """The layers a FLUX.3 file's header names, under the prefix-stripped names in ``keys``: ComfyUI's
    ``_quantization_metadata`` entries, and for NVIDIA ModelOpt's nvfp4 export, Black Forest Labs', every packed layer,
    since its metadata names the scheme once for all of them."""
    return {
        **strip_layer_path_prefix(parse_quantization_metadata(metadata)),
        **modelopt_nvfp4_header_layers(metadata, keys),
    }


def reject_unloadable_quantization(
    state_dict: dict[str, Any], header_layers: Mapping[str, Any], markers: Mapping[str, Mapping[str, Any]]
) -> None:
    """Refuse a FLUX.3 single file whose quantization the loader cannot build, naming the reason (ValueError).

    ``state_dict`` holds the file's tensors under their prefix-stripped native names; meta tensors will do, since names,
    dtypes and shapes are all that is read. ``header_layers`` is :func:`flux3_header_layers` of it, and ``markers`` are
    its ``.comfy_quant`` markers, decoded, by layer.

    Each refusal is of a file that would otherwise load and generate noise, or fail mid-render:

    - torchao's tensor subclasses, which leave no ``.weight`` to load;
    - bitsandbytes 4-bit state, and any packed ``uint8`` weight that is not a named nvfp4 layer;
    - an fp8 weight scale in a spelling the key conversion does not carry, which would be left behind as its weight is
      renamed, and the weight loaded unscaled (AWQ's ``pre_quant_scale`` beside an fp8 weight among them);
    - int8 weights without an ``int8_tensorwise`` marker, or rotated in groups that are not a power of 4;
    - what :func:`find_nvfp4_layers` refuses: AWQ, nvfp4 layers no header or marker names, malformed or half-stored
      ones;
    - int8 mixed with nvfp4 or with scaled fp8;
    - packed input embedders, whose stored width is not the model's.
    """
    keys = list(state_dict)
    torchao = sorted(key for key in keys if key.endswith(_TORCHAO_PAYLOAD_SUFFIX))
    if torchao:
        raise ValueError(
            f"This FLUX.3 file is saved as torchao-quantized tensors (e.g. '{torchao[0]}'), which is not supported. "
            f"{_INSTEAD}"
        )
    bitsandbytes = sorted(key for key in keys if any(state in key for state in _BITSANDBYTES_STATE))
    if bitsandbytes:
        raise ValueError(
            f"This FLUX.3 file carries bitsandbytes 4-bit quantization state (e.g. '{bitsandbytes[0]}'), which is not "
            f"supported. {_INSTEAD}"
        )
    unparsed = unparsed_fp8_scale_keys(state_dict)
    if unparsed:
        raise ValueError(
            f"{len(unparsed)} fp8 weight scale(s) of this FLUX.3 file use a spelling InvokeAI does not read (e.g. "
            f"'{unparsed[0]}'), so their weights would load unscaled. {_INSTEAD}"
        )

    int8_markers = {path: marker for path, marker in markers.items() if marker.get("format") == INT8_TENSORWISE_FORMAT}
    reject_unmarked_int8_weights(state_dict, int8_markers, "FLUX.3")
    for path, marker in int8_markers.items():
        group_size = marker.get("convrot_groupsize", CONVROT_GROUP_SIZE)
        try:
            if marker.get("convrot"):
                check_hadamard_size(int(group_size))
        except (TypeError, ValueError):
            raise ValueError(
                f"int8 layer '{path}' is rotated in groups of {group_size!r}; a rotation is undone only in groups of a "
                f"power of 4 (256 in every release). {_INSTEAD}"
            ) from None

    nvfp4 = set(find_nvfp4_layers(state_dict, header_layers, markers=markers))
    packed = sorted(
        key
        for key in keys
        if key.endswith(".weight")
        and getattr(state_dict[key], "dtype", None) is torch.uint8
        and key[: -len(".weight")] not in nvfp4
    )
    if packed:
        raise ValueError(
            f"{len(packed)} weight(s) of this FLUX.3 file are packed into bytes without an nvfp4 global scale (e.g. "
            f"'{packed[0]}'), a packing InvokeAI does not read, such as bitsandbytes NF4. {_INSTEAD}"
        )

    if int8_markers and nvfp4:
        raise ValueError(f"This FLUX.3 file mixes int8 and nvfp4 layers, which no release does. {_INSTEAD}")
    if int8_markers:
        scaled_fp8 = sorted(
            weight_key
            for weight_key, _scale_key in iter_weight_scale_pairs(state_dict)
            if getattr(state_dict[weight_key], "dtype", None) in FP8_WEIGHT_DTYPES
        )
        if scaled_fp8:
            raise ValueError(
                f"This FLUX.3 file mixes int8 layers with scaled fp8 ones (e.g. '{scaled_fp8[0]}'), which no release "
                f"does. {_INSTEAD}"
            )

    packed_inputs = [key for key in _INPUT_WIDTH_WEIGHTS if key[: -len(".weight")] in nvfp4]
    if packed_inputs:
        raise ValueError(
            f"This FLUX.3 file stores '{packed_inputs[0]}' packed, so its shape is not the model's input width. No "
            f"release quantizes the input embedders. {_INSTEAD}"
        )
