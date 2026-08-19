"""A scaled-fp8 Anima checkpoint must load, and its layer flags must survive the prefix strip.

Two separate failures are covered here. Before scaled-fp8 support reached `AnimaCheckpointModel`,
`_filter_non_model_keys` let the scale and marker keys through and the loader raised on them --
500 unexpected keys on a plain scaled export, 749 on one that also ships `comfy_quant` markers, so
such a checkpoint did not load at all.

The second is quieter: `_quantization_metadata` names its layers `net.`-prefixed, in the
checkpoint's own scheme, while the scales are read after the prefix has been stripped. Reading the
header without renaming matches nothing, and every `full_precision_matrix_mult` flag is dropped
without a word.
"""

import json

import torch

from invokeai.backend.model_manager.load.model_loaders.anima import (
    _filter_non_model_keys,
    _strip_anima_bundle_prefix,
    _strip_anima_prefix_from_layer_paths,
)
from invokeai.backend.quantization.fp8_scaled import (
    FP8_DTYPE,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    is_scale_metadata_key,
)
from tests.backend.model_manager.load.state_dicts.anima_transformer_scaled_fp8_keys import (
    layer_hints as header_hints,
)
from tests.backend.model_manager.load.state_dicts.anima_transformer_scaled_fp8_keys import (
    state_dict_keys as anima_keys,
)

_DTYPES = {"F8_E4M3": FP8_DTYPE, "F32": torch.float32, "BF16": torch.bfloat16, "U8": torch.uint8}


def _build_state_dict() -> dict[str, torch.Tensor]:
    """Rebuild the checkpoint. `comfy_quant` markers are real JSON blobs, not placeholders.

    The marker is the only transport some checkpoints have, so a test that fakes it would not
    exercise the path that reads it.
    """
    sd: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in anima_keys.items():
        torch_dtype = _DTYPES[dtype]
        if key.endswith(".comfy_quant"):
            path = key[: -len(".comfy_quant")]
            blob = header_hints.get(path, {"format": "float8_e4m3fn"})
            # Real JSON: the producer writes `true`/`false`, and `extract_comfy_quant_hints`
            # parses the blob. Python's `False` would not parse and the flag would vanish.
            sd[key] = torch.frombuffer(bytearray(json.dumps(blob).encode()), dtype=torch.uint8).clone()
        elif torch_dtype is FP8_DTYPE:
            sd[key] = torch.zeros(shape, dtype=torch.float32).to(FP8_DTYPE)
        elif key.endswith((".weight_scale", ".input_scale")):
            # 1.0 is the placeholder `_usable_input_scale` rejects, so it must not be used here.
            sd[key] = torch.full(shape, 2.5, dtype=torch_dtype)
        else:
            sd[key] = torch.zeros(shape, dtype=torch_dtype)
    return _filter_non_model_keys(_strip_anima_bundle_prefix(sd))


def test_nothing_the_loader_would_reject_is_left_behind() -> None:
    """`load_state_dict` must see only model tensors; the loader raises on anything else."""
    sd = _build_state_dict()
    assert [k for k in sd if is_scale_metadata_key(k)], "fixture carries no side-channel keys"

    extract_fp8_scaled_layers(sd)

    assert [k for k in sd if is_scale_metadata_key(k)] == []


def test_every_quantized_linear_is_recognized() -> None:
    sd = _build_state_dict()
    fp8_weights = {k for k, v in sd.items() if v.dtype is FP8_DTYPE and k.endswith(".weight")}
    assert fp8_weights

    layers = extract_fp8_scaled_layers(sd, layer_hints=extract_comfy_quant_hints(sd))

    assert {f"{path}.weight" for path in layers} == fp8_weights


def test_header_hints_are_renamed_to_the_stripped_paths() -> None:
    """The header names layers `net.`-prefixed; the scales are keyed on the stripped paths.

    Without the rename the flags match nothing. This is the mistake that already cost a debugging
    round on Krea-2, where the metadata was read against the wrong naming.
    """
    assert all(name.startswith("net.") for name in header_hints), "fixture header is not prefixed"

    mapping = _strip_anima_prefix_from_layer_paths(list(header_hints))

    assert all(not renamed.startswith("net.") for renamed in mapping.values())
    remapped = {mapping[name]: hints for name, hints in header_hints.items()}

    layers = extract_fp8_scaled_layers(_build_state_dict(), layer_hints=remapped)

    assert layers
    assert set(remapped) >= set(layers), "a renamed hint no longer lines up with its layer"


def test_the_single_full_precision_flag_survives_both_transports() -> None:
    """Exactly one layer is marked. It must come through whichever transport is read.

    Every other captured checkpoint marks either no layers or a large fraction, so a bug that
    dropped a lone flag would go unnoticed there.
    """
    marked = [name for name, hints in header_hints.items() if hints.get("full_precision_matrix_mult")]
    assert len(marked) == 1, f"fixture should carry exactly one marked layer, has {len(marked)}"

    mapping = _strip_anima_prefix_from_layer_paths(list(header_hints))
    for hints in (
        {mapping[n]: h for n, h in header_hints.items()},  # header transport
        extract_comfy_quant_hints(_build_state_dict()),  # per-layer marker transport
    ):
        layers = extract_fp8_scaled_layers(_build_state_dict(), layer_hints=hints)
        assert sum(1 for layer in layers.values() if layer.full_precision_matmul) == 1


def test_only_the_full_precision_layer_lacks_an_input_scale() -> None:
    """Every quantized Linear ships a calibrated `scale_input` -- except the marked one.

    That is coherent on the producer's side: a layer excluded from the fp8 matmul has no activation
    scale to calibrate. It also means "all layers have an input scale" is the wrong invariant.
    """
    mapping = _strip_anima_prefix_from_layer_paths(list(header_hints))
    marked = {mapping[n] for n, h in header_hints.items() if h.get("full_precision_matrix_mult")}

    layers = extract_fp8_scaled_layers(_build_state_dict())

    assert layers
    without = {path for path, layer in layers.items() if layer.input_scale is None}
    assert without == marked
