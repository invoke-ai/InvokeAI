"""What the community `int8_tensorwise` Wan 2.2 T2V-14B build actually contains.

`test_wan_checkpoint_loader.py` runs on synthetic tiny geometry, which can say whether the loader handles the
shape it was given and cannot say whether the released file still has that shape. These cells read the header
captured in `state_dicts/wan_int8_convrot_keys.py` and pin what the decode depends on: the marker declares the
rotation, the scales and widths fit it, and every native key lands on a parameter of the released architecture
with the quantized layers on Linears the model does not declare precision-sensitive.
"""

import accelerate
import pytest
import torch
from diffusers import WanTransformer3DModel

from invokeai.backend.model_manager.load.load_default import _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loaders.wan import (
    _convert_wan_native_to_diffusers,
    _rename_wan_native_key,
)
from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    INT8_TENSORWISE_FORMAT,
    marker_from_header_entry,
    resolve_quantized_module_paths,
)
from tests.backend.model_manager.load.state_dicts import wan_int8_convrot_keys as fixture

KEYS = fixture.state_dict_keys
QUANTIZED = sorted(key[: -len(".weight")] for key, (_shape, dtype) in KEYS.items() if dtype == "I8")


@pytest.fixture(scope="module")
def model() -> WanTransformer3DModel:
    """The released T2V-A14B geometry, one block deep, on the meta device."""
    with accelerate.init_empty_weights():
        return WanTransformer3DModel(
            patch_size=(1, 2, 2),
            in_channels=16,
            out_channels=16,
            num_layers=1,
            attention_head_dim=128,
            num_attention_heads=40,
            ffn_dim=13824,
            text_dim=4096,
        )


def test_the_marker_is_rotated_and_the_header_entry_decodes_to_the_same_layer() -> None:
    """`convrot` and its group size are what the decode cannot guess: a build that dropped them would still
    declare `int8_tensorwise`, still load, and derotate nothing."""
    for path, marker in fixture.markers.items():
        assert marker["format"] == INT8_TENSORWISE_FORMAT
        assert marker["convrot"] is True
        assert marker["convrot_groupsize"] == CONVROT_GROUP_SIZE
        entry = marker_from_header_entry(path, fixture.header_layers[path], {INT8_TENSORWISE_FORMAT})
        assert entry is not None
        assert entry["convrot"] is True
        assert entry["convrot_groupsize"] == CONVROT_GROUP_SIZE


def test_the_scales_are_per_output_channel_and_every_width_is_whole_rotation_groups() -> None:
    for path in QUANTIZED:
        weight_shape, _ = KEYS[f"{path}.weight"]
        scale_shape, scale_dtype = KEYS[f"{path}.weight_scale"]
        assert scale_shape == [weight_shape[0], 1], path
        assert scale_dtype == "F32", path
        assert weight_shape[1] % CONVROT_GROUP_SIZE == 0, path


def test_every_key_lands_on_a_parameter_of_the_released_architecture(model: WanTransformer3DModel) -> None:
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    converted = _convert_wan_native_to_diffusers(
        {key: torch.empty(shape, device="meta") for key, (shape, _) in KEYS.items()}
    )
    loadable = {
        key: tuple(value.shape)
        for key, value in converted.items()
        if not key.endswith((".comfy_quant", ".weight_scale"))
    }

    assert loadable == expected


def test_the_quantized_layers_resolve_to_linears_the_model_does_not_declare_sensitive(
    model: WanTransformer3DModel,
) -> None:
    key_map = {key: _rename_wan_native_key(key) for key in KEYS}
    resolved = resolve_quantized_module_paths(fixture.markers, key_map)
    skip = _model_declared_skip_patterns(model)

    assert len(resolved) == len(QUANTIZED)
    for path in resolved:
        assert isinstance(model.get_submodule(path), torch.nn.Linear), path
        assert not any(pattern in path for pattern in skip), path
