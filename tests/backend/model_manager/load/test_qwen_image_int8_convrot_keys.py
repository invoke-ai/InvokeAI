"""What the released `int8_tensorwise` Qwen-Image Edit 2511 transformer and Qwen2.5-VL encoder contain.

The loader suites run on toy geometry, which says whether the loader handles the shape it was given and not whether
the released file still has that shape. These cells read the captured layouts, hold them against the real module
trees (built on the meta device, so no weights exist), and pin the properties the decode depends on.
"""

import accelerate
import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.qwen_image import _remap_qwen_vl_checkpoint_keys
from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX
from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    INT8_TENSORWISE_FORMAT,
)
from invokeai.backend.qwen2_5_vl.qwen2_5_vl_assets import load_bundled_qwen2_5_vl_config_dict
from tests.backend.model_manager.load.state_dicts import qwen_image_int8_convrot_keys as transformer_fixture
from tests.backend.model_manager.load.state_dicts import qwen_vl_encoder_int8_convrot_keys as encoder_fixture

ROTATED = {"format": INT8_TENSORWISE_FORMAT, "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}


def _quantized(keys: dict) -> list[str]:
    return sorted(key[: -len(".weight")] for key, (_shape, dtype) in keys.items() if dtype == "I8")


def _transformer_module_shapes() -> dict[str, torch.Size]:
    from diffusers import QwenImageTransformer2DModel

    with accelerate.init_empty_weights():
        model = QwenImageTransformer2DModel(num_layers=1)
    return {name: module.weight.shape for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}


def _encoder_module_shapes() -> dict[str, torch.Size]:
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    with accelerate.init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig.from_dict(load_bundled_qwen2_5_vl_config_dict()))
    return {name: module.weight.shape for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}


CASES = {
    "transformer": (transformer_fixture, lambda path: path, _transformer_module_shapes),
    # The encoder's layers are named in the legacy layout; the loader remaps them before the swap.
    "encoder": (
        encoder_fixture,
        lambda path: next(iter(_remap_qwen_vl_checkpoint_keys({path: 0}))),
        _encoder_module_shapes,
    ),
}


@pytest.fixture(params=sorted(CASES))
def case(request):
    fixture, rename, module_shapes = CASES[request.param]
    return fixture, rename, module_shapes()


def test_every_int8_layer_is_marked_rotated_and_scaled_per_output_channel(case) -> None:
    fixture, _rename, _shapes = case
    keys = fixture.state_dict_keys
    quantized = _quantized(keys)

    assert quantized
    assert set(fixture.markers) == set(quantized)
    assert all(marker == ROTATED for marker in fixture.markers.values())
    assert {k[: -len(COMFY_QUANT_SUFFIX)] for k in keys if k.endswith(COMFY_QUANT_SUFFIX)} == set(quantized)
    for path in quantized:
        weight_shape, _ = keys[f"{path}.weight"]
        scale_shape, scale_dtype = keys[f"{path}.weight_scale"]
        # `[out, 1]` and not `[out]`: a flat scale would broadcast along the input axis of a square weight.
        assert scale_shape == [weight_shape[0], 1] and scale_dtype == "F32", path
        assert weight_shape[1] % CONVROT_GROUP_SIZE == 0, path


def test_the_captured_layers_are_the_linears_the_real_model_has(case) -> None:
    """A swap needs a `Linear` of the stored shape at the path the (remapped) checkpoint names."""
    fixture, rename, linears = case
    keys = fixture.state_dict_keys

    for path in _quantized(keys):
        assert linears[rename(path)] == torch.Size(keys[f"{path}.weight"][0]), path
