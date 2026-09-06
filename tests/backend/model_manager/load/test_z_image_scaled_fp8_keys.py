"""The scaled-fp8 Z-Image key layout must be recognized, scales and all.

A ComfyUI "scaled fp8" checkpoint stores each quantized Linear as an fp8 `<name>.weight` plus a
`<name>.scale_weight`. Before the scaled-fp8 handling landed, the Z-Image loader deleted those
scale keys in its `keys_to_remove` filter and cast the weight — so every quantized weight was off
by a factor of `weight_scale`, with nothing logged. On the captured checkpoint the scales run from
1.5 to 7.6, which leaves each weight at a *different* fraction of its true magnitude.

The fixture is a real key layout; the tensors are synthetic because only shapes and dtypes matter
to the code under test.
"""

import torch

from invokeai.backend.quantization.fp8_scaled import (
    FP8_DTYPE,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
)
from tests.backend.model_manager.load.state_dicts.z_image_transformer_scaled_fp8_keys import (
    state_dict_keys as scaled_keys,
)

_DTYPES = {"F8_E4M3": FP8_DTYPE, "F32": torch.float32, "BF16": torch.bfloat16}


def _mock_state_dict(scale_value: float = 4.0) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in scaled_keys.items():
        if key.endswith(".scale_weight"):
            sd[key] = torch.full(shape, scale_value, dtype=torch.float32)
        else:
            sd[key] = torch.ones(shape, dtype=torch.float32).to(_DTYPES[dtype])
    return sd


def test_the_fixture_really_is_scaled_fp8() -> None:
    """Guard the fixture itself: a bf16 recapture would make every assertion below vacuous."""
    assert any(k.endswith(".scale_weight") for k in scaled_keys)
    assert not any(k.endswith(".weight_scale") for k in scaled_keys)
    assert any(dtype == "F8_E4M3" for _, dtype in scaled_keys.values())
    assert "scaled_fp8" in scaled_keys


def test_every_scale_weight_is_recognized_as_a_scaled_layer() -> None:
    sd = _mock_state_dict()
    expected = {k[: -len(".scale_weight")] for k in scaled_keys if k.endswith(".scale_weight")}

    layers = extract_fp8_scaled_layers(sd, layer_hints=extract_comfy_quant_hints(sd))

    assert set(layers) == expected, "a .scale_weight spelling was not recognized"


def test_extraction_consumes_the_scale_keys() -> None:
    # Anything left behind reaches `load_state_dict(..., strict=True)` as an unexpected key.
    sd = _mock_state_dict()

    extract_fp8_scaled_layers(sd, layer_hints=extract_comfy_quant_hints(sd))

    assert not [k for k in sd if k.endswith((".scale_weight", ".weight_scale"))]


def test_the_scale_is_carried_through_not_discarded() -> None:
    sd = _mock_state_dict(scale_value=4.0)

    layers = extract_fp8_scaled_layers(sd, layer_hints=extract_comfy_quant_hints(sd))

    assert layers
    for layer in layers.values():
        assert layer.weight_scale.float().reshape(-1)[0].item() == 4.0
