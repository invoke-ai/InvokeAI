"""ERNIE-Image LoRA conversion: every published layer has to land on a module of the real transformer.

A layer key that resolves to nothing is not an error at runtime -- the patcher logs it and skips it, and the LoRA
quietly does less than it should. So each published layout is converted and resolved against
`ErnieImageTransformer2DModel` at the released geometry (on the meta device: shapes only, no weights).
"""

import re

import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.ernie_image import ERNIE_IMAGE_TRANSFORMER_CONFIG
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.ernie_image_lora_constants import (
    ERNIE_IMAGE_LORA_KEY_PREFIXES,
    ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.ernie_image_lora_conversion_utils import (
    lora_model_from_ernie_image_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    ernie_image_lora_aitoolkit_format,
    ernie_image_lora_peft_format,
)


def _state_dict(keys: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    return {key: torch.zeros((), dtype=torch.float32).expand(shape) for key, shape in keys.items()}


@pytest.fixture(scope="module")
def released_transformer() -> torch.nn.Module:
    from diffusers import ErnieImageTransformer2DModel

    with torch.device("meta"):
        return ErnieImageTransformer2DModel(**ERNIE_IMAGE_TRANSFORMER_CONFIG)


def _reprefixed(keys: dict[str, list[int]], prefix: str) -> dict[str, list[int]]:
    return {prefix + key.removeprefix("diffusion_model."): shape for key, shape in keys.items()}


def _lycoris(keys: dict[str, list[int]], kind: str) -> dict[str, list[int]]:
    """The ai-toolkit layout's modules as LoKR or LoHA layers, factor shapes derived from each module's shape."""
    shapes = {
        key.removesuffix(".lora_A.weight"): (keys[key.replace("lora_A", "lora_B")][0], shape[1])
        for key, shape in keys.items()
        if key.endswith(".lora_A.weight")
    }
    out: dict[str, list[int]] = {}
    for module, (rows, cols) in shapes.items():
        if kind == "lokr":
            out[f"{module}.lokr_w1"] = [8, 8]
            out[f"{module}.lokr_w2_a"] = [rows // 8, 4]
            out[f"{module}.lokr_w2_b"] = [4, cols // 8]
        else:
            out[f"{module}.hada_w1_a"], out[f"{module}.hada_w1_b"] = [rows, 4], [4, cols]
            out[f"{module}.hada_w2_a"], out[f"{module}.hada_w2_b"] = [rows, 4], [4, cols]
        out[f"{module}.alpha"] = []
    return out


# Every container prefix the conversion strips: it strips the first one that matches, so their order matters
# -- `base_model.model.` ahead of `base_model.model.transformer.` would leave `transformer.` on every path.
_LAYOUTS = {
    "ai-toolkit": (ernie_image_lora_aitoolkit_format.state_dict_keys, 14),
    # The same blocks plus the six top-level modules.
    "peft-top-level": (ernie_image_lora_peft_format.state_dict_keys, 20),
    **{
        f"prefix={prefix or '<none>'}": (_reprefixed(ernie_image_lora_aitoolkit_format.state_dict_keys, prefix), 14)
        for prefix in (*ERNIE_IMAGE_LORA_KEY_PREFIXES, "")
    },
    "lokr": (_lycoris(ernie_image_lora_aitoolkit_format.state_dict_keys, "lokr"), 14),
    "loha": (_lycoris(ernie_image_lora_aitoolkit_format.state_dict_keys, "loha"), 14),
}


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_every_published_layer_lands_on_a_module_of_the_released_transformer(
    released_transformer: torch.nn.Module, layout: str
) -> None:
    keys, expected_layers = _LAYOUTS[layout]
    patch = lora_model_from_ernie_image_state_dict(_state_dict(keys))

    assert len(patch.layers) == expected_layers
    expected_type = {"lokr": LoKRLayer, "loha": LoHALayer}.get(layout, LoRALayer)
    for key, layer in patch.layers.items():
        assert key.startswith(ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX)
        module = released_transformer.get_submodule(key.removeprefix(ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX))
        assert isinstance(module, torch.nn.Linear), key
        assert isinstance(layer, expected_type), key
        if isinstance(layer, LoRALayer):
            assert layer.down.shape[1] == module.in_features, key
            assert layer.up.shape[0] == module.out_features, key
        else:
            assert layer.get_weight(torch.empty(0)).shape == module.weight.shape, key


def test_alpha_defaults_to_the_rank_and_a_file_alpha_wins() -> None:
    peft = {
        "diffusion_model.layers.0.self_attention.to_q.lora_A.weight": torch.zeros(16, 4096),
        "diffusion_model.layers.0.self_attention.to_q.lora_B.weight": torch.zeros(4096, 16),
    }
    kohya = {
        "diffusion_model.layers.0.self_attention.to_k.lora_down.weight": torch.zeros(16, 4096),
        "diffusion_model.layers.0.self_attention.to_k.lora_up.weight": torch.zeros(4096, 16),
        "diffusion_model.layers.0.self_attention.to_k.alpha": torch.tensor(8.0),
    }

    patch = lora_model_from_ernie_image_state_dict({**peft, **kohya})

    assert patch.layers[f"{ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX}layers.0.self_attention.to_q"].scale() == 1.0
    assert patch.layers[f"{ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX}layers.0.self_attention.to_k"].scale() == 0.5


@pytest.mark.parametrize(
    "extra_key",
    [
        "diffusion_model.layers.0.self_attention.to_q.lora_magnitude_vector.weight",
        "diffusion_model.layers.0.self_attention.to_q.dora_scale",
        "diffusion_model.layers.0.self_attention.to_q.lora_B.bias",
    ],
    ids=["dora-peft", "dora-lycoris", "lora-bias"],
)
def test_tensors_the_conversion_cannot_apply_are_refused(extra_key: str) -> None:
    state_dict = _state_dict(ernie_image_lora_aitoolkit_format.state_dict_keys)
    state_dict[extra_key] = torch.zeros(4096)

    with pytest.raises(ValueError, match=re.escape(extra_key)):
        lora_model_from_ernie_image_state_dict(state_dict)


def test_half_of_a_low_rank_pair_is_named() -> None:
    with pytest.raises(ValueError, match="layers.0.mlp.up_proj"):
        lora_model_from_ernie_image_state_dict(
            {"diffusion_model.layers.0.mlp.up_proj.lora_A.weight": torch.zeros(16, 4096)}
        )


def test_a_layer_on_no_transformer_module_is_refused() -> None:
    """It would be skipped at generation with only a log warning -- the LoRA would silently do less."""
    state_dict = _state_dict(ernie_image_lora_aitoolkit_format.state_dict_keys)
    state_dict["text_encoders.layers.0.mlp.gate_proj.lora_A.weight"] = torch.zeros(16, 4096)
    state_dict["text_encoders.layers.0.mlp.gate_proj.lora_B.weight"] = torch.zeros(12288, 16)

    with pytest.raises(ValueError, match="target no ERNIE-Image transformer module.*text_encoders"):
        lora_model_from_ernie_image_state_dict(state_dict)
