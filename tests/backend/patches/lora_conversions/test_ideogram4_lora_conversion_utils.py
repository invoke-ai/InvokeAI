"""Ideogram 4 LoRA conversion: every published layer has to land on a module of the vendored transformer.

A layer key that resolves to nothing is not an error at runtime -- the patcher logs it and skips it, and the LoRA
quietly does less than it should. So each published layout is converted and resolved against `Ideogram4Transformer`
at the released geometry (on the meta device: shapes only, no weights). The diffusers layout is the one that needs
real work -- its split q/k/v are fused into the vendored `attention.qkv` -- so that fusion is checked for exactness,
and against diffusers' own converter in the other direction.
"""

import re

import pytest
import torch

from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.ideogram4_lora_constants import (
    IDEOGRAM4_LORA_KEY_PREFIXES,
    IDEOGRAM4_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.ideogram4_lora_conversion_utils import (
    lora_model_from_ideogram4_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    ideogram4_lokr_aitoolkit_format,
    ideogram4_lora_aitoolkit_format,
    ideogram4_lora_aitoolkit_unconditional_format,
    ideogram4_lora_diffusers_format,
)


def _state_dict(keys: dict[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    return {key: torch.zeros((), dtype=torch.float32).expand(shape) for key, (shape, _dtype) in keys.items()}


def _reprefixed(keys: dict, prefix: str) -> dict:
    return {prefix + key.removeprefix("diffusion_model."): value for key, value in keys.items()}


@pytest.fixture(scope="module")
def released_transformer() -> torch.nn.Module:
    with torch.device("meta"):
        return Ideogram4Transformer(Ideogram4Config())


# Every container prefix the conversion strips: it strips the first one that matches, so their order matters.
_LAYOUTS = {
    # Block 0, six modules: qkv, o, w1-3, adaln_modulation.
    "ai-toolkit": (ideogram4_lora_aitoolkit_format.state_dict_keys, 6, LoRALayer),
    "ai-toolkit-lokr": (ideogram4_lokr_aitoolkit_format.state_dict_keys, 6, LoKRLayer),
    # The same block plus the seven top-level modules.
    "ai-toolkit-unconditional": (ideogram4_lora_aitoolkit_unconditional_format.state_dict_keys, 13, LoRALayer),
    # q/k/v fused into one qkv layer, to_out.0 -> o, w1-3: five modules.
    "diffusers": (ideogram4_lora_diffusers_format.state_dict_keys, 5, LoRALayer),
    **{
        f"prefix={prefix or '<none>'}": (
            _reprefixed(ideogram4_lora_aitoolkit_format.state_dict_keys, prefix),
            6,
            LoRALayer,
        )
        for prefix in (*IDEOGRAM4_LORA_KEY_PREFIXES, "")
    },
}


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_every_published_layer_lands_on_a_module_of_the_released_transformer(
    released_transformer: torch.nn.Module, layout: str
) -> None:
    keys, expected_layers, expected_type = _LAYOUTS[layout]
    patch = lora_model_from_ideogram4_state_dict(_state_dict(keys))

    assert len(patch.layers) == expected_layers
    for key, layer in patch.layers.items():
        assert key.startswith(IDEOGRAM4_LORA_TRANSFORMER_PREFIX)
        module = released_transformer.get_submodule(key.removeprefix(IDEOGRAM4_LORA_TRANSFORMER_PREFIX))
        assert isinstance(module, torch.nn.Linear), key
        assert isinstance(layer, expected_type), key
        if isinstance(layer, LoRALayer):
            assert layer.down.shape[1] == module.in_features, key
            assert layer.up.shape[0] == module.out_features, key
        else:
            assert layer.get_weight(torch.empty(0)).shape == module.weight.shape, key


def _delta(patch, module: str) -> torch.Tensor:
    layer = patch.layers[f"{IDEOGRAM4_LORA_TRANSFORMER_PREFIX}{module}"]
    return layer.get_weight(torch.empty(0)) * layer.scale()


def test_split_qkv_is_fused_into_exactly_the_three_deltas() -> None:
    """Each projection keeps its own rank and alpha; the fused delta is their row-wise concatenation, and a
    projection the file does not train leaves its rows of `qkv` untouched."""
    generator = torch.Generator().manual_seed(0)
    hidden, ranks, alphas = 48, {"to_q": 4, "to_v": 2}, {"to_q": 2.0, "to_v": 8.0}
    state_dict: dict[str, torch.Tensor] = {}
    expected = {"to_q": None, "to_k": torch.zeros(hidden, hidden), "to_v": None}
    for proj, rank in ranks.items():
        down = torch.randn(rank, hidden, generator=generator)
        up = torch.randn(hidden, rank, generator=generator)
        state_dict[f"transformer.layers.0.attention.{proj}.lora_down.weight"] = down
        state_dict[f"transformer.layers.0.attention.{proj}.lora_up.weight"] = up
        state_dict[f"transformer.layers.0.attention.{proj}.alpha"] = torch.tensor(alphas[proj])
        expected[proj] = up @ down * (alphas[proj] / rank)

    fused = _delta(lora_model_from_ideogram4_state_dict(state_dict), "layers.0.attention.qkv")

    torch.testing.assert_close(fused, torch.cat([expected["to_q"], expected["to_k"], expected["to_v"]]))


def test_a_native_lora_and_its_diffusers_conversion_patch_the_same_weights() -> None:
    """diffusers converts the native (fused) layout into its split one. Both must add the same deltas here, so a
    diffusers-trained file and an ai-toolkit file mean the same thing to the transformer."""
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_ideogram4_lora_to_diffusers

    generator = torch.Generator().manual_seed(1)
    native = {}
    for key, (shape, _dtype) in ideogram4_lora_aitoolkit_format.state_dict_keys.items():
        rank = 4
        shape = [rank, shape[1]] if key.endswith("lora_A.weight") else [shape[0], rank]
        native[key] = torch.randn(shape, generator=generator) / 8
    # A kohya-style alpha on one module, which diffusers folds into the weights.
    native["diffusion_model.layers.0.attention.qkv.alpha"] = torch.tensor(2.0)

    from_native = lora_model_from_ideogram4_state_dict(dict(native))
    from_diffusers = lora_model_from_ideogram4_state_dict(_convert_non_diffusers_ideogram4_lora_to_diffusers(native))

    assert from_native.layers.keys() == from_diffusers.layers.keys()
    for key in from_native.layers:
        module = key.removeprefix(IDEOGRAM4_LORA_TRANSFORMER_PREFIX)
        torch.testing.assert_close(_delta(from_diffusers, module), _delta(from_native, module), msg=key)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"diffusion_model.layers.0.attention.o.lora_magnitude_vector.weight": [4608]}, "lora_magnitude_vector"),
        ({"diffusion_model.layers.0.attention.o.hada_w1_a": [4608, 4]}, "hada_w1_a"),
        ({"diffusion_model.layers.0.attention.o.lora_B.bias": [4608]}, "lora_B.bias"),
        (
            {
                "text_encoders.layers.0.mlp.gate_proj.lora_A.weight": [16, 4096],
                "text_encoders.layers.0.mlp.gate_proj.lora_B.weight": [12288, 16],
            },
            "no Ideogram 4 transformer module.*text_encoders",
        ),
        (
            {
                "transformer.layers.1.attention.to_q.lokr_w1": [8, 8],
                "transformer.layers.1.attention.to_q.lokr_w2": [576, 576],
            },
            "to_q.*not low-rank",
        ),
        ({"diffusion_model.layers.1.attention.o.alpha": []}, "layers.1.attention.o.*incomplete"),
        ({"diffusion_model.layers.1.attention.o.lokr_w1": [8, 8]}, "layers.1.attention.o.*incomplete"),
        (
            {
                "transformer.layers.0.attention.to_q.lora_A.weight": [4, 4608],
                "transformer.layers.0.attention.to_q.lora_B.weight": [4608, 4],
            },
            "layers.0.attention.qkv.*both fused and as split",
        ),
    ],
    ids=[
        "dora",
        "loha",
        "lora-bias",
        "stray-layer",
        "split-lokr",
        "orphan-alpha",
        "one-lokr-factor",
        "fused-and-split",
    ],
)
def test_what_the_conversion_cannot_apply_is_refused(extra: dict[str, list[int]], match: str) -> None:
    state_dict = _state_dict(ideogram4_lora_aitoolkit_format.state_dict_keys)
    state_dict.update({key: torch.zeros(shape) for key, shape in extra.items()})

    with pytest.raises(ValueError, match=match):
        lora_model_from_ideogram4_state_dict(state_dict)


def test_half_of_a_low_rank_pair_is_named() -> None:
    with pytest.raises(ValueError, match=re.escape("layers.0.feed_forward.w2")):
        lora_model_from_ideogram4_state_dict(
            {"diffusion_model.layers.0.feed_forward.w2.lora_A.weight": torch.zeros(16, 12288)}
        )
