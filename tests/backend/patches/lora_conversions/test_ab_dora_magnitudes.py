"""DoRA magnitudes on lora_A/lora_B layers must reach the patch for Z-Image, Qwen-Image and Anima.

Each family's converter used to recognize only some magnitude key spellings and drop the rest, which applied the
adapter as a plain LoRA. The expected patch is computed here from the DoRA paper's formula, independently of
``DoRALayer``.
"""

import pytest
import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.lora_conversions.anima_lora_conversion_utils import lora_model_from_anima_state_dict
from invokeai.backend.patches.lora_conversions.qwen_image_lora_conversion_utils import (
    lora_model_from_qwen_image_state_dict,
)
from invokeai.backend.patches.lora_conversions.z_image_lora_conversion_utils import lora_model_from_z_image_state_dict

OUT_FEATURES = 6
IN_FEATURES = 4
RANK = 2
FILE_ALPHA = 1.0

# family -> (converter, module path inside the transformer, prefixes the family accepts on a transformer layer)
FAMILIES = {
    "z-image": (lora_model_from_z_image_state_dict, "layers.0.attention.to_q", ("diffusion_model.", "transformer.")),
    "anima": (lora_model_from_anima_state_dict, "blocks.0.self_attn.q_proj", ("diffusion_model.", "transformer.")),
    "qwen-image": (
        lora_model_from_qwen_image_state_dict,
        "transformer_blocks.0.attn.to_q",
        ("", "transformer.", "diffusion_model.", "base_model.model."),
    ),
}

# magnitude key suffix -> (shape of the stored magnitude, whether it indexes the output rows)
MAGNITUDE_KEYS = {
    "dora_scale-out": ("dora_scale", (OUT_FEATURES, 1), True),
    "dora_scale-in": ("dora_scale", (1, IN_FEATURES), False),
    "lora_magnitude_vector.weight": ("lora_magnitude_vector.weight", (OUT_FEATURES,), True),
    "lora_magnitude_vector": ("lora_magnitude_vector", (OUT_FEATURES,), True),
    "magnitude": ("magnitude", (OUT_FEATURES,), True),
}

CASES = [
    pytest.param(family, prefixes[0], magnitude, id=f"{family}-{magnitude}")
    for family, (_, _, prefixes) in FAMILIES.items()
    for magnitude in MAGNITUDE_KEYS
]


def _layer_values(generator: torch.Generator) -> dict[str, torch.Tensor]:
    return {
        "lora_A.weight": torch.randn(RANK, IN_FEATURES, generator=generator, dtype=torch.float64),
        "lora_B.weight": torch.randn(OUT_FEATURES, RANK, generator=generator, dtype=torch.float64),
        "alpha": torch.tensor(FILE_ALPHA, dtype=torch.float64),
    }


def _expected_patch(
    weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor, magnitude: torch.Tensor, out_axis: bool
) -> torch.Tensor:
    """W' - W for W' = m * (W + s*B@A) / ||W + s*B@A||, the norm taken per output row or per input column."""
    direction = weight + (FILE_ALPHA / RANK) * (b @ a)
    if out_axis:
        norm = torch.sqrt((direction**2).sum(dim=1, keepdim=True))
        return magnitude.reshape(OUT_FEATURES, 1) * direction / norm - weight
    norm = torch.sqrt((direction**2).sum(dim=0, keepdim=True))
    return magnitude.reshape(1, IN_FEATURES) * direction / norm - weight


@pytest.mark.parametrize(("family", "prefix", "magnitude_case"), CASES)
def test_ab_dora_magnitude_becomes_a_dora_layer_that_applies_it(family: str, prefix: str, magnitude_case: str):
    convert, module, _ = FAMILIES[family]
    suffix, shape, out_axis = MAGNITUDE_KEYS[magnitude_case]
    generator = torch.Generator().manual_seed(0)
    values = _layer_values(generator)
    magnitude = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
    state_dict = {f"{prefix}{module}.{name}": tensor for name, tensor in values.items()}
    state_dict[f"{prefix}{module}.{suffix}"] = magnitude

    model = convert(state_dict, alpha=None)

    assert len(model.layers) == 1
    layer = next(iter(model.layers.values()))
    assert isinstance(layer, DoRALayer)
    assert layer.magnitude_is_out_dim == out_axis
    assert layer.scale() == FILE_ALPHA / RANK
    weight = torch.randn(OUT_FEATURES, IN_FEATURES, generator=generator, dtype=torch.float64)
    patched = layer.get_parameters({"weight": weight}, weight=1.0)["weight"]
    expected = _expected_patch(weight, values["lora_A.weight"], values["lora_B.weight"], magnitude, out_axis)
    torch.testing.assert_close(patched, expected)


@pytest.mark.parametrize(
    ("family", "prefix"),
    [
        pytest.param(family, prefix, id=f"{family}-{prefix or 'bare'}")
        for family, (_, _, p) in FAMILIES.items()
        for prefix in p
    ],
)
def test_every_key_prefix_is_stripped_to_the_same_dora_layer_key(family: str, prefix: str):
    convert, module, _ = FAMILIES[family]
    generator = torch.Generator().manual_seed(2)
    state_dict = {f"{prefix}{module}.{name}": tensor for name, tensor in _layer_values(generator).items()}
    state_dict[f"{prefix}{module}.magnitude"] = torch.ones(OUT_FEATURES, dtype=torch.float64)
    bare = {key.removeprefix(prefix): value for key, value in state_dict.items()}

    layers = convert(state_dict, alpha=None).layers

    assert isinstance(next(iter(layers.values())), DoRALayer)
    assert layers.keys() == convert(bare, alpha=None).layers.keys()


@pytest.mark.parametrize("family", FAMILIES)
def test_caller_alpha_applies_to_a_dora_layer_without_a_file_alpha(family: str):
    convert, module, (prefix, *_) = FAMILIES[family]
    generator = torch.Generator().manual_seed(1)
    values = _layer_values(generator)
    del values["alpha"]
    state_dict = {f"{prefix}{module}.{name}": tensor for name, tensor in values.items()}
    state_dict[f"{prefix}{module}.magnitude"] = torch.ones(OUT_FEATURES, dtype=torch.float64)

    layer = next(iter(convert(state_dict, alpha=4.0).layers.values()))

    assert isinstance(layer, DoRALayer)
    assert layer.scale() == 4.0 / RANK


@pytest.mark.parametrize(
    ("convert", "layer_name"),
    [
        (lora_model_from_z_image_state_dict, "lora_unet__layers_0_attention_to_q"),
        (lora_model_from_anima_state_dict, "lora_unet_blocks_0_self_attn_q_proj"),
        (lora_model_from_qwen_image_state_dict, "lora_unet_transformer_blocks_0_attn_to_q"),
    ],
    ids=["z-image", "anima", "qwen-image"],
)
def test_kohya_dora_scale_stays_a_dora_layer(convert, layer_name: str):
    state_dict = {
        f"{layer_name}.lora_down.weight": torch.zeros(RANK, IN_FEATURES),
        f"{layer_name}.lora_up.weight": torch.zeros(OUT_FEATURES, RANK),
        f"{layer_name}.dora_scale": torch.ones(OUT_FEATURES, 1),
    }

    layers = convert(state_dict, alpha=None).layers

    assert len(layers) == 1
    assert isinstance(next(iter(layers.values())), DoRALayer)
