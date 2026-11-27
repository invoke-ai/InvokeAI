"""SD 3 LoRA conversion: every published layer has to land on a module of the real transformer or CLIP encoder.

A layer key that resolves to nothing is not an error at runtime -- the patcher logs it and skips it, and the LoRA
quietly does less than it should. So each published layout is converted and resolved against the released SD 3.5
Medium transformer and its two CLIP encoders, built on the meta device from the configs of the stripped model fixture.

kohya's MMDiT names need real work -- fused attention split in three, two adaLN projections stored with their halves
swapped -- so that mapping is checked against diffusers' own single-file conversion of the weights it patches.
"""

import json
import logging
import re
from pathlib import Path

import pytest
import torch

from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.sd3_lora_constants import (
    SD3_LORA_CLIP_G_PREFIX,
    SD3_LORA_CLIP_L_PREFIX,
    SD3_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.sd3_lora_conversion_utils import lora_model_from_sd3_state_dict
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    sd3_lokr_simpletuner_format,
    sd3_lora_diffusers_format,
    sd3_lora_diffusers_full_format,
    sd3_lora_diffusers_te_format,
    sd3_lora_kohya_format,
    sd3_lora_peft_format,
)

_SD35_MEDIUM = (
    Path(__file__).parents[3] / "model_identification" / "stripped_models" / "5a5fe5e4-5776-43e9-a062-530080b2ead1"
)


def _state_dict(keys: dict) -> dict[str, torch.Tensor]:
    return {key: torch.zeros((), dtype=torch.float32).expand(shape) for key, (shape, _dtype) in keys.items()}


def _config(subfolder: str) -> dict:
    return {
        k: v
        for k, v in json.loads((_SD35_MEDIUM / subfolder / "config.json").read_text()).items()
        if not k.startswith("_")
    }


@pytest.fixture(scope="module")
def released_modules() -> dict[str, torch.nn.Module]:
    from diffusers import SD3Transformer2DModel
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection

    with torch.device("meta"):
        return {
            SD3_LORA_TRANSFORMER_PREFIX: SD3Transformer2DModel(**_config("transformer")),
            SD3_LORA_CLIP_L_PREFIX: CLIPTextModelWithProjection(CLIPTextConfig(**_config("text_encoder"))),
            SD3_LORA_CLIP_G_PREFIX: CLIPTextModelWithProjection(CLIPTextConfig(**_config("text_encoder_2"))),
        }


_LAYOUTS = {
    # Block 0 (8 attention projections) and the subset's single out-of-stack layer counts.
    "diffusers": (sd3_lora_diffusers_format.state_dict_keys, {SD3_LORA_TRANSFORMER_PREFIX: 8}),
    "peft": (sd3_lora_peft_format.state_dict_keys, {SD3_LORA_TRANSFORMER_PREFIX: 8}),
    # Block 0: attention (no add_*_proj in this file), dual attention, both feed-forwards and both norms -- 15 --
    # plus the seven top-level modules (pos_embed.proj a Conv2d).
    "diffusers-full": (sd3_lora_diffusers_full_format.state_dict_keys, {SD3_LORA_TRANSFORMER_PREFIX: 22}),
    # Transformer block 0 plus CLIP-L and CLIP-G layer 0.
    "diffusers-te": (
        sd3_lora_diffusers_te_format.state_dict_keys,
        {SD3_LORA_TRANSFORMER_PREFIX: 4, SD3_LORA_CLIP_L_PREFIX: 4, SD3_LORA_CLIP_G_PREFIX: 4},
    ),
    # Whole files: every module of the 24 blocks a LyCORIS (LoKR) and a kohya (fused qkv, split in three) trains.
    "simpletuner-lokr": (sd3_lokr_simpletuner_format.state_dict_keys, {SD3_LORA_TRANSFORMER_PREFIX: 243}),
    "kohya": (sd3_lora_kohya_format.state_dict_keys, {SD3_LORA_TRANSFORMER_PREFIX: 385}),
}


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_every_published_layer_lands_on_a_module_of_the_released_models(
    released_modules: dict[str, torch.nn.Module], layout: str
) -> None:
    keys, expected_counts = _LAYOUTS[layout]
    patch = lora_model_from_sd3_state_dict(_state_dict(keys))

    counts: dict[str, int] = {}
    for key, layer in patch.layers.items():
        prefix = next(p for p in released_modules if key.startswith(p))
        counts[prefix] = counts.get(prefix, 0) + 1
        module = released_modules[prefix].get_submodule(key.removeprefix(prefix))
        assert isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)), key
        if isinstance(layer, LoRALayer):
            assert layer.up.shape[0] == module.weight.shape[0], key
            assert layer.down.shape[1] == module.weight.shape[1], key
        else:
            assert isinstance(layer, LoKRLayer), key
            assert layer.get_weight(torch.empty(0)).shape == module.weight.shape, key
    assert counts == expected_counts


# A two-block MMDiT the size of a unit test: 64 wide, dual attention in block 0, block 1 the last (context-only).
_D, _HEAD, _CH, _JOINT, _POOLED, _PATCH = 64, 32, 4, 48, 40, 2


def _mmdit_checkpoint(generator: torch.Generator) -> dict[str, torch.Tensor]:
    def r(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    ckpt = {
        "pos_embed": r(1, 16, _D),
        "x_embedder.proj.weight": r(_D, _CH, _PATCH, _PATCH),
        "x_embedder.proj.bias": r(_D),
        "t_embedder.mlp.0.weight": r(_D, 256),
        "t_embedder.mlp.0.bias": r(_D),
        "t_embedder.mlp.2.weight": r(_D, _D),
        "t_embedder.mlp.2.bias": r(_D),
        "y_embedder.mlp.0.weight": r(_D, _POOLED),
        "y_embedder.mlp.0.bias": r(_D),
        "y_embedder.mlp.2.weight": r(_D, _D),
        "y_embedder.mlp.2.bias": r(_D),
        "context_embedder.weight": r(_D, _JOINT),
        "context_embedder.bias": r(_D),
        "final_layer.linear.weight": r(_PATCH * _PATCH * _CH, _D),
        "final_layer.linear.bias": r(_PATCH * _PATCH * _CH),
        "final_layer.adaLN_modulation.1.weight": r(2 * _D, _D),
        "final_layer.adaLN_modulation.1.bias": r(2 * _D),
    }
    for i, last in ((0, False), (1, True)):
        for stream in ("x_block", "context_block"):
            p = f"joint_blocks.{i}.{stream}"
            ckpt |= {f"{p}.attn.qkv.weight": r(3 * _D, _D), f"{p}.attn.qkv.bias": r(3 * _D)}
            ckpt |= {f"{p}.attn.ln_q.weight": r(_HEAD), f"{p}.attn.ln_k.weight": r(_HEAD)}
            if stream == "x_block" or not last:
                ckpt |= {f"{p}.attn.proj.weight": r(_D, _D), f"{p}.attn.proj.bias": r(_D)}
                ckpt |= {f"{p}.mlp.fc1.weight": r(4 * _D, _D), f"{p}.mlp.fc1.bias": r(4 * _D)}
                ckpt |= {f"{p}.mlp.fc2.weight": r(_D, 4 * _D), f"{p}.mlp.fc2.bias": r(_D)}
        modulation = 9 * _D if i == 0 else 6 * _D
        ckpt |= {f"joint_blocks.{i}.x_block.adaLN_modulation.1.weight": r(modulation, _D)}
        ckpt |= {f"joint_blocks.{i}.x_block.adaLN_modulation.1.bias": r(modulation)}
        context_modulation = 2 * _D if last else 6 * _D
        ckpt |= {f"joint_blocks.{i}.context_block.adaLN_modulation.1.weight": r(context_modulation, _D)}
        ckpt |= {f"joint_blocks.{i}.context_block.adaLN_modulation.1.bias": r(context_modulation)}
    p = "joint_blocks.0.x_block.attn2"
    ckpt |= {f"{p}.qkv.weight": r(3 * _D, _D), f"{p}.qkv.bias": r(3 * _D), f"{p}.proj.weight": r(_D, _D)}
    ckpt |= {f"{p}.proj.bias": r(_D), f"{p}.ln_q.weight": r(_HEAD), f"{p}.ln_k.weight": r(_HEAD)}
    return ckpt


def test_a_kohya_lora_patches_what_diffusers_converts_the_same_weights_to() -> None:
    """Every kohya layer -- the fused and swapped ones included -- has to add, after the conversion, exactly the
    change diffusers' single-file conversion makes of the same change applied to the original weights."""
    from diffusers.loaders.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers

    generator = torch.Generator().manual_seed(0)
    base = _mmdit_checkpoint(generator)
    lora: dict[str, torch.Tensor] = {}
    patched = dict(base)
    for weight_key, weight in base.items():
        if not weight_key.endswith(".weight") or weight.ndim < 2 or ".ln_" in weight_key:
            continue
        name = "lora_unet_" + weight_key.removesuffix(".weight").replace(".", "_")
        rank = 3
        down = torch.randn(rank, *weight.shape[1:], generator=generator)
        up = torch.randn(weight.shape[0], rank, *([1] * (weight.ndim - 2)), generator=generator)
        lora[f"{name}.lora_down.weight"], lora[f"{name}.lora_up.weight"] = down, up
        lora[f"{name}.alpha"] = torch.tensor(6.0)  # scale 2
        delta = (up.flatten(1) @ down.flatten(1)).reshape(weight.shape) * 2.0
        patched[weight_key] = weight + delta

    converted_base = convert_sd3_transformer_checkpoint_to_diffusers(dict(base))
    converted_patched = convert_sd3_transformer_checkpoint_to_diffusers(dict(patched))
    patch = lora_model_from_sd3_state_dict(lora)

    expected = {
        key.removesuffix(".weight"): converted_patched[key] - converted_base[key]
        for key in converted_base
        if key.endswith(".weight") and not torch.equal(converted_patched[key], converted_base[key])
    }
    actual = {
        key.removeprefix(SD3_LORA_TRANSFORMER_PREFIX): layer.get_weight(torch.empty(0)) * layer.scale()
        for key, layer in patch.layers.items()
    }
    assert actual.keys() == expected.keys()
    for module, delta in expected.items():
        torch.testing.assert_close(actual[module].reshape(delta.shape), delta, rtol=1e-4, atol=1e-4, msg=module)


def test_t5_layers_are_dropped_with_a_warning(caplog) -> None:
    state_dict = _state_dict(sd3_lora_diffusers_format.state_dict_keys)
    state_dict["text_encoder_3.encoder.block.0.layer.0.SelfAttention.q.lora_A.weight"] = torch.zeros(4, 4096)
    state_dict["text_encoder_3.encoder.block.0.layer.0.SelfAttention.q.lora_B.weight"] = torch.zeros(4096, 4)

    with caplog.at_level(logging.WARNING):
        patch = lora_model_from_sd3_state_dict(state_dict)

    assert not any("encoder.block" in key for key in patch.layers)
    assert "dropped 1 T5" in caplog.text


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"transformer.transformer_blocks.0.attn.to_q.lora_magnitude_vector.weight": [1536]}, "lora_magnitude_vector"),
        (
            {
                "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": [4, 1536],
                "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight": [1536, 4],
            },
            "no SD 3 transformer or text encoder module",
        ),
        ({"transformer.transformer_blocks.1.attn.to_q.lora_A.weight": [4, 1536]}, "transformer_blocks.1.attn.to_q"),
        (
            {
                "lora_unet_joint_blocks_1_x_block_attn_qkv.lokr_w1": [3, 3],
                "lora_unet_joint_blocks_1_x_block_attn_qkv.lokr_w2": [1536, 512],
            },
            "x_block_attn_qkv.*not low-rank",
        ),
        (
            {
                "lora_unet_joint_blocks_23_context_block_adaLN_modulation_1.lokr_w1": [2, 2],
                "lora_unet_joint_blocks_23_context_block_adaLN_modulation_1.lokr_w2": [1536, 768],
            },
            "context_block_adaLN_modulation_1.*not low-rank",
        ),
        ({"transformer.transformer_blocks.1.attn.to_k.lokr_w1": [6, 6]}, "to_k.*incomplete"),
        ({"transformer.transformer_blocks.1.attn.to_k.lokr_w2": [256, 256]}, "to_k.*incomplete"),
        (
            {
                "transformer.transformer_blocks.1.attn.to_k.hada_w1_a": [1536, 4],
                "transformer.transformer_blocks.1.attn.to_k.hada_w1_b": [4, 1536],
            },
            "to_k.*incomplete",
        ),
        ({"transformer.transformer_blocks.1.attn.to_k.alpha": []}, "to_k.*incomplete"),
        (
            {
                "transformer.transformer_blocks.1.attn.to_k.lora_A.weight": [4, 1536],
                "transformer.transformer_blocks.1.attn.to_k.lora_up.weight": [1536, 4],
            },
            "to_k.*incomplete",
        ),
    ],
    ids=[
        "dora",
        "stray-layer",
        "half-pair",
        "fused-lokr",
        "swapped-lokr",
        "one-lokr-factor",
        "other-lokr-factor",
        "half-a-loha",
        "alpha-alone",
        "mixed-spellings",
    ],
)
def test_what_the_conversion_cannot_apply_is_refused(extra: dict[str, list[int]], match: str) -> None:
    state_dict = _state_dict(sd3_lora_diffusers_format.state_dict_keys)
    state_dict.update({key: torch.zeros(shape) for key, shape in extra.items()})

    with pytest.raises(ValueError, match=match):
        lora_model_from_sd3_state_dict(state_dict)


def test_a_flattened_lycoris_name_resolves_to_the_dotted_module() -> None:
    """Module names contain underscores of their own (`add_q_proj`, `norm1_context`, `linear_1`)."""
    patch = lora_model_from_sd3_state_dict(
        {
            "lycoris_transformer_blocks_7_norm1_context_linear.lokr_w1": torch.zeros(6, 6),
            "lycoris_transformer_blocks_7_norm1_context_linear.lokr_w2": torch.zeros(1536, 256),
            "lycoris_time_text_embed_text_embedder_linear_1.lokr_w1": torch.zeros(6, 6),
            "lycoris_time_text_embed_text_embedder_linear_1.lokr_w2": torch.zeros(256, 341),
        }
    )

    assert set(patch.layers) == {
        f"{SD3_LORA_TRANSFORMER_PREFIX}transformer_blocks.7.norm1_context.linear",
        f"{SD3_LORA_TRANSFORMER_PREFIX}time_text_embed.text_embedder.linear_1",
    }
    assert all(re.fullmatch(r"lora_transformer-[a-z_0-9.]+", key) for key in patch.layers)


def test_a_lokr_on_a_modulation_that_needs_no_swap_is_applied_as_is() -> None:
    """Only the 2x-wide adaLN projections are stored swapped; the others map onto their diffusers module unchanged,
    whatever the layer type."""
    patch = lora_model_from_sd3_state_dict(
        {
            "lora_unet_joint_blocks_0_x_block_adaLN_modulation_1.lokr_w1": torch.zeros(6, 6),
            "lora_unet_joint_blocks_0_x_block_adaLN_modulation_1.lokr_w2": torch.zeros(2304, 256),
        }
    )

    layer = patch.layers[f"{SD3_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.norm1.linear"]
    assert isinstance(layer, LoKRLayer)
    assert layer.get_weight(torch.empty(0)).shape == (13824, 1536)
