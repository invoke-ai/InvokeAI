import accelerate
import pytest
import torch
from diffusers import Krea2Transformer2DModel

from invokeai.backend.model_manager.load.model_loaders.krea2 import KREA2_TRANSFORMER_CONFIG
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import (
    KREA2_LORA_QWEN3VL_PREFIX,
    KREA2_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.krea2_lora_conversion_utils import lora_model_from_krea2_state_dict
from tests.backend.patches.lora_conversions.lora_state_dicts.krea2_lora_kohya_format import (
    state_dict_keys as krea2_kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


def test_peft_layer_preserves_explicit_alpha() -> None:
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.alpha": torch.tensor(1.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, LoRALayer)
    assert layer._alpha == 1.0


def test_peft_dora_layer_preserves_magnitude_and_alpha() -> None:
    dora_scale = torch.full((4, 1), 2.0)
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.dora_scale": dora_scale,
        "transformer.text_fusion.0.attn.to_q.alpha": torch.tensor(1.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, DoRALayer)
    assert layer._alpha == 1.0
    assert torch.equal(layer.dora_scale, dora_scale)


def test_peft_layer_without_explicit_alpha_uses_rank_default() -> None:
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, LoRALayer)
    assert layer._alpha is None


def test_incomplete_peft_pair_raises_descriptive_error() -> None:
    # A layer with lora_A but no matching lora_B is malformed. It must raise a clear ValueError naming the
    # missing key, not an uninformative bare KeyError.
    state_dict = {
        # Complete layer so the dict still looks like a Krea-2 LoRA.
        "transformer.text_fusion.0.attn.to_k.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_k.lora_B.weight": torch.ones(4, 2),
        # Incomplete layer: lora_A present, lora_B missing.
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
    }

    with pytest.raises(ValueError, match="lora_B.weight"):
        lora_model_from_krea2_state_dict(state_dict)


def test_peft_dora_magnitude_vector_key_produces_dora_layer() -> None:
    # Standard PEFT / Diffusers DoRA stores the magnitude as `.lora_magnitude_vector.weight` (not the LyCORIS
    # `.dora_scale`). It must be recognized and produce a DoRALayer preserving the magnitude, so valid
    # Diffusers DoRA adapters load instead of being split into a bogus, unrecognized layer (review 4802322488).
    magnitude = torch.full((4, 1), 3.0)
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.lora_magnitude_vector.weight": magnitude,
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, DoRALayer)
    assert torch.equal(layer.dora_scale, magnitude)


def test_conflicting_transformer_and_diffusion_model_aliases_raise() -> None:
    # `transformer.` and `diffusion_model.` normalize to the same target key. Providing both aliases for one
    # logical layer (with different tensors) must raise, not silently drop one based on dict ordering.
    state_dict = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.full((2, 4), 2.0),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.full((4, 2), 2.0),
    }

    with pytest.raises(ValueError, match="normalize to the same target"):
        lora_model_from_krea2_state_dict(state_dict)


def test_conflicting_native_and_diffusers_aliases_raise() -> None:
    state_dict = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.gate.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.gate.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.full((2, 4), 2.0),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.full((4, 2), 2.0),
    }

    with pytest.raises(ValueError, match="normalize to the same target"):
        lora_model_from_krea2_state_dict(state_dict)


def test_native_comfyui_krea2_keys_are_remapped_to_diffusers_layout() -> None:
    # Native (ComfyUI) Krea-2 LoRAs name modules differently (blocks / attn.wq/wo/gate / mlp / txtfusion).
    # They must be remapped onto the diffusers Krea2Transformer2DModel layout so the LoRA actually applies.
    state_dict = {
        # main transformer block, native gated attention + SwiGLU mlp
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.wo.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wo.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.gate.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.gate.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.mlp.down.lora_A.weight": torch.ones(2, 8),
        "diffusion_model.blocks.0.mlp.down.lora_B.weight": torch.ones(4, 2),
        # text-fusion stage (native `txtfusion`, layerwise + refiner sub-blocks)
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.up.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.up.lora_B.weight": torch.ones(8, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)
    keys = set(model.layers.keys())

    p = KREA2_LORA_TRANSFORMER_PREFIX
    # blocks -> transformer_blocks; attn.wq/wo/gate -> to_q/to_out.0/to_gate; mlp -> ff
    assert f"{p}transformer_blocks.0.attn.to_q" in keys
    assert f"{p}transformer_blocks.0.attn.to_out.0" in keys
    assert f"{p}transformer_blocks.0.attn.to_gate" in keys
    assert f"{p}transformer_blocks.0.ff.down" in keys
    # txtfusion -> text_fusion; layerwise_blocks / refiner_blocks preserved (NOT renamed to transformer_blocks)
    assert f"{p}text_fusion.layerwise_blocks.0.attn.to_k" in keys
    assert f"{p}text_fusion.refiner_blocks.1.ff.up" in keys
    # No native names should survive.
    assert not any(".wq" in k or ".wo" in k or ".mlp." in k or "txtfusion" in k for k in keys)
    assert all(isinstance(v, LoRALayer) for v in model.layers.values())


def test_native_krea2_dora_magnitude_is_preserved_through_remap() -> None:
    # A native DoRA slider (A/B + magnitude) must remap AND keep its magnitude, producing a DoRALayer.
    magnitude = torch.full((4, 1), 2.0)
    state_dict = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.wq.lora_magnitude_vector.weight": magnitude,
        # a second native marker so detection is unambiguous
        "diffusion_model.blocks.0.attn.gate.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.gate.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q"]
    assert isinstance(layer, DoRALayer)
    assert torch.equal(layer.dora_scale, magnitude)


def test_diffusers_layout_krea2_keys_are_left_untouched() -> None:
    # A LoRA already in the diffusers layout must not be altered by the native remap.
    state_dict = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)
    keys = set(model.layers.keys())
    p = KREA2_LORA_TRANSFORMER_PREFIX
    assert f"{p}transformer_blocks.0.attn.to_q" in keys
    assert f"{p}text_fusion.0.attn.to_q" in keys


@pytest.mark.parametrize(
    ("native_module", "diffusers_module"),
    [
        ("blocks.0.attn.wq", "transformer_blocks.0.attn.to_q"),
        ("blocks.0.mlp.down", "transformer_blocks.0.ff.down"),
    ],
)
def test_single_module_native_krea2_lora_is_remapped(
    native_module: str,
    diffusers_module: str,
) -> None:
    state_dict = {
        f"diffusion_model.{native_module}.lora_A.weight": torch.ones(2, 4),
        f"diffusion_model.{native_module}.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert set(model.layers) == {f"{KREA2_LORA_TRANSFORMER_PREFIX}{diffusers_module}"}


@pytest.mark.parametrize(
    ("kohya_module", "diffusers_module"),
    [
        ("blocks_0_attn_wq", "transformer_blocks.0.attn.to_q"),
        ("blocks_0_attn_wk", "transformer_blocks.0.attn.to_k"),
        ("blocks_0_attn_wv", "transformer_blocks.0.attn.to_v"),
        ("blocks_0_attn_wo", "transformer_blocks.0.attn.to_out.0"),
        ("blocks_0_attn_gate", "transformer_blocks.0.attn.to_gate"),
        # Multi-digit block index.
        ("blocks_27_mlp_down", "transformer_blocks.27.ff.down"),
        # `layerwise_blocks` / `refiner_blocks` are the native components that themselves contain an
        # underscore, i.e. the only genuine ambiguity in the flattened form.
        ("txtfusion_layerwise_blocks_0_attn_wo", "text_fusion.layerwise_blocks.0.attn.to_out.0"),
        ("txtfusion_refiner_blocks_1_mlp_gate", "text_fusion.refiner_blocks.1.ff.gate"),
        ("txtfusion_projector", "text_fusion.projector"),
        ("first", "img_in"),
        ("tmlp_0", "time_embed.linear_1"),
        ("tmlp_2", "time_embed.linear_2"),
        ("tproj_1", "time_mod_proj"),
        ("txtmlp_1", "txt_in.linear_1"),
        ("txtmlp_3", "txt_in.linear_2"),
        ("last_linear", "final_layer.linear"),
    ],
)
def test_kohya_flattened_krea2_module_is_remapped(kohya_module: str, diffusers_module: str) -> None:
    # kohya / LyCORIS flatten the module path and prefix it with `lora_unet_`. Without un-flattening, every key
    # misses its module and the adapter is a silent no-op ("Failed to find module for LoRA layer key:
    # lora_transformer-lora_unet_blocks_6_attn_wv").
    state_dict = {
        f"lora_unet_{kohya_module}.lora_down.weight": torch.ones(2, 4),
        f"lora_unet_{kohya_module}.lora_up.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert set(model.layers) == {f"{KREA2_LORA_TRANSFORMER_PREFIX}{diffusers_module}"}


def test_kohya_krea2_lora_layers_match_the_real_transformer() -> None:
    # Every layer of a real kohya Krea-2 adapter must land on an actual Linear of Krea2Transformer2DModel, with
    # in/out features that agree with the LoRA's own down/up shapes. A wrong rename (e.g. ff.gate <-> ff.down,
    # whose SwiGLU shapes are transposed) is caught here rather than as a runtime warning.
    state_dict = keys_to_mock_state_dict(krea2_kohya_state_dict_keys)

    model = lora_model_from_krea2_state_dict(state_dict)

    with accelerate.init_empty_weights():
        transformer = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)

    # 4 blocks x 8 Linears (2 transformer, 1 layerwise + 1 refiner text-fusion) + 8 top-level modules.
    assert len(model.layers) == 40
    for layer_key, layer in model.layers.items():
        module_name = layer_key[len(KREA2_LORA_TRANSFORMER_PREFIX) :]
        submodule = transformer.get_submodule(module_name)
        assert isinstance(submodule, torch.nn.Linear), f"{module_name} is not a Linear"
        out_features, in_features = submodule.weight.shape
        assert layer.down.shape[1] == in_features, f"{module_name} in_features mismatch"
        assert layer.up.shape[0] == out_features, f"{module_name} out_features mismatch"


def test_kohya_flattened_krea2_layer_preserves_alpha() -> None:
    # kohya adapters carry an explicit `.alpha`; it must survive the un-flattening intact, otherwise the LoRA
    # applies at the wrong strength.
    state_dict = {
        "lora_unet_blocks_6_attn_wv.lora_down.weight": torch.ones(2, 4),
        "lora_unet_blocks_6_attn_wv.lora_up.weight": torch.ones(4, 2),
        "lora_unet_blocks_6_attn_wv.alpha": torch.tensor(2.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.6.attn.to_v"]
    assert isinstance(layer, LoRALayer)
    assert layer._alpha == 2.0


def test_kohya_flattened_krea2_keys_tolerate_doubled_separator() -> None:
    state_dict = {
        "lora_unet__blocks_6_attn_wv.lora_down.weight": torch.ones(2, 4),
        "lora_unet__blocks_6_attn_wv.lora_up.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert set(model.layers) == {f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.6.attn.to_v"}


@pytest.mark.parametrize(
    "flat_module",
    [
        # Non-Linear natives: `mod.lin` is folded into the `scale_shift_table` parameter and the norms have no
        # Linear counterpart, so there is nothing to patch. They must not be renamed into a key that merely
        # looks applicable.
        "blocks_6_mod_lin",
        "blocks_6_prenorm",
        "blocks_6_attn_qknorm_qnorm",
        # Not a Krea-2 module layout at all (e.g. a flattened adapter for some other architecture).
        "double_blocks_0_img_attn_proj",
    ],
)
def test_unrecognized_kohya_flattened_keys_are_left_untouched(flat_module: str) -> None:
    state_dict = {
        f"lora_unet_{flat_module}.lora_down.weight": torch.ones(2, 4),
        f"lora_unet_{flat_module}.lora_up.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    # Left verbatim rather than rewritten into a plausible-looking but wrong module path.
    assert set(model.layers) == {f"{KREA2_LORA_TRANSFORMER_PREFIX}lora_unet_{flat_module}"}


def test_conflicting_kohya_and_native_aliases_raise() -> None:
    # The flattened and dotted spellings of one logical layer normalize to the same target. Providing both
    # must raise instead of silently dropping one based on dict ordering.
    state_dict = {
        "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
        "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(4, 2),
        "blocks.0.attn.wq.lora_down.weight": torch.full((2, 4), 2.0),
        "blocks.0.attn.wq.lora_up.weight": torch.full((4, 2), 2.0),
    }

    with pytest.raises(ValueError, match="normalize to the same target"):
        lora_model_from_krea2_state_dict(state_dict)


def test_native_transformer_remap_does_not_change_diffusers_text_encoder_blocks() -> None:
    state_dict = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "text_encoder.visual.blocks.0.attn.qkv.lora_A.weight": torch.ones(2, 4),
        "text_encoder.visual.blocks.0.attn.qkv.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q" in model.layers
    assert f"{KREA2_LORA_QWEN3VL_PREFIX}visual.blocks.0.attn.qkv" in model.layers


def test_native_krea2_top_level_linear_keys_are_remapped() -> None:
    native_to_diffusers = {
        "first": "img_in",
        "tmlp.0": "time_embed.linear_1",
        "tmlp.2": "time_embed.linear_2",
        "tproj.1": "time_mod_proj",
        "txtmlp.1": "txt_in.linear_1",
        "txtmlp.3": "txt_in.linear_2",
        "last.linear": "final_layer.linear",
    }
    state_dict = {
        f"diffusion_model.{module}.{suffix}.weight": torch.ones(2, 4)
        for module in native_to_diffusers
        for suffix in ("lora_A", "lora_B")
    }
    state_dict.update(
        {
            "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
            "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        }
    )

    model = lora_model_from_krea2_state_dict(state_dict)

    expected_keys = {
        f"{KREA2_LORA_TRANSFORMER_PREFIX}{diffusers_module}" for diffusers_module in native_to_diffusers.values()
    }
    assert expected_keys < set(model.layers)
