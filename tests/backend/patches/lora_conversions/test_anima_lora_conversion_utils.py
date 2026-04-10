import pytest
import torch

from invokeai.backend.patches.lora_conversions.anima_lora_constants import (
    ANIMA_LORA_QWEN3_PREFIX,
    ANIMA_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.anima_lora_conversion_utils import (
    _convert_kohya_te_key,
    _convert_kohya_unet_key,
    is_state_dict_likely_anima_lora,
    lora_model_from_anima_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.anima_lora_kohya_format import (
    state_dict_keys as anima_kohya_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.anima_lora_kohya_with_te_format import (
    state_dict_keys as anima_kohya_te_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.anima_lora_lokr_format import (
    state_dict_keys as anima_lokr_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.anima_lora_peft_format import (
    state_dict_keys as anima_peft_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict

# --- Detection Tests ---


@pytest.mark.parametrize(
    "sd_keys",
    [anima_kohya_keys, anima_kohya_te_keys, anima_peft_keys, anima_lokr_keys],
    ids=["kohya", "kohya_te", "peft", "lokr"],
)
def test_is_state_dict_likely_anima_lora_true(sd_keys: dict[str, list[int]]):
    """Test that is_state_dict_likely_anima_lora() correctly identifies Anima LoRA state dicts."""
    state_dict = keys_to_mock_state_dict(sd_keys)
    assert is_state_dict_likely_anima_lora(state_dict)


def test_is_state_dict_likely_anima_lora_false_for_flux():
    """Test that is_state_dict_likely_anima_lora() returns False for a FLUX LoRA state dict."""
    state_dict = {
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.empty([16, 3072]),
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.empty([3072, 16]),
    }
    assert not is_state_dict_likely_anima_lora(state_dict)


def test_is_state_dict_likely_anima_lora_false_for_generic_blocks():
    """Test that is_state_dict_likely_anima_lora() returns False for a hypothetical architecture
    that uses lora_unet_blocks_ but with non-Cosmos DiT subcomponent names."""
    state_dict = {
        # Has lora_unet_blocks_ prefix but uses 'attention' and 'ff' instead of
        # Cosmos DiT subcomponents (cross_attn, self_attn, mlp, adaln_modulation)
        "lora_unet_blocks_0_attention_to_q.lora_down.weight": torch.empty([16, 512]),
        "lora_unet_blocks_0_attention_to_q.lora_up.weight": torch.empty([512, 16]),
        "lora_unet_blocks_0_ff_net_0_proj.lora_down.weight": torch.empty([16, 512]),
        "lora_unet_blocks_0_ff_net_0_proj.lora_up.weight": torch.empty([2048, 16]),
    }
    assert not is_state_dict_likely_anima_lora(state_dict)


def test_is_state_dict_likely_anima_lora_false_for_generic_peft_blocks():
    """Test that is_state_dict_likely_anima_lora() returns False for a hypothetical architecture
    that uses transformer.blocks. in PEFT format but with non-Cosmos subcomponents."""
    state_dict = {
        "transformer.blocks.0.attention.to_q.lora_A.weight": torch.empty([16, 512]),
        "transformer.blocks.0.attention.to_q.lora_B.weight": torch.empty([512, 16]),
        "transformer.blocks.0.ff.net.0.proj.lora_A.weight": torch.empty([16, 512]),
        "transformer.blocks.0.ff.net.0.proj.lora_B.weight": torch.empty([2048, 16]),
    }
    assert not is_state_dict_likely_anima_lora(state_dict)


def test_is_state_dict_likely_anima_lora_false_for_random():
    """Test that is_state_dict_likely_anima_lora() returns False for unrelated state dicts."""
    state_dict = {
        "some_random_key.weight": torch.empty([64, 64]),
        "another_key.bias": torch.empty([64]),
    }
    assert not is_state_dict_likely_anima_lora(state_dict)


# --- Kohya Key Conversion Tests ---


@pytest.mark.parametrize(
    ["kohya_key", "expected"],
    [
        ("lora_unet_blocks_0_cross_attn_k_proj", "blocks.0.cross_attn.k_proj"),
        ("lora_unet_blocks_0_cross_attn_q_proj", "blocks.0.cross_attn.q_proj"),
        ("lora_unet_blocks_0_cross_attn_v_proj", "blocks.0.cross_attn.v_proj"),
        ("lora_unet_blocks_0_cross_attn_output_proj", "blocks.0.cross_attn.output_proj"),
        ("lora_unet_blocks_0_self_attn_k_proj", "blocks.0.self_attn.k_proj"),
        ("lora_unet_blocks_0_self_attn_q_proj", "blocks.0.self_attn.q_proj"),
        ("lora_unet_blocks_0_self_attn_v_proj", "blocks.0.self_attn.v_proj"),
        ("lora_unet_blocks_0_self_attn_output_proj", "blocks.0.self_attn.output_proj"),
        ("lora_unet_blocks_0_mlp_layer1", "blocks.0.mlp.layer1"),
        ("lora_unet_blocks_0_mlp_layer2", "blocks.0.mlp.layer2"),
        ("lora_unet_blocks_27_cross_attn_k_proj", "blocks.27.cross_attn.k_proj"),
        ("lora_unet_blocks_0_adaln_modulation_cross_attn_1", "blocks.0.adaln_modulation_cross_attn.1"),
        ("lora_unet_blocks_0_adaln_modulation_self_attn_1", "blocks.0.adaln_modulation_self_attn.1"),
        ("lora_unet_blocks_0_adaln_modulation_mlp_1", "blocks.0.adaln_modulation_mlp.1"),
        # LLM Adapter keys
        ("lora_unet_llm_adapter_blocks_0_cross_attn_k_proj", "llm_adapter.blocks.0.cross_attn.k_proj"),
        ("lora_unet_llm_adapter_blocks_0_cross_attn_q_proj", "llm_adapter.blocks.0.cross_attn.q_proj"),
        ("lora_unet_llm_adapter_blocks_0_cross_attn_v_proj", "llm_adapter.blocks.0.cross_attn.v_proj"),
        ("lora_unet_llm_adapter_blocks_0_self_attn_k_proj", "llm_adapter.blocks.0.self_attn.k_proj"),
        ("lora_unet_llm_adapter_blocks_0_self_attn_q_proj", "llm_adapter.blocks.0.self_attn.q_proj"),
        ("lora_unet_llm_adapter_blocks_0_self_attn_v_proj", "llm_adapter.blocks.0.self_attn.v_proj"),
        ("lora_unet_llm_adapter_blocks_5_cross_attn_k_proj", "llm_adapter.blocks.5.cross_attn.k_proj"),
    ],
)
def test_convert_kohya_unet_key(kohya_key: str, expected: str):
    """Test that Kohya unet keys are correctly converted to model parameter paths."""
    assert _convert_kohya_unet_key(kohya_key) == expected


@pytest.mark.parametrize(
    ["kohya_key", "expected"],
    [
        ("lora_te_layers_0_self_attn_q_proj", "model.layers.0.self_attn.q_proj"),
        ("lora_te_layers_0_self_attn_k_proj", "model.layers.0.self_attn.k_proj"),
        ("lora_te_layers_0_self_attn_v_proj", "model.layers.0.self_attn.v_proj"),
        ("lora_te_layers_0_self_attn_o_proj", "model.layers.0.self_attn.o_proj"),
        ("lora_te_layers_0_mlp_gate_proj", "model.layers.0.mlp.gate_proj"),
        ("lora_te_layers_0_mlp_down_proj", "model.layers.0.mlp.down_proj"),
        ("lora_te_layers_0_mlp_up_proj", "model.layers.0.mlp.up_proj"),
        ("lora_te_layers_15_self_attn_q_proj", "model.layers.15.self_attn.q_proj"),
    ],
)
def test_convert_kohya_te_key(kohya_key: str, expected: str):
    """Test that Kohya TE keys are correctly converted to Qwen3 model parameter paths.

    The Qwen3 text encoder is loaded as Qwen3ForCausalLM which wraps the base model
    under a `model.` prefix, so all converted paths must include it.
    """
    assert _convert_kohya_te_key(kohya_key) == expected


# --- End-to-End Conversion Tests ---


@pytest.mark.parametrize(
    "sd_keys",
    [anima_kohya_keys, anima_kohya_te_keys, anima_peft_keys, anima_lokr_keys],
    ids=["kohya", "kohya_te", "peft", "lokr"],
)
def test_lora_model_from_anima_state_dict(sd_keys: dict[str, list[int]]):
    """Test that a ModelPatchRaw can be created from all supported Anima LoRA formats."""
    state_dict = keys_to_mock_state_dict(sd_keys)
    lora_model = lora_model_from_anima_state_dict(state_dict)
    assert len(lora_model.layers) > 0


def test_kohya_unet_keys_get_transformer_prefix():
    """Test that Kohya unet keys are prefixed with the transformer prefix."""
    state_dict = keys_to_mock_state_dict(anima_kohya_keys)
    lora_model = lora_model_from_anima_state_dict(state_dict)

    for key in lora_model.layers.keys():
        assert key.startswith(ANIMA_LORA_TRANSFORMER_PREFIX), (
            f"Expected transformer prefix '{ANIMA_LORA_TRANSFORMER_PREFIX}', got key: {key}"
        )


def test_kohya_te_keys_get_qwen3_prefix():
    """Test that Kohya TE keys are prefixed with the Qwen3 prefix."""
    state_dict = keys_to_mock_state_dict(anima_kohya_te_keys)
    lora_model = lora_model_from_anima_state_dict(state_dict)

    has_transformer_keys = False
    has_qwen3_keys = False
    for key in lora_model.layers.keys():
        if key.startswith(ANIMA_LORA_TRANSFORMER_PREFIX):
            has_transformer_keys = True
        elif key.startswith(ANIMA_LORA_QWEN3_PREFIX):
            has_qwen3_keys = True
        else:
            raise AssertionError(f"Key has unexpected prefix: {key}")

    assert has_transformer_keys, "Expected at least one transformer key"
    assert has_qwen3_keys, "Expected at least one Qwen3 key"


def test_qwen3_keys_include_model_prefix():
    """Test that converted Qwen3 TE keys include 'model.' prefix for Qwen3ForCausalLM."""
    state_dict = keys_to_mock_state_dict(anima_kohya_te_keys)
    lora_model = lora_model_from_anima_state_dict(state_dict)

    for key in lora_model.layers.keys():
        if key.startswith(ANIMA_LORA_QWEN3_PREFIX):
            inner_key = key[len(ANIMA_LORA_QWEN3_PREFIX) :]
            assert inner_key.startswith("model."), (
                f"Qwen3 key should start with 'model.' after prefix, got: {inner_key}"
            )


def test_lokr_dora_keys_dont_crash():
    """Test that LoKR layers with dora_scale don't cause a KeyError.

    Some Anima LoRAs combine DoRA (dora_scale) with LoKR (lokr_w1/lokr_w2).
    The dora_scale should be stripped from LoKR layers since shared code
    doesn't support DoRA+LoKR combination.
    """
    state_dict = keys_to_mock_state_dict(anima_lokr_keys)
    lora_model = lora_model_from_anima_state_dict(state_dict)
    assert len(lora_model.layers) > 0


def test_peft_keys_get_transformer_prefix():
    """Test that diffusers PEFT keys are prefixed with the transformer prefix."""
    state_dict = keys_to_mock_state_dict(anima_peft_keys)
    lora_model = lora_model_from_anima_state_dict(state_dict)

    for key in lora_model.layers.keys():
        assert key.startswith(ANIMA_LORA_TRANSFORMER_PREFIX), f"Expected transformer prefix, got key: {key}"
        # Verify the diffusion_model. prefix is stripped
        inner_key = key[len(ANIMA_LORA_TRANSFORMER_PREFIX) :]
        assert not inner_key.startswith("diffusion_model."), (
            f"diffusion_model. prefix should be stripped, got: {inner_key}"
        )


def test_empty_state_dict_returns_empty_model():
    """An empty state dict should produce a ModelPatchRaw with no layers."""
    lora_model = lora_model_from_anima_state_dict({})
    assert len(lora_model.layers) == 0
