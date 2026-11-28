from unittest.mock import MagicMock

import accelerate
import pytest
import torch

import invokeai.backend.patches.layers.dora_layer as dora_module
from invokeai.backend.anima.anima_transformer import AnimaTransformer
from invokeai.backend.anima.checkpoint_layout import ANIMA_TRANSFORMER_CONFIG
from invokeai.backend.patches.layers.dora_layer import FactorizedDoRALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.anima_lora_constants import (
    ANIMA_LORA_QWEN3_PREFIX,
    ANIMA_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.anima_lora_conversion_utils import (
    _convert_kohya_te_key,
    _convert_kohya_unet_key,
    anima_lora_for_depth,
    is_state_dict_likely_anima_lora,
    lora_model_from_anima_state_dict,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
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
        # Non-block transformer modules
        ("lora_unet_final_layer_linear", "final_layer.linear"),
        ("lora_unet_final_layer_adaln_modulation_1", "final_layer.adaln_modulation.1"),
        ("lora_unet_final_layer_adaln_modulation_2", "final_layer.adaln_modulation.2"),
        ("lora_unet_t_embedder_1_linear_1", "t_embedder.1.linear_1"),
        ("lora_unet_t_embedder_1_linear_2", "t_embedder.1.linear_2"),
        ("lora_unet_x_embedder_proj_1", "x_embedder.proj.1"),
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
    "kohya_key",
    [
        "lora_unet_final_layer_linear_extra",
        "lora_unet_x_embedder_proj_10",
    ],
)
def test_convert_kohya_unet_key_does_not_guess_unknown_non_block_paths(kohya_key: str):
    """Near-matches stay unchanged instead of being mapped to a different real module."""
    assert _convert_kohya_unet_key(kohya_key) == kohya_key.removeprefix("lora_unet_")


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


def test_kohya_non_block_layers_resolve_alongside_blocks_adapter_and_text_encoder():
    """Converted Kohya paths and weight shapes match the real Anima module graph."""
    non_block_source_to_module = {
        "lora_unet_final_layer_linear": "final_layer.linear",
        "lora_unet_final_layer_adaln_modulation_1": "final_layer.adaln_modulation.1",
        "lora_unet_final_layer_adaln_modulation_2": "final_layer.adaln_modulation.2",
        "lora_unet_t_embedder_1_linear_1": "t_embedder.1.linear_1",
        "lora_unet_t_embedder_1_linear_2": "t_embedder.1.linear_2",
        "lora_unet_x_embedder_proj_1": "x_embedder.proj.1",
    }
    # Keep block and LLM-adapter keys in the same conversion to catch precedence regressions, and retain a text-encoder
    # key to ensure the transformer-only mapping does not affect the separate Qwen route.
    coexisting_source_to_expected = {
        "lora_unet_blocks_0_self_attn_q_proj": f"{ANIMA_LORA_TRANSFORMER_PREFIX}blocks.0.self_attn.q_proj",
        "lora_unet_llm_adapter_blocks_0_cross_attn_k_proj": (
            f"{ANIMA_LORA_TRANSFORMER_PREFIX}llm_adapter.blocks.0.cross_attn.k_proj"
        ),
        "lora_te_layers_0_self_attn_q_proj": f"{ANIMA_LORA_QWEN3_PREFIX}model.layers.0.self_attn.q_proj",
    }

    with accelerate.init_empty_weights():
        transformer = AnimaTransformer(**{**ANIMA_TRANSFORMER_CONFIG, "num_blocks": 1})

    rank = 4
    state_dict: dict[str, torch.Tensor] = {}
    for source_key, module_name in non_block_source_to_module.items():
        module = transformer.get_submodule(module_name)
        assert isinstance(module, torch.nn.Linear), f"{module_name} is not a Linear"
        state_dict[f"{source_key}.lora_down.weight"] = torch.empty(rank, module.in_features)
        state_dict[f"{source_key}.lora_up.weight"] = torch.empty(module.out_features, rank)

    for source_key, patch_key in coexisting_source_to_expected.items():
        if patch_key.startswith(ANIMA_LORA_TRANSFORMER_PREFIX):
            module_name = patch_key.removeprefix(ANIMA_LORA_TRANSFORMER_PREFIX)
            module = transformer.get_submodule(module_name)
            assert isinstance(module, torch.nn.Linear), f"{module_name} is not a Linear"
            in_features, out_features = module.in_features, module.out_features
        else:
            # This test keeps the Qwen key only to pin routing coexistence; its module graph is owned by the encoder.
            in_features = out_features = 8
        state_dict[f"{source_key}.lora_down.weight"] = torch.empty(rank, in_features)
        state_dict[f"{source_key}.lora_up.weight"] = torch.empty(out_features, rank)

    patch = lora_model_from_anima_state_dict(state_dict)

    expected_non_block_keys = {
        f"{ANIMA_LORA_TRANSFORMER_PREFIX}{module_name}" for module_name in non_block_source_to_module.values()
    }
    assert set(patch.layers) == expected_non_block_keys | set(coexisting_source_to_expected.values())

    for patch_key in expected_non_block_keys:
        module_name = patch_key.removeprefix(ANIMA_LORA_TRANSFORMER_PREFIX)
        module = transformer.get_submodule(module_name)
        assert isinstance(module, torch.nn.Linear), f"{module_name} is not a Linear"
        layer = patch.layers[patch_key]
        assert isinstance(layer, LoRALayer)
        assert layer.down.shape == (rank, module.in_features)
        assert layer.up.shape == (module.out_features, rank)


def test_diffusers_cosmos_layers_map_to_real_anima_modules():
    source_to_module = {
        "transformer.transformer_blocks.0.attn1.to_q": "blocks.0.self_attn.q_proj",
        "transformer.transformer_blocks.0.attn1.to_out.0": "blocks.0.self_attn.output_proj",
        "transformer.transformer_blocks.0.attn2.to_k": "blocks.0.cross_attn.k_proj",
        "transformer.transformer_blocks.0.attn2.to_out.0": "blocks.0.cross_attn.output_proj",
        "transformer.transformer_blocks.0.ff.net.0.proj": "blocks.0.mlp.layer1",
        "transformer.transformer_blocks.0.ff.net.2": "blocks.0.mlp.layer2",
        "transformer.transformer_blocks.0.norm1.linear_1": "blocks.0.adaln_modulation_self_attn.1",
        "transformer.transformer_blocks.0.norm2.linear_2": "blocks.0.adaln_modulation_cross_attn.2",
        "transformer.transformer_blocks.0.norm3.linear_1": "blocks.0.adaln_modulation_mlp.1",
        "transformer.patch_embed.proj": "x_embedder.proj.1",
        "transformer.time_embed.t_embedder.linear_1": "t_embedder.1.linear_1",
        "transformer.time_embed.t_embedder.linear_2": "t_embedder.1.linear_2",
        "transformer.norm_out.linear_1": "final_layer.adaln_modulation.1",
        "transformer.norm_out.linear_2": "final_layer.adaln_modulation.2",
        "transformer.proj_out": "final_layer.linear",
    }
    with accelerate.init_empty_weights():
        transformer = AnimaTransformer(**{**ANIMA_TRANSFORMER_CONFIG, "num_blocks": 1})

    rank = 4
    state_dict: dict[str, torch.Tensor] = {}
    for source, target in source_to_module.items():
        module = transformer.get_submodule(target)
        assert isinstance(module, torch.nn.Linear)
        state_dict[f"{source}.lora_A.weight"] = torch.empty(rank, module.in_features)
        state_dict[f"{source}.lora_B.weight"] = torch.empty(module.out_features, rank)

    patch = lora_model_from_anima_state_dict(state_dict)

    assert set(patch.layers) == {f"{ANIMA_LORA_TRANSFORMER_PREFIX}{target}" for target in source_to_module.values()}
    for patch_key, layer in patch.layers.items():
        module = transformer.get_submodule(patch_key.removeprefix(ANIMA_LORA_TRANSFORMER_PREFIX))
        assert isinstance(module, torch.nn.Linear)
        assert isinstance(layer, LoRALayer)
        assert layer.down.shape == (rank, module.in_features)
        assert layer.up.shape == (module.out_features, rank)


@pytest.mark.parametrize("model_prefix", ["", "base_model.model."], ids=["bare", "peft"])
def test_diffusers_cosmos_prefix_variants_map_to_anima(model_prefix: str):
    source = f"{model_prefix}transformer_blocks.0.attn1.to_q"
    state_dict = {
        f"{source}.lora_A.weight": torch.empty(4, 2048),
        f"{source}.lora_B.weight": torch.empty(2048, 4),
    }

    patch = lora_model_from_anima_state_dict(state_dict)

    assert set(patch.layers) == {f"{ANIMA_LORA_TRANSFORMER_PREFIX}blocks.0.self_attn.q_proj"}


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


def test_lokr_dora_keys_make_weight_decomposed_lokr_layers(monkeypatch: pytest.MonkeyPatch):
    """Some Anima LoRAs combine DoRA (dora_scale) with LoKR (lokr_w1/lokr_w2). A 1-D magnitude of the cross-attention
    K (2048 x 1024) can only be per output row, keeps it, and applies to its real module. The same vector on the square
    Q (2048 x 2048) fits both axes, so Q applies as the plain LoKr, with a warning."""
    with accelerate.init_empty_weights():
        transformer = AnimaTransformer(**{**ANIMA_TRANSFORMER_CONFIG, "num_blocks": 1})
    g = torch.Generator().manual_seed(0)
    state_dict = {
        key: torch.ones(shape, dtype=torch.float64)
        if key.endswith(".alpha")
        else torch.rand(shape, generator=g, dtype=torch.float64) + 0.5
        if key.endswith(".dora_scale")
        else torch.randn(shape, generator=g, dtype=torch.float64) * 0.05
        for key, shape in anima_lokr_keys.items()
    }

    warning = MagicMock()
    monkeypatch.setattr(dora_module.logger, "warning", warning)
    monkeypatch.setattr(dora_module, "_magnitude_ambiguous_warned", False)

    lora_model = lora_model_from_anima_state_dict(state_dict)

    warning.assert_called_once()
    assert "2048x2048" in warning.call_args.args[0]
    assert type(lora_model.layers[f"{ANIMA_LORA_TRANSFORMER_PREFIX}blocks.0.cross_attn.q_proj"]) is LoKRLayer
    expected_axes = {"blocks.0.cross_attn.k_proj": True}
    decomposed = {
        key.removeprefix(ANIMA_LORA_TRANSFORMER_PREFIX): layer
        for key, layer in lora_model.layers.items()
        if isinstance(layer, FactorizedDoRALayer)
    }
    assert {name: layer.magnitude_is_out_dim for name, layer in decomposed.items()} == expected_axes
    assert all(isinstance(layer.layer, LoKRLayer) for layer in decomposed.values())
    assert all(isinstance(layer, LoKRLayer) for layer in lora_model.layers.values() if layer not in decomposed.values())

    for name, layer in decomposed.items():
        module = transformer.get_submodule(name)
        weight = torch.randn(module.weight.shape, generator=g, dtype=torch.float64)
        prefix = f"diffusion_model.{name}"
        # kron(w1, w2) of the file's factors, the weight's shape; and the LyCORIS decomposition in float64.
        w1, w2 = state_dict[f"{prefix}.lokr_w1"], state_dict[f"{prefix}.lokr_w2"]
        update = torch.einsum("ij,kl->ikjl", w1, w2).reshape(w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1])
        assert update.shape == weight.shape
        magnitude = state_dict[f"{prefix}.dora_scale"]
        direction = weight + update
        expected = magnitude[:, None] * direction / direction.norm(dim=1, keepdim=True) - weight

        torch.testing.assert_close(layer.get_parameters({"weight": weight}, 1.0)["weight"], expected)


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


# --- depth-expanded finetunes ------------------------------------------------------------------------


def _patch(*keys: str) -> ModelPatchRaw:
    return ModelPatchRaw(layers={key: object() for key in keys})  # type: ignore[misc]


T = ANIMA_LORA_TRANSFORMER_PREFIX


def test_a_base_lora_moves_to_the_base_blocks_of_anima_2_9b() -> None:
    original = _patch(
        f"{T}blocks.0.self_attn.q_proj",
        f"{T}blocks.2.cross_attn.k_proj",
        f"{T}blocks.27.mlp.layer2",
        f"{T}llm_adapter.blocks.3.self_attn.q_proj",
        f"{ANIMA_LORA_QWEN3_PREFIX}layers.2.self_attn.q_proj",
    )
    moved, moved_from = anima_lora_for_depth(original, 40)

    assert moved_from == 28
    assert sorted(moved.layers) == sorted(
        [
            f"{T}blocks.0.self_attn.q_proj",
            f"{T}blocks.3.cross_attn.k_proj",
            f"{T}blocks.39.mlp.layer2",
            # The LLM adapter's own blocks and the text encoder are not DiT blocks.
            f"{T}llm_adapter.blocks.3.self_attn.q_proj",
            f"{ANIMA_LORA_QWEN3_PREFIX}layers.2.self_attn.q_proj",
        ]
    )
    # The same layer objects, and the cached LoRA itself untouched.
    assert set(map(id, moved.layers.values())) == set(map(id, original.layers.values()))
    assert f"{T}blocks.27.mlp.layer2" in original.layers


def test_a_base_lora_moves_to_the_base_blocks_of_anima_3_8b() -> None:
    moved, moved_from = anima_lora_for_depth(_patch(f"{T}blocks.2.mlp.layer1", f"{T}blocks.27.mlp.layer1"), 52)
    assert moved_from == 28
    assert sorted(moved.layers) == [f"{T}blocks.4.mlp.layer1", f"{T}blocks.51.mlp.layer1"]


def test_an_anima_2_9b_lora_moves_to_its_blocks_in_anima_3_8b() -> None:
    moved, moved_from = anima_lora_for_depth(_patch(f"{T}blocks.3.mlp.layer1", f"{T}blocks.39.mlp.layer1"), 52)
    assert moved_from == 40
    assert sorted(moved.layers) == [f"{T}blocks.4.mlp.layer1", f"{T}blocks.51.mlp.layer1"]


@pytest.mark.parametrize(
    ("keys", "depth"),
    [
        ((f"{T}blocks.27.mlp.layer1",), 28),
        ((f"{T}blocks.39.mlp.layer1",), 40),
        ((f"{T}llm_adapter.blocks.0.self_attn.q_proj", f"{ANIMA_LORA_QWEN3_PREFIX}layers.0.mlp.up_proj"), 40),
    ],
    ids=["base-on-base", "2.9B-on-2.9B", "no-dit-blocks"],
)
def test_a_lora_for_this_depth_is_returned_as_it_is(keys, depth) -> None:
    original = _patch(*keys)
    assert anima_lora_for_depth(original, depth) == (original, None)
