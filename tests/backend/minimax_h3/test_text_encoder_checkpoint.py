"""Tests for MiniMax H3 single-file (truncated, optionally int8-convrot) text-encoder support.

The Comfy-Org ``qwen3vl_32b_minimax_h3_*`` files store a 50-layer truncation of Qwen3-VL-32B
with no final norm and no LM head (the H3 conditioning contract is the UNNORMALIZED hidden
state after layer 50). These tests pin the three load-bearing pieces: the key conversion, the
Identity-final-norm trick that keeps ``hidden_states[50]`` unnormalized on a truncated stack,
and the depth guard that rejects every other truncated encoder.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER
from invokeai.backend.minimax_h3.text_conditioning import validate_text_encoder_depth
from invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils import (
    convert_minimax_h3_text_encoder_checkpoint,
)


def _marker_blob() -> torch.Tensor:
    payload = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8)


def test_te_converter_renames_and_markers() -> None:
    sd = {
        "model.embed_tokens.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4, dtype=torch.int8),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.zeros(4, 1),
        "model.layers.0.self_attn.q_proj.comfy_quant": _marker_blob(),
        "visual.blocks.0.attn.qkv.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
        "visual.patch_embed.proj.weight": torch.zeros(2, 3, 2, 4, 4, dtype=torch.bfloat16),
    }
    converted, markers = convert_minimax_h3_text_encoder_checkpoint(sd)

    assert set(converted.keys()) == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight_scale",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.weight",
    }
    assert set(markers.keys()) == {"model.language_model.layers.0.self_attn.q_proj"}
    assert markers["model.language_model.layers.0.self_attn.q_proj"]["convrot"] is True


class _FakeEncoder(SimpleNamespace):
    """The two attributes validate_text_encoder_depth reads, in real-module form where needed."""

    @classmethod
    def build(cls, num_layers: int, norm: torch.nn.Module) -> "_FakeEncoder":
        return cls(
            config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=num_layers)),
            model=SimpleNamespace(language_model=SimpleNamespace(norm=norm)),
        )


def test_validate_text_encoder_depth() -> None:
    full = _FakeEncoder.build(64, torch.nn.RMSNorm(8))
    validate_text_encoder_depth(full)  # no raise

    truncated_identity = _FakeEncoder.build(MINIMAX_H3_TEXT_ENCODER_LAYER, torch.nn.Identity())
    validate_text_encoder_depth(truncated_identity)  # no raise

    with pytest.raises(ValueError, match="post-norm"):
        validate_text_encoder_depth(_FakeEncoder.build(MINIMAX_H3_TEXT_ENCODER_LAYER, torch.nn.RMSNorm(8)))

    with pytest.raises(ValueError, match="post-norm"):
        validate_text_encoder_depth(_FakeEncoder.build(36, torch.nn.Identity()))


def test_identity_final_norm_exposes_unnormalized_last_layer() -> None:
    """On a truncated stack, transformers puts the post-final-norm output at
    hidden_states[num_layers]. With the final norm replaced by Identity, that entry must be
    bit-identical to the last decoder layer's raw output (the H3 conditioning contract)."""
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "max_position_embeddings": 128,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "depth": 2,
            "num_heads": 2,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 1,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        tie_word_embeddings=True,
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    num_layers = config.text_config.num_hidden_layers
    input_ids = torch.randint(0, 64, (1, 7))

    with torch.no_grad():
        stock = model.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)

    raw_last_layer_output: list[torch.Tensor] = []
    model.model.language_model.layers[-1].register_forward_hook(
        lambda module, args, output: raw_last_layer_output.append(output[0] if isinstance(output, tuple) else output)
    )
    model.model.language_model.norm = torch.nn.Identity()
    with torch.no_grad():
        truncated = model.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)

    # Entries before the final one are unaffected by the norm swap...
    for k in range(num_layers):
        assert torch.equal(stock.hidden_states[k], truncated.hidden_states[k])
    # ...the final entry was post-norm, and with Identity becomes the raw last-layer output.
    assert not torch.equal(stock.hidden_states[num_layers], truncated.hidden_states[num_layers])
    assert torch.equal(truncated.hidden_states[num_layers], raw_last_layer_output[0])


def test_truncated_identity_model_matches_full_model_cross_depth() -> None:
    """The load-bearing equivalence: a model truncated to N layers with an Identity final norm
    must expose at hidden_states[N] the SAME tensor a FULL-depth model exposes at
    hidden_states[N]. Comparing across depths (weights copied) catches any indexing shift that
    the single-model before/after-norm-swap test above would share on both sides."""
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    torch.manual_seed(0)
    text = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "max_position_embeddings": 128,
    }
    vision = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "out_hidden_size": 32,
        "depth": 2,
        "num_heads": 2,
        "patch_size": 4,
        "temporal_patch_size": 2,
        "spatial_merge_size": 1,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
    }

    def build(num_layers: int) -> Qwen3VLForConditionalGeneration:
        config = Qwen3VLConfig(
            text_config={**text, "num_hidden_layers": num_layers},
            vision_config=dict(vision),
            tie_word_embeddings=True,
        )
        return Qwen3VLForConditionalGeneration(config).eval()

    full = build(4)
    truncated = build(2)

    # Copy the shared weights (embeddings, first 2 layers, vision tower) from the full model.
    full_sd = full.state_dict()
    truncated.load_state_dict({k: full_sd[k].clone() for k in truncated.state_dict()})
    truncated.model.language_model.norm = torch.nn.Identity()

    n = 2
    input_ids = torch.randint(0, 60, (1, 9))
    with torch.no_grad():
        full_out = full.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        truncated_out = truncated.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)

    assert torch.equal(full_out.hidden_states[n], truncated_out.hidden_states[n])

    # And a truncated model whose REAL final norm survived would silently corrupt the tensor.
    truncated_real_norm = build(2)
    truncated_real_norm.load_state_dict({k: full_sd[k].clone() for k in truncated_real_norm.state_dict()})
    with torch.no_grad():
        real_norm_out = truncated_real_norm.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    assert not torch.equal(real_norm_out.hidden_states[n], full_out.hidden_states[n])


def test_te_converted_keys_match_real_model_exactly() -> None:
    """Dress the real int8 file's stripped header (meta tensors), convert it, and pin that the
    loader's constructed model (bundled config truncated to 50 layers, Identity final norm,
    Int8ConvrotLinear swaps) expects exactly the converted keys plus the tied lm_head."""
    from pathlib import Path

    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    from invokeai.backend.minimax_h3.int8_convrot import Int8ConvrotLinear
    from invokeai.backend.model_manager.util.qwen3_vl import normalize_qwen3vl_rope_config
    from tests.model_identification.stripped_model_on_disk import StrippedModelOnDisk

    fixture_dir = (
        Path(__file__).parents[2] / "model_identification" / "stripped_models" / "d01a4af4-c438-420e-88ea-a3b431a3152f"
    )
    fixture = fixture_dir / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    sd = StrippedModelOnDisk.load_stripped_model(fixture)
    # The stripped fixture dresses every tensor as meta (no data); the marker blobs need real
    # bytes for the converter to parse. All 350 are identical in the real file.
    for key in list(sd):
        if key.endswith(".comfy_quant"):
            sd[key] = _marker_blob()

    metadata = json.loads(json.load(open(fixture))["metadata_key_for_stripped_models"]["minimax_h3_te"])
    assert metadata["num_hidden_layers"] == MINIMAX_H3_TEXT_ENCODER_LAYER

    converted, markers = convert_minimax_h3_text_encoder_checkpoint(sd)
    assert len(markers) == 350
    assert all(m["format"] == "int8_tensorwise" and m["convrot"] for m in markers.values())

    bundled = (
        Path(__file__).parents[3] / "invokeai" / "backend" / "minimax_h3" / "qwen3vl_32b_h3_text_encoder_config.json"
    )
    config_dict = json.loads(bundled.read_text(encoding="utf-8"))
    config_dict["text_config"]["num_hidden_layers"] = metadata["num_hidden_layers"]
    config_dict["tie_word_embeddings"] = True
    te_config = normalize_qwen3vl_rope_config(Qwen3VLConfig.from_dict(config_dict))

    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration._from_config(te_config)
    model.model.language_model.norm = torch.nn.Identity()
    for module_name in markers:
        parent = model.get_submodule(module_name.rsplit(".", 1)[0])
        weight = converted[module_name + ".weight"]
        setattr(
            parent,
            module_name.rsplit(".", 1)[1],
            Int8ConvrotLinear(
                weight=torch.zeros(weight.shape, dtype=torch.int8),
                weight_scale=torch.zeros(weight.shape[0], 1),
                convrot=True,
            ),
        )

    expected = set(model.state_dict().keys()) - {"lm_head.weight"}
    got = set(converted.keys())
    assert got == expected, (sorted(got - expected)[:5], sorted(expected - got)[:5])

    model_sd = model.state_dict()
    for key, tensor in converted.items():
        assert tuple(model_sd[key].shape) == tuple(tensor.shape), key


def test_probe_requires_structural_keys_even_with_metadata(tmp_path) -> None:
    """A re-tagged arbitrary file carrying the minimax_h3_te metadata must NOT identify - the
    probe requires the structural key minimum on both the metadata and fallback paths, so junk
    fails at identification instead of after a 25 GiB load."""
    from safetensors.torch import save_file

    from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
    from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
        Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config,
    )

    cls_name = Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config.__name__
    metadata = {"minimax_h3_te": '{"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}'}

    junk = tmp_path / "retagged_junk.safetensors"
    save_file({"some.random.weight": torch.zeros(4, 4)}, junk, metadata=metadata)
    result = ModelConfigFactory.from_model_on_disk(junk)
    assert not isinstance(result.details[cls_name], Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config)
    assert "does not look like" in str(result.details[cls_name])

    plausible = tmp_path / "plausible.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 4),
            "visual.blocks.0.attn.qkv.weight": torch.zeros(8, 4),
        },
        plausible,
        metadata=metadata,
    )
    result = ModelConfigFactory.from_model_on_disk(plausible)
    # Assert against THIS config's own attempt: a two-key synthetic also satisfies earlier,
    # more generic encoder configs in the union, so the overall winner is order-dependent for
    # synthetic files. The real file's end-to-end winner is pinned by the identification
    # fixture (d01a4af4-...), which lands on this config.
    plausible_attempt = result.details[cls_name]
    assert isinstance(plausible_attempt, Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config)
    assert plausible_attempt.base.value == "minimax-h3"

    # Without metadata, the structural fallback additionally demands the 32B shape.
    stripped = tmp_path / "stripped.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 4),
            "visual.blocks.0.attn.qkv.weight": torch.zeros(8, 4),
            "model.embed_tokens.weight": torch.zeros(8, 4),
        },
        stripped,
    )
    result = ModelConfigFactory.from_model_on_disk(stripped)
    assert not isinstance(result.details[cls_name], Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config)
