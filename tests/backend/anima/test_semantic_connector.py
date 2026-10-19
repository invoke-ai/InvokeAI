"""Anima-3.8B's semantic connector: layout against the real bundle, and the behaviour the port relies on.

The port was checked numerically against the reference implementation on the real weights (bit-identical
in fp32). These tests pin what that check cannot keep pinned: that the module tree still matches the
checkpoint's key layout, that the header is read into the right hyperparameters, and the edge cases the
denoise loop depends on -- the timestep dependence, and an empty prompt's fully masked Qwen3.5 source.
"""

import re

import accelerate
import pytest
import torch

from invokeai.backend.anima.anima_transformer import AnimaTransformer, LLMAdapter, masked_sdpa
from invokeai.backend.anima.semantic_connector import AnimaSemanticConnector, AnimaSemanticConnectorConfig
from invokeai.backend.model_manager.load.model_loaders.anima import (
    ANIMA_TRANSFORMER_CONFIG,
    _filter_non_model_keys,
    _strip_anima_bundle_prefix,
    anima_transformer_config,
)
from tests.backend.model_manager.load.state_dicts.anima_2_9b_keys import state_dict_keys as anima_2_9b_keys
from tests.backend.model_manager.load.state_dicts.anima_3_8b_connector_keys import (
    ANIMA_3_8B_NUM_BLOCKS,
    connector_keys,
    metadata,
)

_CONNECTOR_PREFIX = "net.anima_v2_connector."
_BLOCK_KEY = re.compile(r"^net\.blocks\.(\d+)\.(.+)$")


class TestConfigFromMetadata:
    def test_real_bundle_header(self) -> None:
        config = AnimaSemanticConnectorConfig.from_metadata(metadata)
        assert config == AnimaSemanticConnectorConfig(
            num_queries=64,
            resampler_blocks=6,
            resampler_dim=2048,
            resampler_heads=16,
            mlp_hidden_dim=5632,
            layer_indices=(7, 15, 23, 31),
        )

    def test_a_header_without_connector_keys_gets_the_trained_defaults(self) -> None:
        assert AnimaSemanticConnectorConfig.from_metadata({}) == AnimaSemanticConnectorConfig()

    def test_a_different_connector_architecture_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Anima semantic connector"):
            AnimaSemanticConnectorConfig.from_metadata(
                {"anima_v2_adapter_architecture": "anima_progressive_qwen35_cross_adapter_v1"}
            )


def test_connector_module_tree_matches_the_real_bundle() -> None:
    with accelerate.init_empty_weights():
        connector = AnimaSemanticConnector(AnimaSemanticConnectorConfig.from_metadata(metadata))
    expected = {k.removeprefix(_CONNECTOR_PREFIX): shape for k, shape in connector_keys.items()}
    actual = {k: list(v.shape) for k, v in connector.state_dict().items()}
    assert actual == expected


def test_transformer_built_from_the_bundle_takes_every_key() -> None:
    """52 blocks plus the connector, from the key layout the loader sees after its prefix passes."""
    block_suffixes = {
        m.group(2): shape for k, shape in anima_2_9b_keys.items() if (m := _BLOCK_KEY.match(k)) and m.group(1) == "0"
    }
    sd = {k: torch.empty(s, device="meta") for k, s in anima_2_9b_keys.items() if not _BLOCK_KEY.match(k)}
    for index in range(ANIMA_3_8B_NUM_BLOCKS):
        for suffix, shape in block_suffixes.items():
            sd[f"net.blocks.{index}.{suffix}"] = torch.empty(shape, device="meta")
    sd.update({k: torch.empty(s, device="meta") for k, s in connector_keys.items()})
    sd = _filter_non_model_keys(_strip_anima_bundle_prefix(sd))

    with accelerate.init_empty_weights():
        model = AnimaTransformer(
            **anima_transformer_config(sd), semantic_connector=AnimaSemanticConnectorConfig.from_metadata(metadata)
        )
    result = model.load_state_dict(sd, strict=False, assign=True)

    assert len(model.blocks) == ANIMA_3_8B_NUM_BLOCKS
    assert model.has_semantic_connector
    assert result.unexpected_keys == []
    assert result.missing_keys == []


def test_fp8_storage_casts_the_whole_connector() -> None:
    """No connector module is kept out of FP8 Storage: keeping its timestep path in bf16 was measured to change
    nothing (see `AnimaTransformer.__init__`), so a model with the connector declares the same patterns as one
    without."""
    with accelerate.init_empty_weights():
        expanded = AnimaTransformer(**ANIMA_TRANSFORMER_CONFIG, semantic_connector=AnimaSemanticConnectorConfig())
    assert expanded.has_semantic_connector
    assert expanded._skip_layerwise_casting_patterns is AnimaTransformer._skip_layerwise_casting_patterns
    kept = [
        name
        for name, module in expanded.named_modules()
        if name.startswith("anima_v2_connector.")
        and isinstance(module, torch.nn.Linear)
        and any(re.search(p, name) for p in expanded._skip_layerwise_casting_patterns)
    ]
    assert kept == []


class TestMaskedSdpa:
    def test_rows_with_keys_match_plain_sdpa(self) -> None:
        q, k, v = (torch.randn(1, 2, 3, 8) for _ in range(3))
        mask = torch.tensor([True, False, True, True]).reshape(1, 1, 1, 4)
        k, v = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert torch.allclose(masked_sdpa(q, k, v, mask), expected)

    def test_a_fully_masked_row_attends_to_nothing(self) -> None:
        q, k, v = torch.randn(2, 2, 3, 8), torch.randn(2, 2, 4, 8), torch.randn(2, 2, 4, 8)
        mask = torch.tensor([[True] * 4, [False] * 4]).reshape(2, 1, 1, 4)
        out = masked_sdpa(q, k, v, mask)
        assert torch.equal(out[1], torch.zeros_like(out[1]))
        assert torch.allclose(out[0], torch.nn.functional.scaled_dot_product_attention(q[:1], k[:1], v[:1]))


def _tiny_connector() -> tuple[LLMAdapter, AnimaSemanticConnector]:
    torch.manual_seed(0)
    config = AnimaSemanticConnectorConfig(
        num_queries=4,
        resampler_blocks=2,
        resampler_dim=32,
        resampler_heads=4,
        mlp_hidden_dim=48,
        semantic_source_dim=24,
    )
    adapter = LLMAdapter(vocab_size=50, dim=32, num_layers=2, num_heads=4)
    connector = AnimaSemanticConnector(config, model_dim=32, num_heads=4, num_adapter_blocks=2)
    for parameter in [*adapter.parameters(), *connector.parameters()]:
        torch.nn.init.normal_(parameter, std=0.2)
    return adapter.eval(), connector.eval()


@torch.no_grad()
def test_connector_output_changes_with_the_timestep() -> None:
    adapter, connector = _tiny_connector()
    source, ids = torch.randn(1, 5, 32), torch.randint(0, 50, (1, 6))
    states = [torch.randn(1, 7, 24) for _ in range(4)]
    mask = torch.ones(1, 7, dtype=torch.bool)

    early = connector(adapter, source, ids, states, mask, torch.tensor([0.9]))
    late = connector(adapter, source, ids, states, mask, torch.tensor([0.1]))

    assert early.shape == (1, 6, 32)
    assert not torch.allclose(early, late)


@torch.no_grad()
def test_an_empty_prompt_contributes_no_qwen3_5_signal_through_the_anchor() -> None:
    """A fully masked source: finite output, and the anchor's states make no difference."""
    adapter, connector = _tiny_connector()
    source, ids = torch.randn(1, 5, 32), torch.randint(0, 50, (1, 6))
    mask = torch.zeros(1, 1, dtype=torch.bool)
    t = torch.tensor([0.5])

    one = connector(adapter, source, ids, [torch.randn(1, 1, 24) for _ in range(4)], mask, t)
    other = connector(adapter, source, ids, [torch.randn(1, 1, 24) for _ in range(4)], mask, t)

    assert torch.isfinite(one).all()
    # The resampler's own state still evolves (self-attention, MLP) but reads nothing from Qwen3.5 either.
    assert torch.allclose(one, other)


def test_a_transformer_with_the_connector_refuses_to_run_without_qwen3_5() -> None:
    with accelerate.init_empty_weights():
        model = AnimaTransformer(
            **{**ANIMA_TRANSFORMER_CONFIG, "num_blocks": 1}, semantic_connector=AnimaSemanticConnectorConfig()
        )
    with pytest.raises(ValueError, match="needs Qwen3.5 conditioning"):
        model.preprocess_text_embeds(torch.empty(1, 3, 1024), torch.zeros(1, 3, dtype=torch.long))
