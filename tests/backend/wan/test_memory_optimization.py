import math
from contextlib import nullcontext

import pytest
import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanTransformerBlock

from invokeai.backend.wan.memory_optimization import wan_memory_optimization


def _build_block() -> WanTransformerBlock:
    return WanTransformerBlock(
        dim=8,
        ffn_dim=16,
        num_heads=2,
        cross_attn_norm=True,
    ).eval()


class _Transformer(torch.nn.Module):
    def __init__(self, block: WanTransformerBlock) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([block])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        return self.blocks[0](hidden_states, encoder_hidden_states, temb, rotary_emb=None)


@pytest.mark.parametrize("per_token_timestep", [False, True])
@pytest.mark.parametrize("use_autocast", [False, True])
def test_wan_memory_optimization_matches_original_and_bounds_ffn_sequence(
    per_token_timestep: bool, use_autocast: bool
) -> None:
    torch.manual_seed(0)
    original = _Transformer(_build_block())
    optimized = _Transformer(_build_block())
    optimized.load_state_dict(original.state_dict())

    batch_size = 2
    sequence_length = 7
    hidden_states = torch.randn(batch_size, sequence_length, 8)
    encoder_hidden_states = torch.randn(batch_size, 5, 8)
    if per_token_timestep:
        temb = torch.randn(batch_size, sequence_length, 6, 8)
    else:
        temb = torch.randn(batch_size, 6, 8)

    chunk_size = 3
    ffn_sequence_lengths: list[int] = []

    def record_ffn_sequence_length(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        ffn_sequence_lengths.append(inputs[0].shape[1])

    handle = optimized.blocks[0].ffn.register_forward_pre_hook(record_ffn_sequence_length)
    try:
        autocast_context = torch.autocast("cpu", dtype=torch.bfloat16) if use_autocast else nullcontext()
        with torch.no_grad(), autocast_context:
            expected = original(hidden_states, encoder_hidden_states, temb)
            with wan_memory_optimization(optimized, enabled=True, activation_chunk_size=chunk_size):
                actual = optimized(hidden_states, encoder_hidden_states, temb)
    finally:
        handle.remove()

    torch.testing.assert_close(actual, expected)
    assert max(ffn_sequence_lengths) <= chunk_size
    assert len(ffn_sequence_lengths) == math.ceil(sequence_length / chunk_size)


def test_wan_memory_optimization_is_not_sticky_between_calls() -> None:
    transformer = _Transformer(_build_block())
    hidden_states = torch.randn(1, 5, 8)
    encoder_hidden_states = torch.randn(1, 3, 8)
    temb = torch.randn(1, 6, 8)
    ffn_sequence_lengths: list[int] = []

    def record_ffn_sequence_length(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        ffn_sequence_lengths.append(inputs[0].shape[1])

    handle = transformer.blocks[0].ffn.register_forward_pre_hook(record_ffn_sequence_length)
    try:
        with torch.no_grad():
            with wan_memory_optimization(transformer, enabled=True, activation_chunk_size=2):
                transformer(hidden_states, encoder_hidden_states, temb)
            optimized_call_count = len(ffn_sequence_lengths)
            transformer(hidden_states, encoder_hidden_states, temb)
    finally:
        handle.remove()

    assert ffn_sequence_lengths[:optimized_call_count] == [2, 2, 1]
    assert ffn_sequence_lengths[optimized_call_count:] == [hidden_states.shape[1]]


def test_wan_memory_optimization_restores_blocks_after_exception() -> None:
    transformer = _Transformer(_build_block())
    original_forward = transformer.blocks[0].forward

    with pytest.raises(RuntimeError, match="boom"):
        with wan_memory_optimization(transformer, enabled=True, activation_chunk_size=2):
            raise RuntimeError("boom")

    assert transformer.blocks[0].forward == original_forward


def test_wan_memory_optimization_rejects_nesting_without_corrupting_outer_context() -> None:
    transformer = _Transformer(_build_block())
    original_forward = transformer.blocks[0].forward

    with wan_memory_optimization(transformer, enabled=True, activation_chunk_size=2):
        optimized_forward = transformer.blocks[0].forward
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with wan_memory_optimization(transformer, enabled=True, activation_chunk_size=2):
                pass
        assert transformer.blocks[0].forward == optimized_forward

    assert transformer.blocks[0].forward == original_forward


def test_wan_memory_optimization_rejects_non_positive_chunk_size() -> None:
    transformer = _Transformer(_build_block())

    with pytest.raises(ValueError, match="activation_chunk_size must be positive"):
        with wan_memory_optimization(transformer, enabled=True, activation_chunk_size=0):
            pass


def test_wan_memory_optimization_uses_original_path_with_gradients() -> None:
    transformer = _Transformer(_build_block())
    hidden_states = torch.randn(1, 5, 8, requires_grad=True)
    encoder_hidden_states = torch.randn(1, 3, 8)
    temb = torch.randn(1, 6, 8)
    ffn_sequence_lengths: list[int] = []

    def record_ffn_sequence_length(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        ffn_sequence_lengths.append(inputs[0].shape[1])

    handle = transformer.blocks[0].ffn.register_forward_pre_hook(record_ffn_sequence_length)
    try:
        with wan_memory_optimization(transformer, enabled=True, activation_chunk_size=2):
            output = transformer(hidden_states, encoder_hidden_states, temb)
            output.sum().backward()
    finally:
        handle.remove()

    assert ffn_sequence_lengths == [hidden_states.shape[1]]
    assert hidden_states.grad is not None


def test_wan_memory_optimization_compacts_per_token_timesteps() -> None:
    torch.manual_seed(0)
    original = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=8,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=16,
    ).eval()
    optimized = WanTransformer3DModel.from_config(original.config).eval()
    optimized.load_state_dict(original.state_dict())

    hidden_states = torch.randn(1, 4, 3, 4, 4)
    sequence_length = 3 * 2 * 2
    timestep = torch.full((1, sequence_length), 500.0)
    timestep[:, :4] = 0
    encoder_hidden_states = torch.randn(1, 5, 16)
    embedded_timestep_counts: list[int] = []

    def record_timestep_count(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        embedded_timestep_counts.append(inputs[0].shape[0])

    handle = optimized.condition_embedder.time_embedder.register_forward_pre_hook(record_timestep_count)
    try:
        with torch.no_grad():
            expected = original(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            # A chunk larger than the sequence verifies that compact conditioning
            # remains valid when block activation chunking is not otherwise needed.
            with wan_memory_optimization(optimized, enabled=True, activation_chunk_size=100):
                actual = optimized(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
    finally:
        handle.remove()

    torch.testing.assert_close(actual, expected)
    assert embedded_timestep_counts == [2]
