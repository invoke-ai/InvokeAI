import math

import pytest
import torch

from invokeai.backend.pid._src.networks.pixeldit_official import PiTBlock


def _build_pit_block() -> PiTBlock:
    return PiTBlock(
        pixel_hidden_size=4,
        patch_hidden_size=8,
        patch_size=2,
        num_heads=2,
        mlp_ratio=2.0,
        attn_hidden_size=8,
        attn_num_heads=2,
    ).eval()


def test_pit_block_chunked_forward_matches_unchunked_and_bounds_adaln_batch() -> None:
    torch.manual_seed(0)
    unchunked = _build_pit_block()
    chunk_size = 3
    chunked = _build_pit_block()
    chunked.load_state_dict(unchunked.state_dict())

    batch_size = 2
    image_height = 4
    image_width = 4
    patch_size = 2
    patch_count = image_height * image_width // patch_size**2
    patch_batch = batch_size * patch_count
    pixels = torch.randn(patch_batch, patch_size**2, 4)
    condition = torch.randn(patch_batch, 8)

    adaln_batch_sizes: list[int] = []

    def record_adaln_batch_size(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        adaln_batch_sizes.append(inputs[0].shape[0])

    handle = chunked.adaLN_modulation.register_forward_pre_hook(record_adaln_batch_size)
    try:
        with torch.no_grad():
            expected = unchunked(pixels, condition, image_height, image_width, patch_size)
            actual = chunked(
                pixels,
                condition,
                image_height,
                image_width,
                patch_size,
                activation_chunk_size=chunk_size,
            )
    finally:
        handle.remove()

    torch.testing.assert_close(actual, expected)
    assert max(adaln_batch_sizes) <= chunk_size
    assert len(adaln_batch_sizes) == 2 * math.ceil(patch_batch / chunk_size)


def test_pit_block_rejects_non_positive_activation_chunk_size() -> None:
    block = _build_pit_block()
    pixels = torch.randn(4, 4, 4)
    condition = torch.randn(4, 8)
    with pytest.raises(ValueError, match="activation_chunk_size must be positive"):
        block(pixels, condition, image_height=4, image_width=4, patch_size=2, activation_chunk_size=0)


def test_pit_block_uses_unchunked_path_when_gradients_are_enabled() -> None:
    block = _build_pit_block()
    pixels = torch.randn(4, 4, 4, requires_grad=True)
    condition = torch.randn(4, 8)
    adaln_batch_sizes: list[int] = []

    def record_adaln_batch_size(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        adaln_batch_sizes.append(inputs[0].shape[0])

    handle = block.adaLN_modulation.register_forward_pre_hook(record_adaln_batch_size)
    try:
        output = block(
            pixels,
            condition,
            image_height=4,
            image_width=4,
            patch_size=2,
            activation_chunk_size=1,
        )
        output.sum().backward()
    finally:
        handle.remove()

    assert adaln_batch_sizes == [pixels.shape[0]]
    assert pixels.grad is not None


def test_pit_block_activation_chunking_is_not_sticky_between_calls() -> None:
    block = _build_pit_block()
    pixels = torch.randn(4, 4, 4)
    condition = torch.randn(4, 8)
    adaln_batch_sizes: list[int] = []

    def record_adaln_batch_size(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        adaln_batch_sizes.append(inputs[0].shape[0])

    handle = block.adaLN_modulation.register_forward_pre_hook(record_adaln_batch_size)
    try:
        with torch.no_grad():
            block(
                pixels,
                condition,
                image_height=4,
                image_width=4,
                patch_size=2,
                activation_chunk_size=1,
            )
            optimized_call_count = len(adaln_batch_sizes)
            block(pixels, condition, image_height=4, image_width=4, patch_size=2)
    finally:
        handle.remove()

    assert adaln_batch_sizes[:optimized_call_count] == [1] * 8
    assert adaln_batch_sizes[optimized_call_count:] == [pixels.shape[0]]
