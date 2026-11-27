"""Cancellation inside the MiniMax H3 video VAE decode, on a micro-config CPU autoencoder.

A decode is one `vae.decode` call that runs the decoder once per spatial tile of every temporal
chunk. A cancel must land at the next tile rather than after the whole decode, and must leave no
hooks on the shared model.
"""

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.vae.minimax_h3_latents_to_video import decode_video_latents
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

LATENT_CHANNELS = 4
# 7 latent frames (5n+2, one chunk) at 20x20 latents = 320x320 px = four 256-px tiles.
LATENTS_SHAPE = (1, LATENT_CHANNELS, 7, 20, 20)
NUM_TILES = 4


@pytest.fixture
def vae() -> AutoencoderKLMiniMaxH3:
    torch.manual_seed(0)
    return AutoencoderKLMiniMaxH3(
        latent_channels=LATENT_CHANNELS,
        block_out_channels=(8, 8, 8, 8, 8, 8),
        norm_num_groups=8,
        layers_per_block=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        latents_mean=(0.0,) * LATENT_CHANNELS,
        latents_std=(1.0,) * LATENT_CHANNELS,
    ).eval()


def _context(vae: AutoencoderKLMiniMaxH3, polls: list[bool]) -> MagicMock:
    """A context whose model loader hands back `vae` on the CPU and whose cancel poll answers
    from `polls` in order."""
    context = MagicMock()
    context.models.load.return_value.model = vae
    context.models.load.return_value.model_on_device.return_value = nullcontext((None, vae))
    answers = iter(polls)
    context.util.is_canceled.side_effect = lambda: next(answers)
    return context


def _count_decoder_forwards(vae: AutoencoderKLMiniMaxH3) -> list[int]:
    count = [0]
    vae.decoder.register_forward_hook(lambda module, args, output: count.__setitem__(0, count[0] + 1))
    return count


def test_cancel_lands_within_a_tile(vae):
    decoder_forwards = _count_decoder_forwards(vae)
    # Polls: tile 0 (False), tile 1 (True); the remaining tiles never run.
    context = _context(vae, [False, True])
    with pytest.raises(CanceledException):
        decode_video_latents(context, MagicMock(), torch.randn(*LATENTS_SHAPE))
    assert decoder_forwards[0] == 1, "the decode did not stop at the first tile boundary after the cancel"
    assert not vae.decoder._forward_pre_hooks, "cancel hooks leaked onto the shared video VAE"


def test_uncanceled_decode_polls_per_tile_and_leaves_no_hooks(vae):
    decoder_forwards = _count_decoder_forwards(vae)
    context = _context(vae, [False] * NUM_TILES)
    decoded = decode_video_latents(context, MagicMock(), torch.randn(*LATENTS_SHAPE))
    assert decoder_forwards[0] == NUM_TILES
    assert context.util.is_canceled.call_count == NUM_TILES
    assert decoded.shape == (3, 22, 320, 320)
    assert decoded.device.type == "cpu"
    assert not vae.decoder._forward_pre_hooks
