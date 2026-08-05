"""Tests for the MiniMax H3 denoising loop, on a micro-config CPU transformer."""

import pytest
import torch

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.minimax_h3.denoise import denoise
from invokeai.backend.minimax_h3.sampling import build_denoise_state
from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

TINY_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 32,
    "num_layers": 1,
    "num_refiner_layers": 1,
    "ffn_dim": 64,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 8,
    "freq_dim": 16,
    "time_embed_hidden_dim": 32,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
    "rope_theta": 10000.0,
}

# Grid points passed to the scheduler; the loop runs NUM_STEPS - 1 model evaluations
# (num_inference_steps counts sigma grid points, terminal included).
NUM_STEPS = 3
NUM_EVALS = NUM_STEPS - 1
TEXT_TAGS = torch.tensor([1, 1, 0], dtype=torch.long)


@pytest.fixture(scope="module")
def tiny_transformer() -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(**TINY_CONFIG)
    model.eval()
    return model


def _state(with_keyframe: bool = False):
    kwargs = {}
    if with_keyframe:
        kwargs = {"keyframe_anchors": ("first",), "clean_condition_rows": torch.zeros(4, 96)}
    return build_denoise_state(
        text_token_tags=TEXT_TAGS,
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=8,
        num_inference_steps=NUM_STEPS,
        seed=42,
        device=torch.device("cpu"),
        **kwargs,
    )


def test_denoise_output_shapes(tiny_transformer):
    state = _state()
    prompt_embeds = torch.randn(1, 3, TINY_CONFIG["text_dim"], generator=torch.Generator().manual_seed(1))
    video_rows, audio_rows = denoise(tiny_transformer, state, prompt_embeds)
    assert video_rows.shape == (8, 96)
    assert audio_rows.shape == (16, 32)
    assert torch.isfinite(video_rows).all()
    assert torch.isfinite(audio_rows).all()


def test_conditioning_rows_survive_the_loop(tiny_transformer):
    state = _state(with_keyframe=True)
    anchors_before = state.video_rows[:4].clone()
    prompt_embeds = torch.randn(1, 3, TINY_CONFIG["text_dim"], generator=torch.Generator().manual_seed(1))
    video_rows, _ = denoise(tiny_transformer, state, prompt_embeds)
    # The loop only writes generated rows; the noise-augmented anchors are untouched.
    assert torch.equal(video_rows[:4], anchors_before)
    assert not torch.equal(video_rows[4:], state.video_rows[4:] * 0)


def test_step_callback_called_per_step(tiny_transformer):
    state = _state()
    calls: list[tuple[int, int]] = []
    prompt_embeds = torch.randn(1, 3, TINY_CONFIG["text_dim"])
    denoise(
        tiny_transformer,
        state,
        prompt_embeds,
        step_callback=lambda step, total, rows: calls.append((step, total)),
    )
    assert calls == [(i + 1, NUM_EVALS) for i in range(NUM_EVALS)]


def test_step_callback_receives_pred_x0_of_generated_rows(tiny_transformer):
    """The callback gets the x-hat-0 estimate of the GENERATED video rows: condition rows are
    excluded, dtype is float32, and at the final step (sigma_next = 0) the Euler update reduces
    to x_next = x0, so the last callback payload must equal the returned generated rows."""
    state = _state(with_keyframe=True)
    num_condition_rows = state.layout.num_condition_video_rows
    num_generated_rows = state.video_rows.shape[0] - num_condition_rows
    seen: list[torch.Tensor] = []
    prompt_embeds = torch.randn(1, 3, TINY_CONFIG["text_dim"], generator=torch.Generator().manual_seed(3))
    video_rows, _ = denoise(
        tiny_transformer,
        state,
        prompt_embeds,
        step_callback=lambda step, total, rows: seen.append(rows.clone()),
    )
    assert all(rows.shape == (num_generated_rows, video_rows.shape[1]) for rows in seen)
    assert all(rows.dtype == torch.float32 for rows in seen)
    torch.testing.assert_close(seen[-1], video_rows[num_condition_rows:].to(torch.float32))


def test_cancellation_raises(tiny_transformer):
    state = _state()
    prompt_embeds = torch.randn(1, 3, TINY_CONFIG["text_dim"])
    polls = iter([False, True, True, True])
    with pytest.raises(CanceledException):
        denoise(tiny_transformer, state, prompt_embeds, is_canceled=lambda: next(polls))


def test_denoise_is_deterministic(tiny_transformer):
    prompt_embeds = torch.randn(1, 3, TINY_CONFIG["text_dim"], generator=torch.Generator().manual_seed(2))
    v1, a1 = denoise(tiny_transformer, _state(), prompt_embeds.clone())
    v2, a2 = denoise(tiny_transformer, _state(), prompt_embeds.clone())
    assert torch.equal(v1, v2)
    assert torch.equal(a1, a2)


def _layout(sequence_length: int, num_pad_rows: int):
    from invokeai.backend.minimax_h3.packing import MiniMaxH3PackedSequence

    token_tags = torch.ones(sequence_length, dtype=torch.long)
    if num_pad_rows:
        token_tags[-num_pad_rows:] = -1
    empty = torch.empty(0, dtype=torch.long)
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=torch.zeros(sequence_length, 3, dtype=torch.float64),
        token_tags=token_tags,
        video_indices=empty,
        audio_indices=empty,
        text_indices=empty,
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
    )


def test_denoise_working_memory_estimate():
    """Pin the reservation formula: per-row activations plus a fixed base, and the padding-mask
    term (bool mask + SDPA's additive-copy and alignment-pad transients = 5 bytes/entry) added
    only when padding rows exist. The band assertion sanity-checks the default linear-UI video
    (124 frames at 768x1344 = ~38k rows): far above the small cache default, below the size of
    the weights themselves. Change the constants deliberately - this test is meant to fail on
    accidental drift."""
    from invokeai.app.invocations.minimax_h3_denoise import MiniMaxH3DenoiseInvocation

    estimate = MiniMaxH3DenoiseInvocation._estimate_working_memory

    MB = 1024**2
    GB = 1024**3
    per_row_bytes = int(0.25 * MB)
    base_bytes = 2 * GB

    assert estimate(_layout(2_000, 0)) == 2_000 * per_row_bytes + base_bytes

    large = estimate(_layout(38_000, 0))
    assert large == 38_000 * per_row_bytes + base_bytes
    assert 8 * GB < large < 20 * GB

    padded = estimate(_layout(38_000, 4))
    assert padded == large + 5 * 38_000**2
