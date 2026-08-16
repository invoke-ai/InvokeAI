"""Tests for the MiniMax H3 packed-sequence math and denoise-state construction."""

import pytest
import torch

from invokeai.backend.minimax_h3.packing import (
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from invokeai.backend.minimax_h3.sampling import build_denoise_state, validate_canvas, validate_num_frames


class TestCanvas:
    def test_every_resolvable_canvas_is_accepted(self):
        # The guard must never reject a canvas resolve_canvas_size can emit. Rounding both axes to
        # the 32 grid happens *after* the area cap, so its own outputs reach ~1.04x the cap.
        canvases = set()
        for index in range(1501):
            ratio = min(0.25 + index * (3.75 / 1500), 4.0)
            canvases.add(resolve_canvas_size(ratio, 1.0))
        assert len(canvases) > 50
        for height, width in canvases:
            validate_canvas(height, width)

    def test_native_canvas_accepted(self):
        validate_canvas(768, 1344)
        validate_canvas(1344, 768)

    def test_oversized_canvas_rejected(self):
        # 4096x4096 builds a ~607k-row packed sequence against the native canvas's ~38k.
        with pytest.raises(ValueError, match="soft area cap"):
            validate_canvas(4096, 4096)
        with pytest.raises(ValueError, match="soft area cap"):
            validate_canvas(2048, 2048)

    def test_off_grid_canvas_rejected(self):
        with pytest.raises(ValueError, match="multiples of 32"):
            validate_canvas(768, 1000)

    def test_extreme_aspect_ratio_rejected(self):
        with pytest.raises(ValueError, match="aspect ratios"):
            validate_canvas(128, 1344)

    def test_nonpositive_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_canvas(0, 1344)


class TestFrameGrid:
    def test_legal_frame_counts_pass(self):
        # 345 = 17*20+5 (14.375 s) is the longest legal clip: 362 rounds past the 15 s cap.
        for n in (5, 124, 141, 345):
            validate_num_frames(n)

    def test_misaligned_frame_counts_rejected(self):
        with pytest.raises(ValueError, match="17"):
            validate_num_frames(6)
        with pytest.raises(ValueError, match="17"):
            validate_num_frames(121)

    def test_durations_outside_window_rejected(self):
        # 22/39/107 frames are grid-aligned but below the 5 s floor (and not the still path).
        for n in (22, 39, 107):
            with pytest.raises(ValueError, match="seconds"):
                validate_num_frames(n)
        # 362 = 17*21+5 is aligned but 15.083 s > 15 s.
        with pytest.raises(ValueError, match="seconds"):
            validate_num_frames(362)

    def test_still_image_minimum_allowed(self):
        validate_num_frames(5)

    def test_align_num_frames(self):
        assert align_num_frames(121) == 124
        assert align_num_frames(5) == 5
        assert align_num_frames(1) == 5

    def test_latent_frame_math(self):
        assert video_latent_num_frames(5) == 2
        assert video_latent_num_frames(124) == 37
        with pytest.raises(ValueError):
            video_latent_num_frames(6)

    def test_audio_latent_math(self):
        # 40 latents/s at 24 fps.
        assert audio_latent_num_frames(24) == 40
        assert audio_latent_num_frames(124) == round(124 / 24 * 40)


def _build_state(seed: int, with_keyframe: bool = False):
    kwargs = {}
    if with_keyframe:
        # One keyframe -> rows_per_frame = (4//2) * (4//2) = 4 condition rows of width 96.
        kwargs = {
            "keyframe_anchors": ("first",),
            "clean_condition_rows": torch.zeros(4, 96),
        }
    return build_denoise_state(
        text_token_tags=torch.tensor([1, 1, 0], dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=8,
        num_inference_steps=4,
        seed=seed,
        device=torch.device("cpu"),
        **kwargs,
    )


class TestDenoiseState:
    def test_shapes_and_layout(self):
        state = _build_state(seed=123)
        # 2 latent frames x (4/2 * 4/2) rows/frame = 8 video rows, patch dim 24*1*2*2 = 96.
        assert state.video_rows.shape == (8, 96)
        # 8 audio latents x 2 stereo channels, 32 channels each.
        assert state.audio_rows.shape == (16, 32)
        assert state.layout.num_condition_video_rows == 0
        assert state.layout.sequence_length == 3 + 0 + 16 + 8
        # num_inference_steps counts sigma grid points (terminal included): N -> N-1 evals.
        assert len(state.row_timestep_plan) == 3
        assert len(state.timesteps) == 3

    def test_noise_is_deterministic_per_seed(self):
        a, b = _build_state(seed=7), _build_state(seed=7)
        assert torch.equal(a.video_rows, b.video_rows)
        assert torch.equal(a.audio_rows, b.audio_rows)
        c = _build_state(seed=8)
        assert not torch.equal(a.video_rows, c.video_rows)
        assert not torch.equal(a.audio_rows, c.audio_rows)

    def test_keyframe_draw_shifts_generated_noise(self):
        # The keyframe conditioning noise is the generator's FIRST draw, so the video/audio
        # draws of a keyframed request differ from an unkeyframed one at the same seed.
        plain = _build_state(seed=7)
        keyframed = _build_state(seed=7, with_keyframe=True)
        assert keyframed.layout.num_condition_video_rows == 4
        assert keyframed.video_rows.shape == (12, 96)  # 4 condition rows + 8 generated
        assert not torch.equal(keyframed.video_rows[4:], plain.video_rows)

    def test_condition_rows_are_noise_augmented(self):
        keyframed = _build_state(seed=7, with_keyframe=True)
        # Clean rows were zeros; scale_noise(x0=0, t=0.999, noise) = 0.001 * noise != 0.
        condition = keyframed.video_rows[:4]
        assert not torch.equal(condition, torch.zeros_like(condition))
        assert condition.abs().max() < 0.1  # 0.1% of unit-normal noise

    def test_mismatched_condition_rows_rejected(self):
        with pytest.raises(ValueError, match="rows"):
            build_denoise_state(
                text_token_tags=torch.tensor([1], dtype=torch.long),
                num_latent_frames=2,
                latent_height=4,
                latent_width=4,
                num_audio_latents=8,
                num_inference_steps=2,
                seed=0,
                device=torch.device("cpu"),
                keyframe_anchors=("first",),
                clean_condition_rows=torch.zeros(3, 96),  # wrong: layout expects 4
            )

    def test_anchors_and_rows_must_travel_together(self):
        with pytest.raises(ValueError, match="together"):
            build_denoise_state(
                text_token_tags=torch.tensor([1], dtype=torch.long),
                num_latent_frames=2,
                latent_height=4,
                latent_width=4,
                num_audio_latents=8,
                num_inference_steps=2,
                seed=0,
                device=torch.device("cpu"),
                keyframe_anchors=("first",),
                clean_condition_rows=None,
            )
