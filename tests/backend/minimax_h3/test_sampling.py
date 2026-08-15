"""Tests for the MiniMax H3 packed-sequence math and denoise-state construction."""

import pytest
import torch

from invokeai.backend.minimax_h3.packing import (
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from invokeai.backend.minimax_h3.presets import MINIMAX_H3_VIDEO_FRAME_CHOICES, resolve_lowres_canvas_size
from invokeai.backend.minimax_h3.sampling import build_denoise_state, validate_num_frames


class TestFrameGrid:
    def test_legal_frame_counts_pass(self):
        # 90 (3.75 s) is the accepted floor for video; 345 = 17*20+5 (14.375 s) is the longest legal
        # clip, since 362 rounds past the 15 s cap.
        for n in (5, 90, 107, 124, 141, 345):
            validate_num_frames(n)

    def test_misaligned_frame_counts_rejected(self):
        with pytest.raises(ValueError, match="17"):
            validate_num_frames(6)
        with pytest.raises(ValueError, match="17"):
            validate_num_frames(121)

    def test_frame_counts_outside_the_accepted_range_rejected(self):
        # 22/39/56/73 are grid-aligned but below the 90-frame floor (and not the still path).
        for n in (22, 39, 56, 73):
            with pytest.raises(ValueError, match="frames on the 17n"):
                validate_num_frames(n)
        # 362 = 17*21+5 is aligned but 15.083 s > 15 s.
        with pytest.raises(ValueError, match="frames on the 17n"):
            validate_num_frames(362)

    def test_still_image_minimum_allowed(self):
        validate_num_frames(5)

    def test_every_offered_choice_validates(self):
        # The node's dropdown is built from this tuple; nothing in it may be rejected at invoke time.
        assert MINIMAX_H3_VIDEO_FRAME_CHOICES[0] == 90
        assert MINIMAX_H3_VIDEO_FRAME_CHOICES[-1] == 345
        for n in MINIMAX_H3_VIDEO_FRAME_CHOICES:
            validate_num_frames(n)

    def test_choices_are_exactly_the_grid_points_in_range(self):
        expected = tuple(n for n in range(90, 361) if n % 17 == 5)
        assert MINIMAX_H3_VIDEO_FRAME_CHOICES == expected

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


class TestLowresCanvasResolution:
    # The native short-edge-768 policy is covered by tests/app/invocations/test_minimax_h3_ideal_dimensions.py.

    def test_lowres_pins_the_long_edge(self):
        # Same aspect inputs, but the LONG edge is fixed at 768.
        assert resolve_lowres_canvas_size(1, 1) == (768, 768)
        # 2:3 portrait -> 512x768 (the user-facing "small test render" case).
        assert resolve_lowres_canvas_size(800, 1200) == (768, 512)
        assert resolve_lowres_canvas_size(1200, 800) == (512, 768)
        # 16:9 -> short edge 432 rounds to the nearest multiple of 32 (448).
        assert resolve_lowres_canvas_size(16, 9) == (448, 768)
        assert resolve_lowres_canvas_size(9, 16) == (768, 448)

    def test_lowres_never_exceeds_the_highres_area(self):
        for w, h in ((1, 1), (16, 9), (9, 16), (2, 3), (4, 1), (1, 4), (1024, 1000)):
            lo_h, lo_w = resolve_lowres_canvas_size(w, h)
            hi_h, hi_w = resolve_canvas_size(w, h)
            assert lo_h * lo_w <= hi_h * hi_w
            assert lo_h % 32 == 0 and lo_w % 32 == 0

    def test_extreme_aspect_ratios_stay_on_grid(self):
        # 4:1 is the limit of the supported envelope; 768/4 = 192 divides the grid exactly (6 * 32).
        assert resolve_lowres_canvas_size(4, 1) == (192, 768)
        assert resolve_lowres_canvas_size(1, 4) == (768, 192)

    def test_invalid_aspects_rejected_by_both_presets(self):
        for fn in (resolve_canvas_size, resolve_lowres_canvas_size):
            with pytest.raises(ValueError, match="positive"):
                fn(0, 100)
            with pytest.raises(ValueError, match="1:4"):
                fn(9, 1)


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
