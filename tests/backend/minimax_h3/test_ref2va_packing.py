"""Golden tests for the Ref2VA packed-sequence layout and denoise state.

The layout contract is ported from the merged diffusers MiniMax-H3 integration
(``modular_pipelines/minimax_h3/before_denoise.py::build_ref2va_packed_sequence`` at commit
f53d552036a0d1bd5570782a39cd40cfabf112bc); the micro-geometry expectations below are hand-derived
from that source. Run on CPU with tiny latents.
"""

import math

import pytest
import torch

from invokeai.backend.minimax_h3.packing import (
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3ReferenceLayoutEntry,
    _temporal_position_span,
    build_ref2va_packed_sequence,
    validate_reference_kinds,
)
from invokeai.backend.minimax_h3.sampling import (
    MiniMaxH3EncodedReference,
    build_denoise_state,
    build_ref2va_denoise_state,
)

TEXT_TAGS = torch.tensor([1, 1, 0], dtype=torch.long)

# Target micro-geometry shared by every test: 2 latent frames on a 4x4 latent canvas, 8 audio
# latents. rows_per_target_frame = (4//2) * (4//2) = 4, so 8 target video rows and 16 target
# audio rows.
TARGET = {"num_latent_frames": 2, "latent_height": 4, "latent_width": 4, "num_audio_latents": 8}


def _sequential_span(num_latent_frames: int) -> float:
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(num_latent_frames)
    )


def _layout(references, condition_latent_shapes, audio_condition_row_counts):
    return build_ref2va_packed_sequence(
        TEXT_TAGS,
        references,
        condition_latent_shapes,
        audio_condition_row_counts,
        patch_size=(1, 2, 2),
        **TARGET,
    )


IMAGE = MiniMaxH3ReferenceLayoutEntry(kind="image", has_audio=False)
VIDEO = MiniMaxH3ReferenceLayoutEntry(kind="video", has_audio=False)
VIDEO_AUDIO = MiniMaxH3ReferenceLayoutEntry(kind="video", has_audio=True)
AUDIO = MiniMaxH3ReferenceLayoutEntry(kind="audio", has_audio=True)


class TestValidateReferenceKinds:
    def test_accepts_ordered_mixes(self):
        validate_reference_kinds(["image", "video", "audio"])
        validate_reference_kinds(["video"])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one reference"):
            validate_reference_kinds([])

    def test_rejects_all_audio(self):
        with pytest.raises(ValueError, match="cannot be used alone"):
            validate_reference_kinds(["audio", "audio"])

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="got 'movie'"):
            validate_reference_kinds(["movie"])

    @pytest.mark.parametrize(
        ("kinds", "message"),
        [
            (["image"] * 10, "At most 9 image"),
            (["video"] * 4, "At most 3 video"),
            (["audio"] * 4 + ["image"], "At most 3 audio"),
        ],
    )
    def test_rejects_per_kind_caps(self, kinds, message):
        with pytest.raises(ValueError, match=message):
            validate_reference_kinds(kinds)


class TestRef2VALayout:
    def test_single_image_reference(self):
        # Image 1x6x2 -> (6//2) * (2//2) = 3 rows on its OWN aspect-normalized grid.
        layout = _layout([IMAGE], ((1, 6, 2),), ())

        assert layout.sequence_length == 3 + 3 + 16 + 8
        assert layout.num_condition_video_rows == 3
        assert layout.num_condition_audio_rows == 0
        # Reference video rows lead video_indices; targets follow after the target audio block.
        assert layout.video_indices.tolist() == [3, 4, 5] + list(range(3 + 3 + 16, layout.sequence_length))
        assert layout.audio_indices.tolist() == list(range(6, 22))
        # All image rows share ONE rotary time: the text length. The image consumes a single
        # integer slot, so target audio starts at num_text_tokens + 1.
        assert torch.all(layout.position_ids[3:6, 0] == 3.0)
        assert float(layout.position_ids[6, 0]) == 4.0
        # The image's (h, w) grid is its own (6x2 latent), not the target's 4x4: its rows'
        # height coordinates span 3 distinct values.
        assert len(set(layout.position_ids[3:6, 1].tolist())) == 3
        # Tags: text rows keep their tags, reference rows are video, audio rows audio.
        assert layout.token_tags[:3].tolist() == TEXT_TAGS.tolist()
        assert torch.all(layout.token_tags[3:6] == MINIMAX_H3_VIDEO_TAG)
        assert torch.all(layout.token_tags[6:22] == MINIMAX_H3_AUDIO_TAG)
        assert MINIMAX_H3_TEXT_TAG in layout.token_tags[:3].tolist()

    def test_standalone_audio_uses_target_width_grid(self):
        # [audio (6 rows = 3 latents), image 1x4x4]: the audio block borrows the TARGET width
        # grid; the image then starts at num_text_tokens + 3.
        layout = _layout([AUDIO, IMAGE], ((1, 4, 4),), (6,))

        assert layout.num_condition_audio_rows == 6
        assert layout.num_condition_video_rows == 4
        # Audio rows 3..8; image rows 9..12.
        assert layout.audio_indices[:6].tolist() == [3, 4, 5, 6, 7, 8]
        assert layout.video_indices[:4].tolist() == [9, 10, 11, 12]
        # Standalone audio at the target's width-grid extremes: identical w pins as the
        # generated audio rows (which use the same 4x4 target grid).
        target_audio_start = 13
        assert float(layout.position_ids[3, 2]) == float(layout.position_ids[target_audio_start, 2])
        assert float(layout.position_ids[8, 2]) == float(layout.position_ids[target_audio_start + 15, 2])
        # Audio advances the clock by its latent count (3): the image sits at time 3 + 3.
        assert torch.all(layout.position_ids[9:13, 0] == 6.0)
        # h stays 0 on audio rows.
        assert torch.all(layout.position_ids[3:9, 1] == 0.0)

    def test_video_reference_soundtrack_adjacency_and_clock(self):
        # [video+audio]: video 7 latent frames on a 2x4 latent grid (2 rows/frame -> 14 rows),
        # soundtrack 8 rows (4 latents). Soundtrack rows pack immediately BEFORE the video rows
        # and share its origin; the block advances the clock by max(4, span(7)).
        layout = _layout([VIDEO_AUDIO], ((7, 2, 4),), (8,))

        assert layout.num_condition_video_rows == 14
        assert layout.num_condition_audio_rows == 8
        assert layout.audio_indices[:8].tolist() == list(range(3, 11))
        assert layout.video_indices[:14].tolist() == list(range(11, 25))
        # Soundtrack and video share the origin (time 3.0 at their first latent).
        assert float(layout.position_ids[3, 0]) == 3.0
        assert float(layout.position_ids[11, 0]) == 3.0
        # The soundtrack is pinned to the VIDEO's own width grid, which differs from the
        # target's: compare against the first target audio row's pin.
        target_audio_start = 25
        assert float(layout.position_ids[3, 2]) != float(layout.position_ids[target_audio_start, 2])
        # Clock: span(7) = 5/3 * (1+4+4+4+4+1+4) = 36.67 > 4 audio latents, so the target
        # audio starts at 3 + span(7), summed SEQUENTIALLY.
        assert float(layout.position_ids[target_audio_start, 0]) == 3.0 + _sequential_span(7)

    def test_silent_video_reference_contributes_no_audio_rows(self):
        layout = _layout([VIDEO], ((2, 2, 4),), ())

        assert layout.num_condition_audio_rows == 0
        assert layout.num_condition_video_rows == 4
        assert layout.audio_indices.tolist() == list(range(3 + 4, 3 + 4 + 16))

    def test_full_mix_order_and_counts(self):
        # [image 1x6x2, video+audio 3x2x4 (audio 8 rows), audio 6 rows] - the hand-verified
        # composite: blocks pack in request order, targets last.
        layout = _layout([IMAGE, VIDEO_AUDIO, AUDIO], ((1, 6, 2), (3, 2, 4)), (8, 6))

        assert layout.sequence_length == 50
        assert layout.num_condition_video_rows == 9
        assert layout.num_condition_audio_rows == 14
        assert layout.video_indices[:9].tolist() == [3, 4, 5] + list(range(14, 20))
        assert layout.audio_indices[:14].tolist() == list(range(6, 14)) + list(range(20, 26))
        # Clock walk: image (+1.0) -> video block at 4.0, advancing by max(4, span(3)) =
        # span(3) = 5/3 * (1+4+4) = 15 -> audio ref at 19.0, 3 latents -> targets at 22.0.
        assert torch.all(layout.position_ids[3:6, 0] == 3.0)
        assert float(layout.position_ids[6, 0]) == 4.0  # soundtrack origin
        assert float(layout.position_ids[14, 0]) == 4.0  # video rows origin
        assert float(layout.position_ids[20, 0]) == 4.0 + _sequential_span(3)
        assert float(layout.position_ids[26, 0]) == 4.0 + _sequential_span(3) + 3.0

    def test_reference_order_changes_the_layout(self):
        a = _layout([IMAGE, VIDEO_AUDIO], ((1, 6, 2), (3, 2, 4)), (8,))
        b = _layout([VIDEO_AUDIO, IMAGE], ((3, 2, 4), (1, 6, 2)), (8,))

        assert a.sequence_length == b.sequence_length
        assert not torch.equal(a.position_ids, b.position_ids)

    @pytest.mark.parametrize("num_latent_frames", [2, 6, 7, 17, 18, 22])
    def test_video_span_is_summed_sequentially(self, num_latent_frames):
        # The ref2va clock sums a video reference's span sequentially in float64 - NOT the
        # numpy-pairwise order `_temporal_position_span` reproduces for the FL2VA "last"
        # anchor. The two differ in the last ulp for several of these counts (empirically
        # from n = 6 in this series); the builder must match the sequential sum bit-exactly.
        layout = _layout([VIDEO], ((num_latent_frames, 2, 4),), ())
        target_audio_start = 3 + num_latent_frames * 2

        assert float(layout.position_ids[target_audio_start, 0]) == 3.0 + _sequential_span(num_latent_frames)

    def test_the_two_summation_orders_really_differ(self):
        # Documents why both span implementations exist: numpy's pairwise sum and the
        # sequential sum disagree in the last ulp at n = 18 (among others).
        assert _temporal_position_span(18) != _sequential_span(18)


def _encoded_references(kinds=("image",)):
    """Build consistent encoded references for the state builder at the micro geometry."""
    refs = []
    for kind in kinds:
        if kind == "image":
            refs.append(MiniMaxH3EncodedReference(kind="image", video_rows=torch.zeros(3, 96), latent_shape=(1, 6, 2)))
        elif kind == "video":
            refs.append(MiniMaxH3EncodedReference(kind="video", video_rows=torch.zeros(6, 96), latent_shape=(3, 2, 4)))
        elif kind == "video_audio":
            refs.append(
                MiniMaxH3EncodedReference(
                    kind="video",
                    video_rows=torch.zeros(6, 96),
                    latent_shape=(3, 2, 4),
                    audio_rows=torch.full((8, 32), 0.25),
                )
            )
        elif kind == "audio":
            refs.append(MiniMaxH3EncodedReference(kind="audio", audio_rows=torch.full((6, 32), 0.5)))
    return refs


def _build_ref_state(seed: int, kinds=("image",)):
    return build_ref2va_denoise_state(
        text_token_tags=TEXT_TAGS,
        references=_encoded_references(kinds),
        num_inference_steps=4,
        seed=seed,
        device=torch.device("cpu"),
        **TARGET,
    )


class TestRef2VADenoiseState:
    def test_shapes_and_layout(self):
        state = _build_ref_state(0, kinds=("image", "video_audio", "audio"))

        assert state.layout.num_condition_video_rows == 9
        assert state.layout.num_condition_audio_rows == 14
        assert state.video_rows.shape == (9 + 8, 96)
        assert state.audio_rows.shape == (14 + 16, 32)

    def test_determinism_per_seed(self):
        a = _build_ref_state(3, kinds=("image", "video_audio"))
        b = _build_ref_state(3, kinds=("image", "video_audio"))
        c = _build_ref_state(4, kinds=("image", "video_audio"))

        assert torch.equal(a.video_rows, b.video_rows)
        assert torch.equal(a.audio_rows, b.audio_rows)
        assert not torch.equal(a.video_rows, c.video_rows)

    def test_reference_audio_rows_stay_bitwise_clean(self):
        # Reference soundtracks are never noised: the leading audio rows equal the encoded
        # rows exactly, for any seed.
        state = _build_ref_state(9, kinds=("video_audio", "audio"))

        assert torch.equal(state.audio_rows[:8], torch.full((8, 32), 0.25))
        assert torch.equal(state.audio_rows[8:14], torch.full((6, 32), 0.5))

    def test_visual_condition_rows_are_noised(self):
        # Visual reference rows ARE noise-augmented (to t = 0.999): zero input rows come out
        # non-zero.
        state = _build_ref_state(9, kinds=("image",))

        assert not torch.all(state.video_rows[:3] == 0)

    def test_condition_draws_precede_generated_draws(self):
        # The per-reference condition draws come off the generator FIRST, so the generated
        # rows differ from a reference-free FL2VA request at the same seed.
        ref_state = _build_ref_state(5, kinds=("image",))
        plain_state = build_denoise_state(
            text_token_tags=TEXT_TAGS,
            num_inference_steps=4,
            seed=5,
            device=torch.device("cpu"),
            **TARGET,
        )

        assert not torch.equal(ref_state.video_rows[3:], plain_state.video_rows)

    def test_swapping_references_permutes_the_noise(self):
        a = _build_ref_state(6, kinds=("image", "video"))
        b = _build_ref_state(6, kinds=("video", "image"))

        # Same total rows, different packing order -> different leading rows.
        assert a.video_rows.shape == b.video_rows.shape
        assert not torch.equal(a.video_rows, b.video_rows)

    def test_timestep_plan_pins_reference_audio_clean(self):
        state = _build_ref_state(0, kinds=("video_audio",))

        for unique_timesteps, timestep_indices in state.row_timestep_plan:
            row_timesteps = unique_timesteps[timestep_indices]
            # Reference audio rows sit at exactly 1.0 on every step...
            assert torch.all(row_timesteps[state.audio_indices[:8]] == 1.0)
            # ...and visual reference rows at >= 0.999.
            assert torch.all(row_timesteps[state.video_indices[:6]] >= 0.999)

    def test_rejects_all_audio(self):
        with pytest.raises(ValueError, match="cannot be used alone"):
            _build_ref_state(0, kinds=("audio",))

    def test_rejects_row_shape_mismatch(self):
        refs = [MiniMaxH3EncodedReference(kind="image", video_rows=torch.zeros(5, 96), latent_shape=(1, 6, 2))]
        with pytest.raises(ValueError, match="packs into 3"):
            build_ref2va_denoise_state(
                text_token_tags=TEXT_TAGS,
                references=refs,
                num_inference_steps=4,
                seed=0,
                device=torch.device("cpu"),
                **TARGET,
            )

    def test_rejects_inconsistent_kinds(self):
        audio_with_video_rows = [
            MiniMaxH3EncodedReference(kind="audio", video_rows=torch.zeros(3, 96), audio_rows=torch.zeros(6, 32)),
            MiniMaxH3EncodedReference(kind="image", video_rows=torch.zeros(3, 96), latent_shape=(1, 6, 2)),
        ]
        with pytest.raises(ValueError, match="audio-only but carries video rows"):
            build_ref2va_denoise_state(
                text_token_tags=TEXT_TAGS,
                references=audio_with_video_rows,
                num_inference_steps=4,
                seed=0,
                device=torch.device("cpu"),
                **TARGET,
            )

    def test_fl2va_state_is_frozen(self):
        # Guards the FL2VA path against drift from any shared-code change: a fixed-seed
        # request's noise must stay put. If this moves beyond the tolerance, existing FL2VA
        # seeds change - do not update the golden without understanding why.
        #
        # The checksum is math.fsum (exact, reduction-order-independent) over the float32
        # elements, but the elements themselves are NOT bit-identical across platforms:
        # torch's seeded CPU normal draws differ in the last ulp between x86/arm64 and
        # Linux/macOS/Windows libms (observed spreads up to ~2e-7 relative on CI). The
        # tolerance absorbs exactly that platform variance and nothing more - any real
        # regression this test guards against (a reordered/extra draw, a changed noise-aug
        # level) moves the sum by O(1).
        state = build_denoise_state(
            text_token_tags=TEXT_TAGS,
            num_inference_steps=4,
            seed=1234,
            device=torch.device("cpu"),
            **TARGET,
        )
        checksum = math.fsum(state.video_rows.flatten().tolist()) + math.fsum(state.audio_rows.flatten().tolist())

        assert checksum == pytest.approx(-77.11209162930027, rel=1e-6)
