"""Regression tests for VideoConcatInvocation._iter_joined_frames.

Covers two JPPhoto code-review findings (PR #9163):

1. ``fade_through_black`` claimed to emit ``transition_frames`` frames per boundary but
   used a symmetric ``tf // 2`` split, dropping one frame for odd ``tf``. The fix splits
   asymmetrically: ``tail_half = tf // 2`` consumed from the previous clip's tail,
   ``head_half = tf - tail_half`` from the next clip's head, so the emitted total is
   exactly ``tf`` for both parities.

2. The node fully decoded every input into RAM before encoding, so a long upload (the
   API admits 1 GB compressed) could expand to tens of gigabytes of frames. Joining is
   now a generator that pulls frames lazily and buffers only the transition windows —
   the streaming tests pin that output begins before the inputs are exhausted and that
   look-ahead stays bounded by ``transition_frames``.

We exercise ``_iter_joined_frames`` directly because it is the pure transformation
implementing the contract (the surrounding ``invoke()`` deals with imageio
encode/decode plumbing that isn't germane to the math).
"""

from typing import Iterator

import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_concat import MAX_TRANSITION_MEMORY_BYTES, VideoConcatInvocation, _crossfade
from invokeai.app.services.session_processor.session_processor_common import CanceledException


def _invocation(transition: str, transition_frames: int) -> VideoConcatInvocation:
    # ``videos`` requires min_length=2 to construct; values are unused by ``_iter_joined_frames``.
    return VideoConcatInvocation(
        videos=[VideoField(video_name="a"), VideoField(video_name="b")],
        transition=transition,  # type: ignore[arg-type]
        transition_frames=transition_frames,
    )


def _clip(value: int, n: int) -> list[np.ndarray]:
    return [np.full((4, 4, 3), value, dtype=np.uint8) for _ in range(n)]


class TestFadeThroughBlackOddTf:
    """fade_through_black must emit exactly tf frames per boundary, even for odd tf."""

    @pytest.mark.parametrize("tf", [1, 2, 3, 4, 5, 7, 8])
    def test_total_length_preserved(self, tf: int) -> None:
        clip_a = _clip(200, 10)
        clip_b = _clip(100, 10)
        v = _invocation("fade_through_black", tf)
        out = list(v._iter_joined_frames([clip_a, clip_b]))
        # fade_through_black: each boundary consumes tf frames (tail_half from A + head_half from B)
        # and emits exactly tf frames in their place, so the total length is preserved.
        assert len(out) == len(clip_a) + len(clip_b)

    def test_three_clip_chain(self) -> None:
        clip_a = _clip(200, 10)
        clip_b = _clip(150, 10)
        clip_c = _clip(100, 10)
        v = _invocation("fade_through_black", 3)
        out = list(v._iter_joined_frames([clip_a, clip_b, clip_c]))
        # 30 input frames - 2 boundaries each consuming 3 + emitting 3 = 30 output frames.
        assert len(out) == 30


class TestFadeThroughBlackTransitionFrames:
    """Validation should accept odd tf when both halves fit within their adjacent clips."""

    def test_odd_tf_validation_uses_correct_halves(self) -> None:
        # tf=5 → tail_half=2, head_half=3. A 5-frame clip would fail with the previous
        # symmetric split (2+2 ≤ 5 was fine, but the math now needs head_half=3 from
        # clip[1] head + tail_half=2 from clip[1] tail = 5 ≤ 5 → still fits).
        clip_a = _clip(200, 5)
        clip_b = _clip(100, 5)
        clip_c = _clip(50, 5)
        v = _invocation("fade_through_black", 5)
        out = list(v._iter_joined_frames([clip_a, clip_b, clip_c]))
        assert len(out) == 15


class TestCrossfadeTransitionLength:
    """Crossfade behaviour is unchanged; pin its length for safety."""

    def test_crossfade_shortens_by_tf_per_boundary(self) -> None:
        clip_a = _clip(200, 10)
        clip_b = _clip(100, 10)
        v = _invocation("crossfade", 5)
        out = list(v._iter_joined_frames([clip_a, clip_b]))
        # 20 input - 5 frames consumed from each side (10 total) + 5 blended emitted = 15.
        assert len(out) == 15


class TestCutNoTransitionFrames:
    """transition='cut' or tf=0 returns the raw concatenation."""

    def test_cut_concatenates_directly(self) -> None:
        clip_a = _clip(200, 4)
        clip_b = _clip(100, 6)
        v = _invocation("cut", 0)
        out = list(v._iter_joined_frames([clip_a, clip_b]))
        assert len(out) == 10

    def test_fade_through_black_tf_zero(self) -> None:
        clip_a = _clip(200, 4)
        clip_b = _clip(100, 6)
        v = _invocation("fade_through_black", 0)
        out = list(v._iter_joined_frames([clip_a, clip_b]))
        assert len(out) == 10


class TestStreamingJoin:
    """Joining must stream: output frames are yielded before the inputs are exhausted,
    and look-ahead into the decoders stays bounded by the transition window."""

    def _tracked_clip(self, value: int, n: int, pulled: list[int]) -> Iterator[np.ndarray]:
        def gen() -> Iterator[np.ndarray]:
            for _ in range(n):
                pulled[0] += 1
                yield np.full((4, 4, 3), value, dtype=np.uint8)

        return gen()

    def test_cut_emits_first_frame_after_single_pull(self) -> None:
        pulled = [0]
        v = _invocation("cut", 0)
        gen = v._iter_joined_frames([self._tracked_clip(200, 100, pulled), self._tracked_clip(100, 100, pulled)])
        next(gen)
        # Producing the first output frame must not decode the rest of either clip.
        assert pulled[0] == 1

    @pytest.mark.parametrize("transition", ["crossfade", "fade_through_black"])
    def test_lookahead_bounded_by_transition_window(self, transition: str) -> None:
        tf = 5
        pulled = [0]
        v = _invocation(transition, tf)
        gen = v._iter_joined_frames([self._tracked_clip(200, 60, pulled), self._tracked_clip(100, 60, pulled)])
        emitted = 0
        max_lookahead = 0
        for _ in gen:
            emitted += 1
            max_lookahead = max(max_lookahead, pulled[0] - emitted)
        # At any moment the generator holds at most the previous clip's tail window plus
        # the next clip's head window — never a whole decoded clip.
        assert max_lookahead <= 2 * tf + 1
        assert pulled[0] == 120  # every input frame was still consumed exactly once

    def test_streamed_output_matches_eager_reference(self) -> None:
        # The join must be frame-for-frame identical whether clips arrive as one-shot
        # iterators (the decoder) or materialized lists — guarding against any accidental
        # reliance on len()/indexing sneaking back into the streaming path.
        clip_a = [np.full((4, 4, 3), 20 + i, dtype=np.uint8) for i in range(12)]
        clip_b = [np.full((4, 4, 3), 120 + i, dtype=np.uint8) for i in range(12)]
        for transition, tf in [("cut", 0), ("crossfade", 4), ("fade_through_black", 5)]:
            v = _invocation(transition, tf)
            streamed = list(v._iter_joined_frames([iter(clip_a), iter(clip_b)]))
            eager = list(v._iter_joined_frames([clip_a, clip_b]))
            assert len(streamed) == len(eager)
            for s, e in zip(streamed, eager, strict=True):
                assert np.array_equal(s, e)

    def test_short_clip_still_raises(self) -> None:
        v = _invocation("crossfade", 8)
        with pytest.raises(ValueError, match="transitions need"):
            list(v._iter_joined_frames([iter(_clip(200, 4)), iter(_clip(100, 20))]))

    def test_crossfade_yields_without_materializing_all_blends(self) -> None:
        a = _clip(200, 8)
        b = _clip(100, 8)
        blended = _crossfade(a, b)
        assert isinstance(blended, Iterator)
        assert next(blended).shape == (4, 4, 3)


class TestResourceBounds:
    def test_rejects_transition_window_over_memory_budget(self) -> None:
        v = _invocation("crossfade", 240)
        with pytest.raises(ValueError, match="memory budget"):
            v._validate_transition_memory(width=8192, height=8192)

    def test_accepts_small_transition_window(self) -> None:
        v = _invocation("crossfade", 8)
        assert v._estimate_transition_memory(width=512, height=512) < MAX_TRANSITION_MEMORY_BYTES
        v._validate_transition_memory(width=512, height=512)

    def test_cancellation_stops_join(self) -> None:
        v = _invocation("cut", 0)
        with pytest.raises(CanceledException):
            next(v._iter_joined_frames([iter(_clip(200, 10)), iter(_clip(100, 10))], is_canceled=lambda: True))
