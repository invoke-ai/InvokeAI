"""Regression tests for VideoConcatInvocation._assemble.

Covers JPPhoto's code-review finding (PR #9163): ``fade_through_black`` claimed to
emit ``transition_frames`` frames per boundary but used a symmetric ``tf // 2``
split, dropping one frame for odd ``tf``. The fix splits asymmetrically:
``tail_half = tf // 2`` consumed from the previous clip's tail, ``head_half =
tf - tail_half`` from the next clip's head, so the emitted total is exactly
``tf`` for both parities.

We exercise ``_assemble`` directly because it is the pure transformation
implementing the contract (the surrounding ``invoke()`` deals with imageio
encode/decode plumbing that isn't germane to the math).
"""

import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_concat import VideoConcatInvocation


def _invocation(transition: str, transition_frames: int) -> VideoConcatInvocation:
    # ``videos`` requires min_length=2 to construct; values are unused by ``_assemble``.
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
        out = v._assemble([clip_a, clip_b])
        # fade_through_black: each boundary consumes tf frames (tail_half from A + head_half from B)
        # and emits exactly tf frames in their place, so the total length is preserved.
        assert len(out) == len(clip_a) + len(clip_b)

    def test_three_clip_chain(self) -> None:
        clip_a = _clip(200, 10)
        clip_b = _clip(150, 10)
        clip_c = _clip(100, 10)
        v = _invocation("fade_through_black", 3)
        out = v._assemble([clip_a, clip_b, clip_c])
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
        out = v._assemble([clip_a, clip_b, clip_c])
        assert len(out) == 15


class TestCrossfadeTransitionLength:
    """Crossfade behaviour is unchanged; pin its length for safety."""

    def test_crossfade_shortens_by_tf_per_boundary(self) -> None:
        clip_a = _clip(200, 10)
        clip_b = _clip(100, 10)
        v = _invocation("crossfade", 5)
        out = v._assemble([clip_a, clip_b])
        # 20 input - 5 frames consumed from each side (10 total) + 5 blended emitted = 15.
        assert len(out) == 15


class TestCutNoTransitionFrames:
    """transition='cut' or tf=0 returns the raw concatenation."""

    def test_cut_concatenates_directly(self) -> None:
        clip_a = _clip(200, 4)
        clip_b = _clip(100, 6)
        v = _invocation("cut", 0)
        out = v._assemble([clip_a, clip_b])
        assert len(out) == 10

    def test_fade_through_black_tf_zero(self) -> None:
        clip_a = _clip(200, 4)
        clip_b = _clip(100, 6)
        v = _invocation("fade_through_black", 0)
        out = v._assemble([clip_a, clip_b])
        assert len(out) == 10
