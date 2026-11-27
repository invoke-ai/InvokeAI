"""Tests for the LTX-2 duration node.

The clamp-and-snap arithmetic belongs to `LTX2DurationHead.predict_num_frames`, so these drive a
real head with only its regression stubbed. What is under test is this node's side of the contract:
that it hands the head the frame rate, the grid and the bounds the user asked for, and that it
refuses the inputs for which no single frame count is a correct answer.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.pipelines.ltx2.duration_head import LTX2DurationHead
from pydantic import ValidationError

from invokeai.app.invocations.fields import LTX2ConditioningField
from invokeai.app.invocations.ltx2.ltx2_duration import LTX2DurationInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import LTX2ConditioningInfo

# Widths of the two connector streams the head reads; a mismatch here would be a silent shape bug.
_VIDEO_DIM = 4096
_AUDIO_DIM = 2048


def _conditioning(count: int = 1) -> SimpleNamespace:
    infos = [
        LTX2ConditioningInfo(
            video_embeds=torch.zeros(1, 8, _VIDEO_DIM),
            audio_embeds=torch.zeros(1, 8, _AUDIO_DIM),
            attention_mask=torch.ones(1, 8, dtype=torch.int64),
        )
        for _ in range(count)
    ]
    return SimpleNamespace(conditionings=infos)


def _context(head: LTX2DurationHead, conditioning_count: int = 1) -> MagicMock:
    context = MagicMock()
    context.logger = MagicMock()
    context.conditioning.load.return_value = _conditioning(conditioning_count)

    @contextmanager
    def on_device(**_kwargs):
        yield (None, head)

    context.models.load.return_value = SimpleNamespace(model_on_device=on_device)
    return context


def _head(monkeypatch: pytest.MonkeyPatch, seconds: float) -> LTX2DurationHead:
    """A real head whose regression is pinned, so the clamp and grid snap under test are the real ones."""
    head = LTX2DurationHead()
    monkeypatch.setattr(
        type(head),
        "forward",
        lambda _self, video_tokens=None, audio_tokens=None: torch.tensor([seconds]),
    )
    return head


def _node(**kwargs) -> LTX2DurationInvocation:
    defaults = {
        "id": "duration",
        "conditioning": LTX2ConditioningField(conditioning_name="positive"),
        "duration_head": ModelIdentifierField(
            key="duration_head",
            hash="hash",
            name="duration_head",
            base=BaseModelType.LTX2,
            type=ModelType.LTX2DurationHead,
            format=ModelFormat.Checkpoint,
        ),
    }
    return LTX2DurationInvocation(**{**defaults, **kwargs})


@pytest.mark.parametrize(("fps", "expected"), [(24.0, 113), (12.0, 57)])
def test_the_frame_count_is_the_predicted_seconds_at_the_clip_s_own_frame_rate(
    monkeypatch: pytest.MonkeyPatch, fps: float, expected: int
) -> None:
    """5s is 120 frames at 24 fps and 60 at 12, and each snaps down to the grid point below (113 / 57).

    Reading the rate from the node rather than assuming 24 is what makes the two differ; passing a
    fixed rate would give the same count for both and no other assertion here would notice.
    """
    head = _head(monkeypatch, seconds=5.0)
    output = _node(fps=fps).invoke(_context(head))

    assert output.num_frames == expected
    assert (output.num_frames - 1) % 8 == 0


def test_a_prediction_below_the_floor_is_raised_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flooring onto the grid can undershoot the minimum; the next grid point up is the answer.

    The floor is deliberately not upstream's own default of 1 s, so the test fails if the node
    stopped passing `min_seconds` at all.
    """
    head = _head(monkeypatch, seconds=0.3)
    output = _node(fps=24.0, min_seconds=3.0).invoke(_context(head))

    assert output.num_frames == 73  # 3s * 24fps = 72, which is not 8k+1; the grid point above is 73
    assert output.num_frames >= 3.0 * 24.0


def test_a_prediction_beyond_the_bounds_is_clamped_before_it_becomes_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A head predicting 30s under a 4s ceiling must yield a 4s clip, not a 30s one."""
    head = _head(monkeypatch, seconds=30.0)
    output = _node(fps=24.0, max_seconds=4.0).invoke(_context(head))

    assert output.num_frames == 89  # 4s * 24fps = 96 -> grid point at or below
    assert output.num_frames <= 4.0 * 24.0
    # The raw regression is reported unclamped, so a user can see the model wanted far longer.
    assert output.seconds == pytest.approx(30.0)


@pytest.mark.parametrize(("min_seconds", "max_seconds"), [(10.0, 2.0), (4.0, 4.0)])
def test_it_refuses_bounds_that_leave_no_range(
    monkeypatch: pytest.MonkeyPatch, min_seconds: float, max_seconds: float
) -> None:
    """min above max is a contradiction; equal bounds make the head's grid fallback land outside them (upstream refuses both)."""
    head = _head(monkeypatch, seconds=5.0)
    with pytest.raises(ValueError, match="must be less than max_seconds"):
        _node(min_seconds=min_seconds, max_seconds=max_seconds).invoke(_context(head))


def test_it_refuses_a_batch_of_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two prompts have two natural lengths; one frame count cannot serve both."""
    head = _head(monkeypatch, seconds=5.0)
    with pytest.raises(ValueError, match="exactly one conditioning"):
        _node().invoke(_context(head, conditioning_count=2))


def test_an_extension_s_context_is_added_in_front_of_the_predicted_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prompt sizes the new material: 5s at 24 fps is 113 frames, i.e. 112 after the first, behind 17 held ones."""
    head = _head(monkeypatch, seconds=5.0)
    output = _node(fps=24.0, context_frames=17).invoke(_context(head))

    assert output.num_frames == 17 + 112
    assert (output.num_frames - 1) % 8 == 0


def test_an_extension_always_keeps_one_group_of_new_material(monkeypatch: pytest.MonkeyPatch) -> None:
    """At 1 fps a 1s floor is one frame; added naively that is the context alone, a continuation of nothing."""
    head = _head(monkeypatch, seconds=0.3)
    output = _node(fps=1.0, min_seconds=1.0, max_seconds=5.0, context_frames=17).invoke(_context(head))

    assert output.num_frames == 17 + 8


@pytest.mark.parametrize("context_frames", [0, 17])
def test_the_frame_ceiling_holds_when_the_rate_is_higher_than_the_bounds_assumed(
    monkeypatch: pytest.MonkeyPatch, context_frames: int
) -> None:
    """Bounds sized at a guessed 24 fps, read at a real 48: the seconds clamp alone would double the run."""
    head = _head(monkeypatch, seconds=30.0)
    output = _node(fps=48.0, max_seconds=5.0, context_frames=context_frames, max_num_frames=124).invoke(_context(head))

    assert output.num_frames == 121  # 124 snapped down onto the grid


def test_the_frame_ceiling_is_reached_when_the_rate_is_lower_than_the_caller_assumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source recorded at 24 fps that really plays at 16: seconds sized at 24 would stop at 81 of 121 frames."""
    head = _head(monkeypatch, seconds=30.0)
    output = _node(fps=16.0, context_frames=17, max_num_frames=121).invoke(_context(head))

    assert output.num_frames == 121


def test_a_standalone_extension_stays_within_the_family_s_longest_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """241 held frames plus a 20 s prediction at 24 fps would be 713 frames, past what LTX-2 generates."""
    head = _head(monkeypatch, seconds=30.0)
    output = _node(fps=24.0, context_frames=241).invoke(_context(head))

    assert output.num_frames == 481
    # Nor can a caller ask for more: no LTX-2 run goes past it.
    with pytest.raises(ValidationError):
        _node(max_num_frames=482)


@pytest.mark.parametrize(
    ("context_frames", "max_num_frames", "match"),
    [(16, 481, "8k\\+1 grid"), (17, 20, "leaves no room")],
)
def test_it_refuses_a_context_it_cannot_continue_from(
    monkeypatch: pytest.MonkeyPatch, context_frames: int, max_num_frames: int, match: str
) -> None:
    head = _head(monkeypatch, seconds=5.0)
    with pytest.raises(ValueError, match=match):
        _node(context_frames=context_frames, max_num_frames=max_num_frames).invoke(_context(head))
