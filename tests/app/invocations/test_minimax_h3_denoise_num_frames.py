"""The MiniMax H3 denoise node's frame count is a choice list, not a free integer.

Only `17n + 5` counts exist on the video VAE's chunk grid, so the node offers them as an enum
(strings, the only enum shape both workflow editors render as a dropdown) and converts to an int
at invoke time.
"""

import pytest
from pydantic import ValidationError

from invokeai.app.invocations.minimax_h3_denoise import (
    MINIMAX_H3_NUM_FRAMES_LABELS,
    MiniMaxH3DenoiseInvocation,
)
from invokeai.backend.minimax_h3.packing import MINIMAX_H3_VIDEO_FRAME_CHOICES
from invokeai.backend.minimax_h3.sampling import MINIMAX_H3_STILL_NUM_FRAMES, validate_num_frames


def _node(**kwargs) -> MiniMaxH3DenoiseInvocation:
    return MiniMaxH3DenoiseInvocation(id="denoise", **kwargs)


def test_default_is_the_five_second_grid_point():
    assert _node().num_frames == "124"


@pytest.mark.parametrize("frames", MINIMAX_H3_VIDEO_FRAME_CHOICES)
def test_every_video_choice_is_accepted_and_valid(frames: int):
    node = _node(num_frames=str(frames))
    # The stored value round-trips to the int the denoise loop uses, and clears its own validator.
    assert int(node.num_frames) == frames
    validate_num_frames(int(node.num_frames))


def test_still_image_block_is_offered():
    # The linear UI's image output mode sets exactly this value; dropping it would break that path.
    assert _node(num_frames=str(MINIMAX_H3_STILL_NUM_FRAMES)).num_frames == "5"


@pytest.mark.parametrize("value", ["6", "121", "362", "89", "1024", "", "124 frames"])
def test_off_grid_and_out_of_range_values_are_rejected(value: str):
    with pytest.raises(ValidationError):
        _node(num_frames=value)


def test_integer_values_are_rejected():
    # An old graph carrying `num_frames: 124` must fail loudly at graph validation rather than
    # silently coercing (pydantic does not coerce int -> str Literal in lax mode).
    with pytest.raises(ValidationError):
        _node(num_frames=124)


def test_labels_cover_every_choice_and_name_the_duration():
    assert set(MINIMAX_H3_NUM_FRAMES_LABELS) == {
        str(MINIMAX_H3_STILL_NUM_FRAMES),
        *(str(n) for n in MINIMAX_H3_VIDEO_FRAME_CHOICES),
    }
    assert MINIMAX_H3_NUM_FRAMES_LABELS["124"] == "124 frames - 5.17 s"
    assert MINIMAX_H3_NUM_FRAMES_LABELS["90"] == "90 frames - 3.75 s"
    assert MINIMAX_H3_NUM_FRAMES_LABELS["5"] == "5 frames - still image only"
    # Every label opens with "<count> frames" so the list reads consistently in the dropdown.
    assert all(label.startswith(f"{key} frames") for key, label in MINIMAX_H3_NUM_FRAMES_LABELS.items())


def test_choices_appear_in_the_generated_json_schema():
    schema = MiniMaxH3DenoiseInvocation.model_json_schema()
    num_frames = schema["properties"]["num_frames"]
    # ui_choice_labels rides along so the editors can show "124 frames - 5.17 s".
    assert num_frames["ui_choice_labels"] == MINIMAX_H3_NUM_FRAMES_LABELS
    assert num_frames.get("default") == "124"
