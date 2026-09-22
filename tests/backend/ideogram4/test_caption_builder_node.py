"""Validation tests for the Ideogram4CaptionBuilderInvocation region bbox contract.

The caption builder forwards each region's bbox verbatim into the structured JSON, so a malformed box
(wrong length or out-of-range coordinate) would emit a prompt the model may misapply. The Ideogram4Region
model must reject such boxes up front.
"""

import pytest
from pydantic import ValidationError

from invokeai.app.invocations.ideogram4_caption import Ideogram4Region


def test_valid_bbox_is_accepted():
    region = Ideogram4Region(prompt="a cat", bbox=[0, 0, 1000, 1000])
    assert region.bbox == [0, 0, 1000, 1000]


def test_none_bbox_is_accepted():
    region = Ideogram4Region(prompt="a cat", bbox=None)
    assert region.bbox is None


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 1000],  # too short
        [0, 0, 1000, 1000, 7],  # too long
        [-1, 0, 1000, 1000],  # negative
        [0, 0, 1000, 1001],  # above 1000
        [900, 900, 100, 100],  # inverted (y_min > y_max and x_min > x_max)
        [0, 900, 1000, 100],  # inverted x only (x_min > x_max)
    ],
)
def test_malformed_bbox_is_rejected(bbox: list[int]):
    with pytest.raises(ValidationError):
        Ideogram4Region(prompt="a cat", bbox=bbox)
