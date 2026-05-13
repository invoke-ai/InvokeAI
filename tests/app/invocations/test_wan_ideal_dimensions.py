"""Unit tests for WanI2VIdealDimensionsInvocation.

The node is a pure math transform — no context dependencies — so we can call
``invoke`` with ``None`` directly.
"""

import pytest

from invokeai.app.invocations.wan_ideal_dimensions import (
    WAN_TARGET_RESOLUTION_PX,
    WanI2VIdealDimensionsInvocation,
)


def _resolve(w: int, h: int, target: str = "720p", rounding: str = "nearest") -> tuple[int, int]:
    inv = WanI2VIdealDimensionsInvocation(
        width=w,
        height=h,
        target_resolution=target,  # type: ignore[arg-type]
        rounding=rounding,  # type: ignore[arg-type]
    )
    out = inv.invoke(None)  # type: ignore[arg-type]
    return out.width, out.height


class TestCommonResolutions:
    """The output table from the docs."""

    @pytest.mark.parametrize(
        "w, h, target, expected",
        [
            (1920, 1080, "720p", (1280, 720)),
            (1080, 1920, "720p", (720, 1280)),
            (832, 480, "720p", (1248, 720)),
            (4032, 3024, "720p", (960, 720)),
            (3840, 2160, "720p", (1280, 720)),
            (1024, 1024, "720p", (720, 720)),
            (1920, 1080, "480p", (848, 480)),
            (1920, 1080, "1080p", (1920, 1088)),  # 1080 → snaps to 1088 (next multiple of 16)
        ],
    )
    def test_nearest(self, w: int, h: int, target: str, expected: tuple[int, int]) -> None:
        assert _resolve(w, h, target=target) == expected


class TestRoundingModes:
    """Floor / ceiling produce the expected over- or under-shoot vs. nearest."""

    def test_floor_never_exceeds_raw(self) -> None:
        # 1920x1080 → 480p has raw_w = 853.33; floor → 848, ceil → 864
        assert _resolve(1920, 1080, target="480p", rounding="floor") == (848, 480)
        assert _resolve(1920, 1080, target="480p", rounding="ceiling") == (864, 480)

    def test_floor_and_ceiling_diverge_for_non_grid_aspect(self) -> None:
        # 21:9-ish: 2048x858, raw_w = 1718.27 → floor 1712, ceil 1728
        assert _resolve(2048, 858, target="720p", rounding="floor") == (1712, 720)
        assert _resolve(2048, 858, target="720p", rounding="ceiling") == (1728, 720)


class TestPostconditions:
    """Output invariants that must always hold."""

    @pytest.mark.parametrize(
        "w, h, target",
        [
            (1920, 1080, "480p"),
            (1920, 1080, "720p"),
            (1080, 1920, "720p"),
            (832, 480, "720p"),
            (2048, 858, "720p"),
            (4032, 3024, "480p"),
            (17, 17, "720p"),  # tiny input
        ],
    )
    @pytest.mark.parametrize("rounding", ["nearest", "floor", "ceiling"])
    def test_output_dims_are_multiples_of_16(
        self, w: int, h: int, target: str, rounding: str
    ) -> None:
        ow, oh = _resolve(w, h, target=target, rounding=rounding)
        assert ow % 16 == 0
        assert oh % 16 == 0

    @pytest.mark.parametrize(
        "w, h, target",
        [
            (1920, 1080, "720p"),
            (1080, 1920, "720p"),
            (832, 480, "720p"),
        ],
    )
    def test_output_aspect_ratio_within_1_percent(self, w: int, h: int, target: str) -> None:
        ow, oh = _resolve(w, h, target=target)
        input_aspect = w / h
        output_aspect = ow / oh
        # 16-grid snap can shift aspect by at most half a 16-step on the long axis,
        # which is ~1.1% at 720 short.
        assert abs(output_aspect - input_aspect) / input_aspect < 0.012

    def test_output_dims_never_zero(self) -> None:
        # Pathologically small input shouldn't return 0×0 even at the smallest preset.
        ow, oh = _resolve(8, 8, target="480p", rounding="floor")
        assert ow >= 16
        assert oh >= 16


class TestResolutionPresetTable:
    """The dropdown values must map to the documented short-side pixel counts."""

    def test_presets_cover_canonical_video_sizes(self) -> None:
        assert WAN_TARGET_RESOLUTION_PX == {"480p": 480, "720p": 720, "1080p": 1080}


class TestInputValidation:
    """Reject obviously bad inputs at the schema layer."""

    def test_zero_width_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WanI2VIdealDimensionsInvocation(width=0, height=720)

    def test_negative_height_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WanI2VIdealDimensionsInvocation(width=720, height=-1)

    def test_unknown_resolution_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WanI2VIdealDimensionsInvocation(
                width=1920, height=1080, target_resolution="2160p"  # type: ignore[arg-type]
            )
