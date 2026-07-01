"""Unit tests for WanTI2VIdealDimensionsInvocation.

Mirrors ``test_wan_ideal_dimensions.py`` but for the TI2V-5B variant, which
snaps to a multiple of 32 (16x Wan 2.2-VAE × 2x transformer patch) instead of
16. The node is a pure math transform — no context dependencies — so we can
call ``invoke`` with ``None`` directly.
"""

import pytest

from invokeai.app.invocations.wan_ideal_dimensions import (
    WAN_TARGET_RESOLUTION_PX,
    WAN_TI2V_PIXEL_MULTIPLE,
    WanTI2VIdealDimensionsInvocation,
)


def _resolve(w: int, h: int, target: str = "720p", rounding: str = "nearest") -> tuple[int, int]:
    inv = WanTI2VIdealDimensionsInvocation(
        width=w,
        height=h,
        target_resolution=target,  # type: ignore[arg-type]
        rounding=rounding,  # type: ignore[arg-type]
    )
    out = inv.invoke(None)  # type: ignore[arg-type]
    return out.width, out.height


class TestCommonResolutions:
    """Verified output table for the 32-px grid."""

    @pytest.mark.parametrize(
        "w, h, target, expected",
        [
            (1920, 1080, "720p", (1280, 704)),
            (1080, 1920, "720p", (704, 1280)),
            (832, 480, "720p", (1248, 704)),
            (4032, 3024, "720p", (960, 704)),
            (3840, 2160, "720p", (1280, 704)),
            (1024, 1024, "720p", (704, 704)),
            (1920, 1080, "480p", (864, 480)),
            (1920, 1080, "1080p", (1920, 1088)),  # 1080 → 1088 (next multiple of 32)
            # The reported failing case: 720x480 is not a multiple of 32 (720 % 32 != 0);
            # the node snaps it to valid dims at both presets.
            (720, 480, "480p", (704, 480)),
            (720, 480, "720p", (1088, 704)),
        ],
    )
    def test_nearest(self, w: int, h: int, target: str, expected: tuple[int, int]) -> None:
        assert _resolve(w, h, target=target) == expected


class TestRoundingModes:
    """Floor / ceiling produce the expected over- or under-shoot vs. nearest."""

    def test_floor_never_exceeds_raw(self) -> None:
        # 1920x1080 → 480p has raw_w = 853.33; floor → 832, ceil → 864
        assert _resolve(1920, 1080, target="480p", rounding="floor") == (832, 480)
        assert _resolve(1920, 1080, target="480p", rounding="ceiling") == (864, 480)

    def test_floor_and_ceiling_diverge_for_non_grid_aspect(self) -> None:
        # 2048x858, raw_w = 1718.6 → floor 1696, ceil 1728
        assert _resolve(2048, 858, target="720p", rounding="floor") == (1696, 704)
        assert _resolve(2048, 858, target="720p", rounding="ceiling") == (1728, 736)


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
            (720, 480, "720p"),
            (33, 33, "720p"),  # tiny input
        ],
    )
    @pytest.mark.parametrize("rounding", ["nearest", "floor", "ceiling"])
    def test_output_dims_are_multiples_of_32(self, w: int, h: int, target: str, rounding: str) -> None:
        ow, oh = _resolve(w, h, target=target, rounding=rounding)
        assert ow % 32 == 0
        assert oh % 32 == 0

    @pytest.mark.parametrize(
        "w, h, target",
        [
            (1920, 1080, "720p"),
            (1080, 1920, "720p"),
            (832, 480, "720p"),
        ],
    )
    def test_output_aspect_ratio_within_2_percent(self, w: int, h: int, target: str) -> None:
        ow, oh = _resolve(w, h, target=target)
        input_aspect = w / h
        output_aspect = ow / oh
        # 32-grid snap can shift aspect by at most half a 32-step on the long axis,
        # which is ~2.2% at 704 short — looser than the I2V node's 16-grid tolerance.
        assert abs(output_aspect - input_aspect) / input_aspect < 0.023

    def test_smallest_valid_input_still_snaps_to_32_grid(self) -> None:
        # 32×32 is the minimum input the guard accepts. The downstream clamp ensures
        # the output is at least 32×32 even when floor rounding would zero it.
        ow, oh = _resolve(32, 32, target="480p", rounding="floor")
        assert ow >= 32
        assert oh >= 32


class TestResolutionPresetTable:
    """The dropdown values must map to the documented short-side pixel counts."""

    def test_presets_cover_canonical_video_sizes(self) -> None:
        assert WAN_TARGET_RESOLUTION_PX == {"480p": 480, "720p": 720, "1080p": 1080}

    def test_pixel_multiple_is_32(self) -> None:
        assert WAN_TI2V_PIXEL_MULTIPLE == 32


class TestInputValidation:
    """Reject obviously bad inputs at the schema layer."""

    def test_zero_width_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WanTI2VIdealDimensionsInvocation(width=0, height=720)

    def test_negative_height_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WanTI2VIdealDimensionsInvocation(width=720, height=-1)

    def test_input_smaller_than_pixel_grid_rejected(self) -> None:
        # If the longer side is below the 32-px TI2V grid, the floor-rounding output
        # would silently disconnect from the requested aspect ratio. Fail fast instead.
        with pytest.raises(ValueError, match="smaller than the Wan pixel grid"):
            _resolve(16, 16, target="480p", rounding="floor")
        with pytest.raises(ValueError, match="smaller than the Wan pixel grid"):
            _resolve(31, 31, target="720p", rounding="nearest")

    def test_unknown_resolution_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WanTI2VIdealDimensionsInvocation(
                width=1920,
                height=1080,
                target_resolution="2160p",  # type: ignore[arg-type]
            )
