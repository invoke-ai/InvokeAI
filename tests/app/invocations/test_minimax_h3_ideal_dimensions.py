"""Unit tests for MiniMaxH3IdealDimensionsInvocation.

The node is a pure math transform — no context dependencies — so we can call
``invoke`` with ``None`` directly.
"""

import pytest

from invokeai.app.invocations.minimax_h3.minimax_h3_ideal_dimensions import MiniMaxH3IdealDimensionsInvocation
from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_MAX_PIXELS,
)


def _resolve(w: int, h: int, **kwargs) -> tuple[int, int]:
    out = MiniMaxH3IdealDimensionsInvocation(width=w, height=h, **kwargs).invoke(None)  # type: ignore[arg-type]
    return out.width, out.height


class TestCommonResolutions:
    @pytest.mark.parametrize(
        "w, h, expected",
        [
            (1920, 1080, (1344, 768)),  # 16:9 — over the area cap, scales onto it
            (1080, 1920, (768, 1344)),  # 9:16
            (1024, 1024, (768, 768)),  # square: short edge exactly 768
            (1536, 1024, (1152, 768)),  # 3:2 — under the area cap, short edge stays 768
            (1024, 1536, (768, 1152)),  # 2:3
            (4032, 3024, (1024, 768)),  # 4:3 photo
            (768, 1344, (768, 1344)),  # already the native portrait canvas
        ],
    )
    def test_canvas(self, w: int, h: int, expected: tuple[int, int]) -> None:
        assert _resolve(w, h) == expected

    def test_only_aspect_ratio_matters(self) -> None:
        assert _resolve(192, 108) == _resolve(1920, 1080) == _resolve(3840, 2160)


class TestTargetResolutionPresets:
    def test_default_is_native_highres(self) -> None:
        # The pre-1.1.0 behavior: adding the dropdown must not change stored workflows.
        assert _resolve(1024, 1536) == _resolve(1024, 1536, target_resolution="768 highres") == (768, 1152)

    @pytest.mark.parametrize(
        "w, h, expected",
        [
            (1024, 1536, (512, 768)),  # 2:3 — the "small test render" case
            (1920, 1080, (768, 448)),  # 16:9 — true 432 short edge rounds up to the 32-grid
            (1080, 1920, (448, 768)),  # 9:16
            (1024, 1024, (768, 768)),  # square: both presets agree
        ],
    )
    def test_lowres_pins_long_edge(self, w: int, h: int, expected: tuple[int, int]) -> None:
        assert _resolve(w, h, target_resolution="768 lowres") == expected

    @pytest.mark.parametrize("w, h", [(1920, 1080), (500, 500), (1000, 300), (300, 1000), (4032, 3024)])
    def test_lowres_never_larger_than_highres(self, w: int, h: int) -> None:
        lw, lh = _resolve(w, h, target_resolution="768 lowres")
        hw, hh = _resolve(w, h, target_resolution="768 highres")
        assert lw * lh <= hw * hh
        assert lw % MINIMAX_H3_CANVAS_MULTIPLE == 0
        assert lh % MINIMAX_H3_CANVAS_MULTIPLE == 0


class TestPostconditions:
    @pytest.mark.parametrize("w, h", [(1920, 1080), (500, 500), (1000, 300), (300, 1000), (4032, 3024)])
    def test_grid_and_area(self, w: int, h: int) -> None:
        rw, rh = _resolve(w, h)
        assert rw % MINIMAX_H3_CANVAS_MULTIPLE == 0
        assert rh % MINIMAX_H3_CANVAS_MULTIPLE == 0
        # Rounding to the 32-grid may land slightly above the pre-rounding budget, but
        # never by more than one grid step per axis.
        assert rw * rh <= ((MINIMAX_H3_MAX_PIXELS**0.5 + MINIMAX_H3_CANVAS_MULTIPLE) ** 2), (
            f"{rw}x{rh} blows the area budget"
        )


class TestRejection:
    @pytest.mark.parametrize("w, h", [(4100, 1000), (1000, 4100)])
    def test_extreme_aspect_ratio_rejected(self, w: int, h: int) -> None:
        with pytest.raises(ValueError, match="aspect ratio"):
            _resolve(w, h)
