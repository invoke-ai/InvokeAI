"""Numerical contracts shared with the canvas spatial-adjustment kernels.

webv2 previews Clarity, Texture and Contrast Dehaze on the canvas with its own TypeScript ports
(`workbench/canvas-engine/render/spatial/`). Its tests read the fixtures in that directory's
`__fixtures__/`; these tests pin the same files to the Python implementations, so a backend change that
moves the numerics fails here until the parity fixture is regenerated and the ports follow.
"""

import base64
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from invokeai.backend.image_util.clarity import clarity
from invokeai.backend.image_util.color import prepare_lightness
from invokeai.backend.image_util.contrast_dehaze import contrast_dehaze
from invokeai.backend.image_util.guided_filter import guided_filter
from invokeai.backend.image_util.multiscale import edge_aware_band
from invokeai.backend.image_util.texture import texture

FIXTURE_DIR = (
    Path(__file__).resolve().parents[3]
    / "invokeai/frontend/webv2/src/workbench/canvas-engine/render/spatial/__fixtures__"
)
FIXTURE_PATH = FIXTURE_DIR / "shared_image_manipulation_numerics.json"
PARITY_FIXTURE_PATH = FIXTURE_DIR / "spatial_adjustment_parity.json"

REGEN_HINT = (
    "Regenerate it with:\n"
    f"    REGEN_SPATIAL_PARITY_FIXTURE=1 pytest {Path(__file__).name}\n"
    "then make webv2's spatial kernels pass `evaluateSpatialSteps.test.ts` against it."
)

# The parity fixture holds only objects, strings and integers, so `json.dumps(indent=2)` is already
# oxfmt's output and webv2's `format:check` accepts the regenerated file. Keep number arrays out of it.

_STEPS: dict[str, Callable[..., Image.Image]] = {
    "clarity": lambda image, step: clarity(image, step["amount"], step["size"]),
    "texture": lambda image, step: texture(image, step["amount"], step["size"]),
    "dehaze": lambda image, step: contrast_dehaze(image, step["amount"]),
}


def _encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _decode(text: str, dtype: type[np.generic], shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=dtype).reshape(shape)


def _visible_raster_labels(hidden: np.ndarray) -> np.ndarray:
    """cv2's nearest-visible labels, as each pixel's visible pixel's index among visible pixels in raster order."""
    _, labels = cv2.distanceTransformWithLabels(
        np.ascontiguousarray(hidden, dtype=np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    # DIST_LABEL_PIXEL numbers the zero (visible) pixels from 1 in raster order.
    return (labels - 1).astype("<i4")


def _apply_steps(rgba: np.ndarray, steps: list[dict[str, Any]]) -> np.ndarray:
    """Runs the nodes on the alpha bounds, as the canvas does, leaving everything outside them untouched."""
    visible_rows, visible_columns = np.nonzero(rgba[..., 3])
    if visible_rows.size == 0:
        return rgba.copy()
    bounds = (
        slice(visible_rows.min(), visible_rows.max() + 1),
        slice(visible_columns.min(), visible_columns.max() + 1),
    )
    image = Image.fromarray(np.ascontiguousarray(rgba[bounds]), mode="RGBA")
    for step in steps:
        image = _STEPS[step["kind"]](image, step)
    result = rgba.copy()
    result[bounds] = np.asarray(image, dtype=np.uint8)
    return result


def _label_masks(rng: np.random.Generator) -> dict[str, np.ndarray]:
    corners = np.ones((7, 9), dtype=bool)
    corners[[0, 0, -1, -1], [0, -1, 0, -1]] = False
    centre = np.ones((5, 5), dtype=bool)
    centre[2, 2] = False
    ring = np.ones((11, 11), dtype=bool)
    ring[2, 2:9] = ring[8, 2:9] = ring[2:9, 2] = ring[2:9, 8] = False
    row = np.ones((1, 12), dtype=bool)
    row[0, [3, 8]] = False
    column = np.ones((12, 1), dtype=bool)
    column[[2, 9], 0] = False
    masks = {"equidistant_corners": corners, "single_centre": centre, "hollow_ring": ring, "row": row, "column": column}
    for name, shape, hidden_share in (
        ("random_sparse", (13, 17), 0.97),
        ("random_half", (19, 23), 0.5),
        ("random_dense", (8, 31), 0.9),
    ):
        mask = rng.random(shape) < hidden_share
        mask[rng.integers(shape[0]), rng.integers(shape[1])] = False
        masks[name] = mask
    return masks


def _parity_image(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Detailed RGBA content whose alpha bounds are the whole image, with holes, partial alpha and saturated color."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    rgb = np.stack(
        (
            255 * x / max(width - 1, 1),
            128 + 100 * np.sin(x / 2.5) * np.cos(y / 3.5),
            255 * y / max(height - 1, 1),
        ),
        axis=-1,
    )
    rgb += rng.normal(0, 18, rgb.shape)
    # Saturated colors leave the sRGB gamut once adjusted. Exact primaries sit on the boundaries between the gamut
    # clip's cusp approximations, and colors with a 0 or 255 channel sit on the gamut boundary itself, where
    # float32 round-trip noise must not decide whether the clip runs; both sides must agree on them.
    if width >= 12 and height >= 9:
        rgb[height // 3 : height // 3 + 3, width // 2 : width // 2 + 6] = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [250, 20, 10], [30, 20, 250], [245, 240, 20]]
        )[None]
        rgb[height // 3 + 3 : height // 3 + 6, 2:6] = (0, 17, 166)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    alpha = np.full((height, width), 255, dtype=np.uint8)
    partial = rng.random((height, width)) < 0.12
    alpha[partial] = rng.integers(1, 255, int(partial.sum()), dtype=np.uint8)
    alpha[rng.random((height, width)) < 0.06] = 0
    alpha[2:6, 3:8] = 0
    alpha[[0, 0, -1, -1], [0, -1, 0, -1]] = 255
    rgba[..., 3] = alpha
    # Hidden pixels keep arbitrary color bytes, which the adjustments must neither read nor rewrite.
    hidden = alpha == 0
    rgba[hidden, :3] = rng.integers(0, 256, (int(hidden.sum()), 3), dtype=np.uint8)
    return rgba


def _flat_image(width: int, height: int, rgb: tuple[int, int, int]) -> np.ndarray:
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., :3] = rgb
    return rgba


def _padded(rng: np.random.Generator, content: np.ndarray, width: int, height: int, left: int, top: int) -> np.ndarray:
    """Content inside a larger transparent buffer whose hidden colors are arbitrary."""
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    rgba[top : top + content.shape[0], left : left + content.shape[1]] = content
    return rgba


_PARITY_CASES: tuple[tuple[str, Callable[[np.random.Generator], np.ndarray], list[dict[str, Any]]], ...] = (
    ("clarity_default_size", lambda rng: _parity_image(rng, 24, 18), [{"kind": "clarity", "amount": 80, "size": 40}]),
    (
        "clarity_size_10_half_even_radius",
        lambda rng: _parity_image(rng, 24, 18),
        [{"kind": "clarity", "amount": 120, "size": 10}],
    ),
    ("clarity_negative", lambda rng: _parity_image(rng, 24, 18), [{"kind": "clarity", "amount": -150, "size": 25}]),
    ("texture_positive_gated", lambda rng: _parity_image(rng, 24, 18), [{"kind": "texture", "amount": 150, "size": 8}]),
    ("texture_negative", lambda rng: _parity_image(rng, 24, 18), [{"kind": "texture", "amount": -120, "size": 6}]),
    ("dehaze_positive", lambda rng: _parity_image(rng, 24, 18), [{"kind": "dehaze", "amount": 90}]),
    ("dehaze_negative", lambda rng: _parity_image(rng, 24, 18), [{"kind": "dehaze", "amount": -140}]),
    (
        "dehaze_short_side_20_half_even_radius",
        lambda rng: _parity_image(rng, 28, 20),
        [{"kind": "dehaze", "amount": 70}],
    ),
    (
        "dehaze_short_side_36_half_even_radius",
        lambda rng: _parity_image(rng, 40, 36),
        [{"kind": "dehaze", "amount": 200}],
    ),
    (
        "clarity_then_dehaze",
        lambda rng: _parity_image(rng, 24, 18),
        [{"kind": "clarity", "amount": 60, "size": 12}, {"kind": "dehaze", "amount": 50}],
    ),
    (
        "flat_zero_channel_clarity",
        lambda _: _flat_image(6, 5, (0, 17, 166)),
        [{"kind": "clarity", "amount": 100, "size": 40}],
    ),
    ("flat_pure_blue_dehaze", lambda _: _flat_image(6, 5, (0, 0, 255)), [{"kind": "dehaze", "amount": 50}]),
    ("flat_pure_red_dehaze", lambda _: _flat_image(6, 5, (255, 0, 0)), [{"kind": "dehaze", "amount": 120}]),
    (
        "cropped_alpha_bounds",
        lambda rng: _padded(rng, _parity_image(rng, 20, 14), 31, 22, 6, 3),
        [{"kind": "clarity", "amount": 90, "size": 15}, {"kind": "dehaze", "amount": 60}],
    ),
    (
        "single_row_alpha_bounds",
        lambda rng: _padded(rng, _parity_image(rng, 15, 1), 19, 7, 2, 4),
        [{"kind": "clarity", "amount": 150, "size": 8}, {"kind": "texture", "amount": 120, "size": 4}],
    ),
    (
        "single_column_alpha_bounds",
        lambda rng: _padded(rng, _parity_image(rng, 1, 13), 6, 16, 3, 1),
        [{"kind": "dehaze", "amount": 80}, {"kind": "texture", "amount": -90, "size": 3}],
    ),
)


def _render_parity_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(9891)
    labels = []
    for name, hidden in _label_masks(rng).items():
        height, width = hidden.shape
        labels.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "hidden": _encode(hidden.astype(np.uint8)),
                "labels": _encode(_visible_raster_labels(hidden)),
            }
        )
    adjustments = []
    for name, build, steps in _PARITY_CASES:
        rgba = build(rng)
        height, width = rgba.shape[:2]
        adjustments.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "steps": steps,
                "rgba": _encode(rgba),
                "expected": _encode(_apply_steps(rgba, steps)),
            }
        )
    return {
        "description": (
            "Written by tests/backend/image_util/test_shared_numerical_fixtures.py. Labels: base64 uint8 hidden "
            "masks and base64 little-endian int32 visible-pixel raster indices from cv2.distanceTransformWithLabels. "
            "Adjustments: base64 RGBA8 inputs and the Python nodes' outputs, steps applied in order to the alpha "
            "bounds."
        ),
        "opencv": cv2.__version__,
        "labels": labels,
        "adjustments": adjustments,
    }


def test_shared_numerical_fixtures_match_python_cores() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    guided = fixtures["guided_filter"]
    guided_result = guided_filter(
        np.asarray(guided["guide"], dtype=np.float32),
        np.asarray(guided["signal"], dtype=np.float32),
        radius=guided["radius"],
        epsilon=guided["epsilon"],
    )
    np.testing.assert_allclose(guided_result, guided["expected"], atol=guided["atol"], rtol=0)

    multiscale = fixtures["multiscale_band"]
    band_result = edge_aware_band(
        np.asarray(multiscale["lightness"], dtype=np.float32),
        inner_radius=multiscale["inner_radius"],
        outer_radius=multiscale["outer_radius"],
        epsilon=multiscale["epsilon"],
    )
    np.testing.assert_allclose(band_result, multiscale["expected"], atol=multiscale["atol"], rtol=0)

    adapter = fixtures["color_adapter"]
    source = Image.fromarray(np.asarray(adapter["rgb"], dtype=np.uint8), mode="RGB")
    color_tensor, lightness, alpha = prepare_lightness(source)
    assert alpha is None
    np.testing.assert_allclose(
        color_tensor.numpy().transpose(1, 2, 0), adapter["expected_oklab"], atol=adapter["atol"], rtol=0
    )
    np.testing.assert_allclose(lightness, np.asarray(adapter["expected_oklab"])[:, :, 0], atol=adapter["atol"], rtol=0)
    assert color_tensor.dtype == torch.float32


def test_spatial_parity_fixture_matches_the_python_adjustments() -> None:
    if os.environ.get("REGEN_SPATIAL_PARITY_FIXTURE"):
        PARITY_FIXTURE_PATH.write_text(
            json.dumps(_render_parity_fixture(), indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    assert PARITY_FIXTURE_PATH.exists(), f"{PARITY_FIXTURE_PATH} is missing. {REGEN_HINT}"
    fixture = json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in fixture["labels"]:
        shape = (case["height"], case["width"])
        hidden = _decode(case["hidden"], np.uint8, shape).astype(bool)
        expected = _decode(case["labels"], np.dtype("<i4").type, shape)
        assert np.array_equal(_visible_raster_labels(hidden), expected), (
            f"{case['name']}: cv2 labels moved. {REGEN_HINT}"
        )

    for case in fixture["adjustments"]:
        shape = (case["height"], case["width"], 4)
        rgba = _decode(case["rgba"], np.uint8, shape)
        expected = _decode(case["expected"], np.uint8, shape)
        assert np.array_equal(_apply_steps(rgba.copy(), case["steps"]), expected), (
            f"{case['name']}: the Python adjustment output moved. {REGEN_HINT}"
        )
