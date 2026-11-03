# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

import re
from typing import Literal

import numpy as np
from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)
from PIL import Image
from scipy.interpolate import CubicSpline, PchipInterpolator

from .util import oklch_to_rgb, rgb_to_oklch  # noqa: TID252

_INTERPOLATION_METHODS = {"lin": "linear", "cat": "catmull_rom", "cub": "cubic", "pch": "pchip"}


@invocation(
    "curves_invocation", title="CurvesInvocation", tags=["curves", "interpolation", "dithering"], version="1.3.0"
)
class CurvesInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Applies interpolation and dithering to an image based on user inputs"""

    image: ImageField = InputField(description="The image to apply curves and dithering to")
    mapping_str: str = InputField(
        description="String of input-output mappings", default="0:0,64:48,128:128,200:240,255:255"
    )
    interpolation_method: Literal["catmull_rom", "linear", "cubic", "pchip"] = InputField(
        default="catmull_rom", description="Interpolation method"
    )
    dithering_method: Literal["none", "blue_noise", "floyd_steinberg", "ordered"] = InputField(
        default="blue_noise", description="Dithering method"
    )
    channel: Literal["RGB", "R", "G", "B", "L (OKLCH)"] = InputField(default="RGB", description="Channel to alter")

    def parse_mapping(self, mapping_str: str) -> list[tuple[int, int]]:
        """Parses a string of mappings into a list of (input, output) pairs."""
        pairs = mapping_str.replace(" ", "").split(",")
        pts: list[tuple[int, int]] = []
        for pair in pairs:
            i, o = pair.split(":")
            pts.append((int(i), int(o)))
        return pts

    def denormalize_image(self, img_array: np.ndarray) -> np.ndarray:
        """Denormalize the image back to [0, 255] range."""
        return np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

    def catmull_rom_spline(self, x: float, points: list[tuple[float, float]]) -> float:
        """Performs Catmull-Rom spline interpolation given a set of points."""
        pts = sorted(points, key=lambda p: p[0])
        n = len(pts)
        if x <= pts[0][0]:
            result = pts[0][1]
        elif x >= pts[-1][0]:
            result = pts[-1][1]
        else:
            j = next(i for i in range(n - 1) if pts[i][0] <= x <= pts[i + 1][0])
            p1 = pts[j]
            p2 = pts[j + 1]
            p0 = pts[j - 1] if j - 1 >= 0 else p1
            p3 = pts[j + 2] if j + 2 < n else p2
            t = (x - p1[0]) / (p2[0] - p1[0])
            a0 = -0.5 * p0[1] + 1.5 * p1[1] - 1.5 * p2[1] + 0.5 * p3[1]
            a1 = p0[1] - 2.5 * p1[1] + 2 * p2[1] - 0.5 * p3[1]
            a2 = -0.5 * p0[1] + 0.5 * p2[1]
            a3 = p1[1]
            result = a0 * t**3 + a1 * t**2 + a2 * t + a3
        return result

    def linear_interpolation(self, x: float, points: list[tuple[float, float]]) -> float:
        """Performs linear interpolation between two points."""
        pts = sorted(points, key=lambda p: p[0])
        for i in range(1, len(pts)):
            if x < pts[i][0]:
                x0, y0 = pts[i - 1]
                x1, y1 = pts[i]
                return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        return pts[-1][1]

    def apply_interpolation(self, channel: np.ndarray, points: list[tuple[int, int]], method: str) -> np.ndarray:
        """Apply interpolation to a single-channel (2D) image array."""
        # normalize control points once
        pts = [(xi / 255.0, yi / 255.0) for xi, yi in points]
        if method in ("cubic", "pchip"):
            pts = sorted(pts, key=lambda p: p[0])
            xs, ys = zip(*pts)
            interpolator = CubicSpline if method == "cubic" else PchipInterpolator
            return interpolator(xs, ys)(channel).astype(np.float32)

        h, w = channel.shape
        out = np.empty((h, w), dtype=np.float32)
        for y in range(h):
            row = channel[y]
            for x in range(w):
                v = float(row[x])  # already in [0,1]
                if method == "catmull_rom":
                    out[y, x] = self.catmull_rom_spline(v, pts)
                elif method == "linear":
                    out[y, x] = self.linear_interpolation(v, pts)
                else:
                    raise ValueError(f"Unsupported interpolation: {method}")

        return out

    def no_dither(self, channel: np.ndarray) -> np.ndarray:
        """A no-op conversion"""
        return np.clip(channel, 0.0, 1.0)

    def floyd_steinberg_dither(self, channel: np.ndarray) -> np.ndarray:
        """Optimized Floyd–Steinberg dithering with 8-bit quantization using NumPy."""
        d = channel.astype(np.float32).copy()
        h, w = d.shape

        # Precomputed weights
        w1, w2, w3, w4 = 7 / 16, 3 / 16, 5 / 16, 1 / 16

        for y in range(h - 1):
            for x in range(1, w - 1):
                old = d[y, x]
                new = round(old * 255) / 255
                if old != new:
                    err = old - new
                    d[y, x] = new

                    # Propagate error
                    d[y, x + 1] += err * w1
                    d[y + 1, x - 1] += err * w2
                    d[y + 1, x] += err * w3
                    d[y + 1, x + 1] += err * w4
        return np.clip(d, 0.0, 1.0)

    def generate_bayer_matrix(self, n: int) -> np.ndarray:
        """Generates an N×N Bayer matrix normalized to [0, 1), where N is a power of 2."""
        assert (n & (n - 1)) == 0 and n > 0, "Size must be a power of 2"

        def build(k: int) -> np.ndarray:
            if k == 1:
                return np.array([[0]])
            prev = build(k // 2)
            tiled = np.block([[4 * prev + 0, 4 * prev + 2], [4 * prev + 3, 4 * prev + 1]])
            return tiled

        mat = build(n)
        return mat / (n * n)

    def ordered_dither(self, channel: np.ndarray) -> np.ndarray:
        """Vectorized ordered dithering with 8-bit quantization using a Bayer matrix."""
        threshold = self.generate_bayer_matrix(256)
        h, w = channel.shape
        th_h, th_w = threshold.shape
        tile = np.tile(threshold, (h // th_h + 1, w // th_w + 1))[:h, :w]
        quantized = np.clip(np.round(channel * 255 + (tile - 0.5)) / 255, 0.0, 1.0)
        return quantized

    def blue_noise_dither(self, channel: np.ndarray) -> np.ndarray:
        """Applies blue-noise-like dithering using a locally shuffled Bayer matrix approximation."""
        h, w = channel.shape

        # Step 1: Generate a large Bayer matrix
        base = self.generate_bayer_matrix(256)  # In [0,1)

        # Step 2: Localized random shuffle
        rng = np.random.default_rng(seed=0)  # Remove seed for non-deterministic noise
        flat = base.flatten()
        shuffled = rng.permutation(flat)  # Local, reproducible shuffle
        blue_noise = shuffled.reshape(base.shape)

        # Step 3: Tile to match input size
        tile = np.tile(blue_noise, (h // 64 + 1, w // 64 + 1))[:h, :w]

        # Step 4: Apply dithering
        quantized = np.clip(np.round(channel * 255 + (tile - 0.5)) / 255, 0.0, 1.0)
        return quantized

    def apply_dithering(self, channel: np.ndarray) -> np.ndarray:
        """Apply dithering to a single-channel array using the selected method."""
        if self.dithering_method == "none":
            return self.no_dither(channel)
        elif self.dithering_method == "blue_noise":
            return self.blue_noise_dither(channel)
        elif self.dithering_method == "floyd_steinberg":
            return self.floyd_steinberg_dither(channel)
        elif self.dithering_method == "ordered":
            return self.ordered_dither(channel)
        else:
            raise ValueError(f"Unsupported dithering: {self.dithering_method}")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mode = image.mode

        # extract alpha
        if mode == "RGBA":
            alpha = np.array(image.split()[3])
            image = image.convert("RGB")
        elif mode == "LA":
            alpha = np.array(image.split()[1])
            image = image.convert("L")
        else:
            alpha = None

        ops = self._parse_adjustments(self.mapping_str)
        if not ops:
            ops = [(None, None, self.parse_mapping(self.mapping_str))]  # Legacy fallback

        arr_rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0

        for ch_spec, method_abbrev, pts in ops:
            channel = ch_spec or self.channel
            if channel == "L":
                channel = "L (OKLCH)"

            if method_abbrev is None:
                method = self.interpolation_method
            else:
                method = _INTERPOLATION_METHODS.get(method_abbrev)

            if channel == "L (OKLCH)":
                oklch = rgb_to_oklch(arr_rgb)
                L = oklch[..., 0]
                interp = self.apply_interpolation(L, pts, method)
                oklch[..., 0] = interp
                arr_rgb = oklch_to_rgb(oklch)
            elif channel in ["R", "G", "B"]:
                ch_index = {"R": 0, "G": 1, "B": 2}[channel]
                interp = self.apply_interpolation(arr_rgb[..., ch_index], pts, method)
                arr_rgb[..., ch_index] = interp
            else:  # RGB, all channels
                interp = np.stack(
                    [self.apply_interpolation(arr_rgb[..., c], pts, method) for c in range(arr_rgb.shape[2])],
                    axis=-1,
                )
                arr_rgb = interp

        dithered = np.stack(
            [self.apply_dithering(arr_rgb[..., c]) for c in range(arr_rgb.shape[2])],
            axis=-1,
        )
        result = self.denormalize_image(dithered)

        # reassemble
        if alpha is not None:
            final = Image.fromarray(np.dstack([result, alpha]))
        else:
            final = Image.fromarray(result)

        dto = context.images.save(image=final)
        return ImageOutput.build(dto)

    def _parse_adjustments(self, s: str) -> list[tuple[list[str] | None, str | None, list[tuple[int, int]]]]:
        """
        Parses operations:
          - New form:  R[cat](0:0,128:150,255:255)  G(0:0,128:130,255:250)  RGB[lin](...)
          - Legacy:    0:0,64:48,128:128,200:240,255:255
        Returns list of (channels, method_abbrev, points)
          channels -> ['R'] or ['R','G','B'] or None (use defaults)
        """
        s = s.strip()
        ops: list[tuple[list[str] | None, str | None, list[tuple[int, int]]]] = []
        pat = re.compile(r"(?:(RGB|R|G|B|L)(?:\[(lin|cat|cub|pch)\])?)?\(([^)]*)\)")
        matches = list(pat.finditer(s))
        if matches:
            for m in matches:
                channel = m.group(1)
                method = m.group(2)
                pts_str = m.group(3)
                pts = self.parse_mapping(pts_str)
                ops.append((channel, method, pts))
            return ops
        # legacy bare mapping (backward-compat)
        if ":" in s:
            ops.append((None, None, self.parse_mapping(s)))
        return ops
