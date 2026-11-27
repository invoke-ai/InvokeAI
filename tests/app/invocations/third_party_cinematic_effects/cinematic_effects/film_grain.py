# Copyright (c) 2024 Jonathan S. Pollack (https://github.com/JPPhoto)

from typing import Optional

import numpy as np
from invokeai.invocation_api import (
    SEED_MAX,
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    get_random_seed,
    invocation,
)
from PIL import Image, ImageChops, ImageFilter


@invocation("film_grain", title="FilmGrain", tags=["film_grain"], version="1.2.0")
class FilmGrainInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adds film grain to an image"""

    image: ImageField = InputField(description="The image to add film grain to")
    amount_1: int = InputField(ge=0, le=800, description="Amount of the first noise layer", default=100)
    amount_2: int = InputField(ge=0, le=800, description="Amount of the second noise layer", default=50)
    seed_1: Optional[int] = InputField(ge=0, le=SEED_MAX, description="The first seed to use (omit for random)")
    seed_2: Optional[int] = InputField(ge=0, le=SEED_MAX, description="The second seed to use (omit for random)")
    blur_1: float = InputField(ge=0, le=100, description="The strength of the first noise blur", default=0.5)
    blur_2: float = InputField(ge=0, le=100, description="The strength of the second noise blur", default=0.5)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mode = image.mode

        if mode == "RGBA":
            image = image.convert("RGB")

        rng = np.random.default_rng(seed=self.seed_1 if self.seed_1 is not None else get_random_seed())
        noise_1 = rng.normal(0, 1, (image.size[1], image.size[0], 3)) * 127.5 * (self.amount_1 / 800.0)
        noise_1 = noise_1 + 127.5
        noise_1 = Image.fromarray(noise_1.astype("uint8"))
        noise_1 = noise_1.filter(ImageFilter.GaussianBlur(radius=self.blur_1))

        rng = np.random.default_rng(seed=self.seed_2 if self.seed_2 is not None else get_random_seed())
        noise_2 = rng.normal(0, 1, (image.size[1], image.size[0], 3)) * 127.5 * (self.amount_2 / 800.0)
        noise_2 = noise_2 + 127.5
        noise_2 = Image.fromarray(noise_2.astype("uint8"))
        noise_2 = noise_2.filter(ImageFilter.GaussianBlur(radius=self.blur_2))

        image = ImageChops.overlay(image, noise_1)
        image = ImageChops.overlay(image, noise_2)

        if mode == "RGBA":
            image = image.convert("RGBA")

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)


@invocation("monochrome_film_grain", title="MonochromeFilmGrain", tags=["film_grain", "monochrome"], version="1.2.0")
class MonochromeFilmGrainInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adds monochrome film grain to an image"""

    image: ImageField = InputField(description="The image to add film grain to")
    amount_1: int = InputField(ge=0, le=800, description="Amount of the first noise layer", default=100)
    amount_2: int = InputField(ge=0, le=800, description="Amount of the second noise layer", default=50)
    seed_1: Optional[int] = InputField(ge=0, le=SEED_MAX, description="The first seed to use (omit for random)")
    seed_2: Optional[int] = InputField(ge=0, le=SEED_MAX, description="The second seed to use (omit for random)")
    blur_1: float = InputField(ge=0, le=100, description="The strength of the first noise blur", default=0.5)
    blur_2: float = InputField(ge=0, le=100, description="The strength of the second noise blur", default=0.5)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mode = image.mode

        if mode == "RGBA":
            image = image.convert("RGB")

        rng = np.random.default_rng(seed=self.seed_1 if self.seed_1 is not None else get_random_seed())
        noise_1 = rng.normal(0, 1, (image.size[1], image.size[0])) * 127.5 * (self.amount_1 / 800.0)
        noise_1 = noise_1 + 127.5
        noise_1 = Image.fromarray(noise_1.astype("uint8"))
        noise_1 = noise_1.filter(ImageFilter.GaussianBlur(radius=self.blur_1))
        noise_1 = noise_1.convert("RGB")

        rng = np.random.default_rng(seed=self.seed_2 if self.seed_2 is not None else get_random_seed())
        noise_2 = rng.normal(0, 1, (image.size[1], image.size[0])) * 127.5 * (self.amount_2 / 800.0)
        noise_2 = noise_2 + 127.5
        noise_2 = Image.fromarray(noise_2.astype("uint8"))
        noise_2 = noise_2.filter(ImageFilter.GaussianBlur(radius=self.blur_2))
        noise_2 = noise_2.convert("RGB")

        image = ImageChops.overlay(image, noise_1)
        image = ImageChops.overlay(image, noise_2)

        if mode == "RGBA":
            image = image.convert("RGBA")

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)


@invocation("chroma_noise", title="ChromaNoise", tags=["film_grain", "chroma_noise"], version="1.2.0")
class ChromaNoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adds chroma-only noise (in OKLab space) to an image."""

    image: ImageField = InputField(description="The image to add chroma noise to")
    amount_1: int = InputField(ge=0, le=800, description="Amount of the first chroma noise layer", default=100)
    amount_2: int = InputField(ge=0, le=800, description="Amount of the second chroma noise layer", default=50)
    seed_1: Optional[int] = InputField(ge=0, le=SEED_MAX, description="The first seed to use (omit for random)")
    seed_2: Optional[int] = InputField(ge=0, le=SEED_MAX, description="The second seed to use (omit for random)")
    blur_1: float = InputField(ge=0, le=100, description="The strength of the first noise blur", default=0.5)
    blur_2: float = InputField(ge=0, le=100, description="The strength of the second noise blur", default=0.5)

    def _srgb_to_linear(self, rgb: np.ndarray) -> np.ndarray:
        a = 0.055
        rgb = np.clip(rgb, 0.0, 1.0)
        return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)

    def _linear_to_srgb(self, rgb_linear: np.ndarray) -> np.ndarray:
        a = 0.055
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        return np.where(
            rgb_linear <= 0.0031308,
            rgb_linear * 12.92,
            (1 + a) * (rgb_linear ** (1 / 2.4)) - a,
        )

    def _srgb_to_oklab(self, rgb: np.ndarray) -> np.ndarray:
        rgb_linear = self._srgb_to_linear(rgb)

        lms = np.tensordot(
            rgb_linear,
            np.array(
                [
                    [0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005],
                ],
                dtype=np.float64,
            ).T,
            axes=([2], [0]),
        )

        lms_cbrt = np.cbrt(lms)

        lab = np.tensordot(
            lms_cbrt,
            np.array(
                [
                    [0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660],
                ],
                dtype=np.float64,
            ).T,
            axes=([2], [0]),
        )

        return lab.astype(np.float32)

    def _oklab_to_srgb(self, lab: np.ndarray) -> np.ndarray:
        lms_cbrt = np.tensordot(
            lab,
            np.array(
                [
                    [1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480],
                ],
                dtype=np.float64,
            ).T,
            axes=([2], [0]),
        )

        lms = lms_cbrt**3

        rgb_linear = np.tensordot(
            lms,
            np.array(
                [
                    [4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010],
                ],
                dtype=np.float64,
            ).T,
            axes=([2], [0]),
        )

        return self._linear_to_srgb(rgb_linear).astype(np.float32)

    def _ab_sigma(self, amount: int) -> float:
        # Map 0..800 -> OKLab a/b stddev
        return 0.0008 * float(amount)

    def _blur_unit_noise(self, unit_noise: np.ndarray, radius: float) -> np.ndarray:
        """
        Blur 2-channel unit noise using PIL in uint8 RGB (avoids Pillow's 'F' mode limitation).
        unit_noise is expected to be roughly N(0,1).
        """
        if radius <= 0:
            return unit_noise

        # Pack (a_noise, b_noise, dummy) into RGB, blur once.
        rgb = np.empty((unit_noise.shape[0], unit_noise.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(unit_noise[..., 0] * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8)
        rgb[..., 1] = np.clip(unit_noise[..., 1] * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8)
        rgb[..., 2] = 127  # constant channel

        img = Image.fromarray(rgb, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius))
        arr = np.asarray(img, dtype=np.float32)

        out = np.empty_like(unit_noise, dtype=np.float32)
        out[..., 0] = (arr[..., 0] - 127.5) / 127.5
        out[..., 1] = (arr[..., 1] - 127.5) / 127.5
        return out

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mode = image.mode

        alpha = None
        if mode == "RGBA":
            alpha = image.getchannel("A")
            image = image.convert("RGB")
        elif mode != "RGB":
            image = image.convert("RGB")

        rgb = np.asarray(image, dtype=np.float32) / 255.0
        lab = self._srgb_to_oklab(rgb)

        h, w = rgb.shape[0], rgb.shape[1]

        rng = np.random.default_rng(seed=self.seed_1 if self.seed_1 is not None else get_random_seed())
        n1_unit = rng.normal(0.0, 1.0, (h, w, 2)).astype(np.float32)
        n1_unit = self._blur_unit_noise(n1_unit, self.blur_1)
        n1 = n1_unit * self._ab_sigma(self.amount_1)

        rng = np.random.default_rng(seed=self.seed_2 if self.seed_2 is not None else get_random_seed())
        n2_unit = rng.normal(0.0, 1.0, (h, w, 2)).astype(np.float32)
        n2_unit = self._blur_unit_noise(n2_unit, self.blur_2)
        n2 = n2_unit * self._ab_sigma(self.amount_2)

        lab_noisy = lab.copy()
        lab_noisy[..., 1] = lab_noisy[..., 1] + n1[..., 0] + n2[..., 0]
        lab_noisy[..., 2] = lab_noisy[..., 2] + n1[..., 1] + n2[..., 1]

        rgb_out = (self._oklab_to_srgb(lab_noisy) * 255.0 + 0.5).astype(np.uint8)
        out_img = Image.fromarray(rgb_out, mode="RGB")

        if alpha is not None:
            out_img = out_img.convert("RGBA")
            out_img.putalpha(alpha)

        image_dto = context.images.save(image=out_img)
        return ImageOutput.build(image_dto)
