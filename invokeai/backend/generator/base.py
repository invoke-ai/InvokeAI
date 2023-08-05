"""
Base class for invokeai.backend.generator.*
including img2img, txt2img, and inpaint
"""
from __future__ import annotations

import itertools
import dataclasses
import diffusers
import os
import random
import traceback
from abc import ABCMeta
from argparse import Namespace
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter
from accelerate.utils import set_seed
from diffusers import DiffusionPipeline
from tqdm import trange
from typing import Callable, List, Iterator, Optional, Type, Union
from dataclasses import dataclass, field
from diffusers.schedulers import SchedulerMixin as Scheduler

import invokeai.backend.util.logging as logger
from ..image_util import configure_model_padding
from ..util.util import rand_perlin_2d
from ..stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline
from ..stable_diffusion.schedulers import SCHEDULER_MAP

downsampling = 8


@dataclass
class InvokeAIGeneratorBasicParams:
    seed: Optional[int] = None
    width: int = 512
    height: int = 512
    cfg_scale: float = 7.5
    steps: int = 20
    ddim_eta: float = 0.0
    scheduler: str = "ddim"
    precision: str = "float16"
    perlin: float = 0.0
    threshold: float = 0.0
    seamless: bool = False
    seamless_axes: List[str] = field(default_factory=lambda: ["x", "y"])
    h_symmetry_time_pct: Optional[float] = None
    v_symmetry_time_pct: Optional[float] = None
    variation_amount: float = 0.0
    with_variations: list = field(default_factory=list)


@dataclass
class InvokeAIGeneratorOutput:
    """
    InvokeAIGeneratorOutput is a dataclass that contains the outputs of a generation
    operation, including the image, its seed, the model name used to generate the image
    and the model hash, as well as all the generate() parameters that went into
    generating the image (in .params, also available as attributes)
    """

    image: Image.Image
    seed: int
    model_hash: str
    attention_maps_images: List[Image.Image]
    params: Namespace


# we are interposing a wrapper around the original Generator classes so that
# old code that calls Generate will continue to work.
class InvokeAIGenerator(metaclass=ABCMeta):
    def __init__(
        self,
        model_info: dict,
        params: InvokeAIGeneratorBasicParams = InvokeAIGeneratorBasicParams(),
        **kwargs,
    ):
        self.model_info = model_info
        self.params = params
        self.kwargs = kwargs

    def generate(
        self,
        conditioning: tuple,
        scheduler,
        callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
        iterations: int = 1,
        **keyword_args,
    ) -> Iterator[InvokeAIGeneratorOutput]:
        """
        Return an iterator across the indicated number of generations.
        Each time the iterator is called it will return an InvokeAIGeneratorOutput
        object. Use like this:

           outputs = txt2img.generate(prompt='banana sushi', iterations=5)
           for result in outputs:
               print(result.image, result.seed)

        In the typical case of wanting to get just a single image, iterations
        defaults to 1 and do:

           output = next(txt2img.generate(prompt='banana sushi')

        Pass None to get an infinite iterator.

           outputs = txt2img.generate(prompt='banana sushi', iterations=None)
           for o in outputs:
               print(o.image, o.seed)

        """
        generator_args = dataclasses.asdict(self.params)
        generator_args.update(keyword_args)

        model_info = self.model_info
        model_name = model_info.name
        model_hash = model_info.hash
        with model_info.context as model:
            gen_class = self._generator_class()
            generator = gen_class(model, self.params.precision, **self.kwargs)
            if self.params.variation_amount > 0:
                generator.set_variation(
                    generator_args.get("seed"),
                    generator_args.get("variation_amount"),
                    generator_args.get("with_variations"),
                )

            if isinstance(model, DiffusionPipeline):
                for component in [model.unet, model.vae]:
                    configure_model_padding(
                        component, generator_args.get("seamless", False), generator_args.get("seamless_axes")
                    )
            else:
                configure_model_padding(
                    model, generator_args.get("seamless", False), generator_args.get("seamless_axes")
                )

            iteration_count = range(iterations) if iterations else itertools.count(start=0, step=1)
            for i in iteration_count:
                results = generator.generate(
                    conditioning=conditioning,
                    step_callback=step_callback,
                    sampler=scheduler,
                    **generator_args,
                )
                output = InvokeAIGeneratorOutput(
                    image=results[0][0],
                    seed=results[0][1],
                    attention_maps_images=results[0][2],
                    model_hash=model_hash,
                    params=Namespace(model_name=model_name, **generator_args),
                )
                if callback:
                    callback(output)
            yield output

    @classmethod
    def schedulers(self) -> List[str]:
        """
        Return list of all the schedulers that we currently handle.
        """
        return list(SCHEDULER_MAP.keys())

    def load_generator(self, model: StableDiffusionGeneratorPipeline, generator_class: Type[Generator]):
        return generator_class(model, self.params.precision)

    @classmethod
    def _generator_class(cls) -> Type[Generator]:
        """
        In derived classes return the name of the generator to apply.
        If you don't override will return the name of the derived
        class, which nicely parallels the generator class names.
        """
        return Generator


# ------------------------------------
class Img2Img(InvokeAIGenerator):
    def generate(
        self, init_image: Union[Image.Image, torch.FloatTensor], strength: float = 0.75, **keyword_args
    ) -> Iterator[InvokeAIGeneratorOutput]:
        return super().generate(init_image=init_image, strength=strength, **keyword_args)

    @classmethod
    def _generator_class(cls):
        from .img2img import Img2Img

        return Img2Img


# ------------------------------------
# Takes all the arguments of Img2Img and adds the mask image and the seam/infill stuff
class Inpaint(Img2Img):
    def generate(
        self,
        mask_image: Union[Image.Image, torch.FloatTensor],
        # Seam settings - when 0, doesn't fill seam
        seam_size: int = 96,
        seam_blur: int = 16,
        seam_strength: float = 0.7,
        seam_steps: int = 30,
        tile_size: int = 32,
        inpaint_replace=False,
        infill_method=None,
        inpaint_width=None,
        inpaint_height=None,
        inpaint_fill: tuple(int) = (0x7F, 0x7F, 0x7F, 0xFF),
        **keyword_args,
    ) -> Iterator[InvokeAIGeneratorOutput]:
        return super().generate(
            mask_image=mask_image,
            seam_size=seam_size,
            seam_blur=seam_blur,
            seam_strength=seam_strength,
            seam_steps=seam_steps,
            tile_size=tile_size,
            inpaint_replace=inpaint_replace,
            infill_method=infill_method,
            inpaint_width=inpaint_width,
            inpaint_height=inpaint_height,
            inpaint_fill=inpaint_fill,
            **keyword_args,
        )

    @classmethod
    def _generator_class(cls):
        from .inpaint import Inpaint

        return Inpaint


class Generator:
    downsampling_factor: int
    latent_channels: int
    precision: str
    model: DiffusionPipeline

    def __init__(self, model: DiffusionPipeline, precision: str, **kwargs):
        self.model = model
        self.precision = precision
        self.seed = None
        self.latent_channels = model.unet.config.in_channels
        self.downsampling_factor = downsampling  # BUG: should come from model or config
        self.perlin = 0.0
        self.threshold = 0
        self.variation_amount = 0
        self.with_variations = []
        self.use_mps_noise = False
        self.free_gpu_mem = None

    # this is going to be overridden in img2img.py, txt2img.py and inpaint.py
    def get_make_image(self, **kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        """
        raise NotImplementedError("image_iterator() must be implemented in a descendent class")

    def set_variation(self, seed, variation_amount, with_variations):
        self.seed = seed
        self.variation_amount = variation_amount
        self.with_variations = with_variations

    def generate(
        self,
        width,
        height,
        sampler,
        init_image=None,
        iterations=1,
        seed=None,
        image_callback=None,
        step_callback=None,
        threshold=0.0,
        perlin=0.0,
        h_symmetry_time_pct=None,
        v_symmetry_time_pct=None,
        free_gpu_mem: bool = False,
        **kwargs,
    ):
        scope = nullcontext
        self.free_gpu_mem = free_gpu_mem
        attention_maps_images = []
        attention_maps_callback = lambda saver: attention_maps_images.append(saver.get_stacked_maps_image())
        make_image = self.get_make_image(
            sampler=sampler,
            init_image=init_image,
            width=width,
            height=height,
            step_callback=step_callback,
            threshold=threshold,
            perlin=perlin,
            h_symmetry_time_pct=h_symmetry_time_pct,
            v_symmetry_time_pct=v_symmetry_time_pct,
            attention_maps_callback=attention_maps_callback,
            **kwargs,
        )
        results = []
        seed = seed if seed is not None and seed >= 0 else self.new_seed()
        first_seed = seed
        seed, initial_noise = self.generate_initial_noise(seed, width, height)

        # There used to be an additional self.model.ema_scope() here, but it breaks
        # the inpaint-1.5 model. Not sure what it did.... ?
        with scope(self.model.device.type):
            for n in trange(iterations, desc="Generating"):
                x_T = None
                if self.variation_amount > 0:
                    set_seed(seed)
                    target_noise = self.get_noise(width, height)
                    x_T = self.slerp(self.variation_amount, initial_noise, target_noise)
                elif initial_noise is not None:
                    # i.e. we specified particular variations
                    x_T = initial_noise
                else:
                    set_seed(seed)
                    try:
                        x_T = self.get_noise(width, height)
                    except:
                        logger.error("An error occurred while getting initial noise")
                        print(traceback.format_exc())

                # Pass on the seed in case a layer beneath us needs to generate noise on its own.
                image = make_image(x_T, seed)

                results.append([image, seed, attention_maps_images])

                if image_callback is not None:
                    attention_maps_image = None if len(attention_maps_images) == 0 else attention_maps_images[-1]
                    image_callback(
                        image,
                        seed,
                        first_seed=first_seed,
                        attention_maps_image=attention_maps_image,
                    )

                seed = self.new_seed()

                # Free up memory from the last generation.
                clear_cuda_cache = kwargs["clear_cuda_cache"] if "clear_cuda_cache" in kwargs else None
                if clear_cuda_cache is not None:
                    clear_cuda_cache()

        return results

    def sample_to_image(self, samples) -> Image.Image:
        """
        Given samples returned from a sampler, converts
        it into a PIL Image
        """
        with torch.inference_mode():
            image = self.model.decode_latents(samples)
        return self.model.numpy_to_pil(image)[0]

    def repaste_and_color_correct(
        self,
        result: Image.Image,
        init_image: Image.Image,
        init_mask: Image.Image,
        mask_blur_radius: int = 8,
    ) -> Image.Image:
        if init_image is None or init_mask is None:
            return result

        # Get the original alpha channel of the mask if there is one.
        # Otherwise it is some other black/white image format ('1', 'L' or 'RGB')
        pil_init_mask = init_mask.getchannel("A") if init_mask.mode == "RGBA" else init_mask.convert("L")
        pil_init_image = init_image.convert("RGBA")  # Add an alpha channel if one doesn't exist

        # Build an image with only visible pixels from source to use as reference for color-matching.
        init_rgb_pixels = np.asarray(init_image.convert("RGB"), dtype=np.uint8)
        init_a_pixels = np.asarray(pil_init_image.getchannel("A"), dtype=np.uint8)
        init_mask_pixels = np.asarray(pil_init_mask, dtype=np.uint8)

        # Get numpy version of result
        np_image = np.asarray(result, dtype=np.uint8)

        # Mask and calculate mean and standard deviation
        mask_pixels = init_a_pixels * init_mask_pixels > 0
        np_init_rgb_pixels_masked = init_rgb_pixels[mask_pixels, :]
        np_image_masked = np_image[mask_pixels, :]

        if np_init_rgb_pixels_masked.size > 0:
            init_means = np_init_rgb_pixels_masked.mean(axis=0)
            init_std = np_init_rgb_pixels_masked.std(axis=0)
            gen_means = np_image_masked.mean(axis=0)
            gen_std = np_image_masked.std(axis=0)

            # Color correct
            np_matched_result = np_image.copy()
            np_matched_result[:, :, :] = (
                (
                    (
                        (np_matched_result[:, :, :].astype(np.float32) - gen_means[None, None, :])
                        / gen_std[None, None, :]
                    )
                    * init_std[None, None, :]
                    + init_means[None, None, :]
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            matched_result = Image.fromarray(np_matched_result, mode="RGB")
        else:
            matched_result = Image.fromarray(np_image, mode="RGB")

        # Blur the mask out (into init image) by specified amount
        if mask_blur_radius > 0:
            nm = np.asarray(pil_init_mask, dtype=np.uint8)
            nmd = cv2.erode(
                nm,
                kernel=np.ones((3, 3), dtype=np.uint8),
                iterations=int(mask_blur_radius / 2),
            )
            pmd = Image.fromarray(nmd, mode="L")
            blurred_init_mask = pmd.filter(ImageFilter.BoxBlur(mask_blur_radius))
        else:
            blurred_init_mask = pil_init_mask

        multiplied_blurred_init_mask = ImageChops.multiply(blurred_init_mask, self.pil_image.split()[-1])

        # Paste original on color-corrected generation (using blurred mask)
        matched_result.paste(init_image, (0, 0), mask=multiplied_blurred_init_mask)
        return matched_result

    @staticmethod
    def sample_to_lowres_estimated_image(samples):
        # origingally adapted from code by @erucipe and @keturn here:
        # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7

        # these updated numbers for v1.5 are from @torridgristle
        v1_5_latent_rgb_factors = torch.tensor(
            [
                #    R        G        B
                [0.3444, 0.1385, 0.0670],  # L1
                [0.1247, 0.4027, 0.1494],  # L2
                [-0.3192, 0.2513, 0.2103],  # L3
                [-0.1307, -0.1874, -0.7445],  # L4
            ],
            dtype=samples.dtype,
            device=samples.device,
        )

        latent_image = samples[0].permute(1, 2, 0) @ v1_5_latent_rgb_factors
        latents_ubyte = (
            ((latent_image + 1) / 2).clamp(0, 1).mul(0xFF).byte()  # change scale from -1..1 to 0..1  # to 0..255
        ).cpu()

        return Image.fromarray(latents_ubyte.numpy())

    def generate_initial_noise(self, seed, width, height):
        initial_noise = None
        if self.variation_amount > 0 or len(self.with_variations) > 0:
            # use fixed initial noise plus random noise per iteration
            set_seed(seed)
            initial_noise = self.get_noise(width, height)
            for v_seed, v_weight in self.with_variations:
                seed = v_seed
                set_seed(seed)
                next_noise = self.get_noise(width, height)
                initial_noise = self.slerp(v_weight, initial_noise, next_noise)
            if self.variation_amount > 0:
                random.seed()  # reset RNG to an actually random state, so we can get a random seed for variations
                seed = random.randrange(0, np.iinfo(np.uint32).max)
        return (seed, initial_noise)

    def get_perlin_noise(self, width, height):
        fixdevice = "cpu" if (self.model.device.type == "mps") else self.model.device
        # limit noise to only the diffusion image channels, not the mask channels
        input_channels = min(self.latent_channels, 4)
        # round up to the nearest block of 8
        temp_width = int((width + 7) / 8) * 8
        temp_height = int((height + 7) / 8) * 8
        noise = torch.stack(
            [
                rand_perlin_2d((temp_height, temp_width), (8, 8), device=self.model.device).to(fixdevice)
                for _ in range(input_channels)
            ],
            dim=0,
        ).to(self.model.device)
        return noise[0:4, 0:height, 0:width]

    def new_seed(self):
        self.seed = random.randrange(0, np.iinfo(np.uint32).max)
        return self.seed

    def slerp(self, t, v0, v1, DOT_THRESHOLD=0.9995):
        """
        Spherical linear interpolation
        Args:
            t (float/np.ndarray): Float value between 0.0 and 1.0
            v0 (np.ndarray): Starting vector
            v1 (np.ndarray): Final vector
            DOT_THRESHOLD (float): Threshold for considering the two vectors as
                                colineal. Not recommended to alter this.
        Returns:
            v2 (np.ndarray): Interpolation vector between v0 and v1
        """
        inputs_are_torch = False
        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            v0 = v0.detach().cpu().numpy()
        if not isinstance(v1, np.ndarray):
            inputs_are_torch = True
            v1 = v1.detach().cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(self.model.device)

        return v2

    # this is a handy routine for debugging use. Given a generated sample,
    # convert it into a PNG image and store it at the indicated path
    def save_sample(self, sample, filepath):
        image = self.sample_to_image(sample)
        dirname = os.path.dirname(filepath) or "."
        if not os.path.exists(dirname):
            logger.info(f"creating directory {dirname}")
            os.makedirs(dirname, exist_ok=True)
        image.save(filepath, "PNG")

    def torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.precision == "float16" else torch.float32

    # returns a tensor filled with random numbers from a normal distribution
    def get_noise(self, width, height):
        device = self.model.device
        # limit noise to only the diffusion image channels, not the mask channels
        input_channels = min(self.latent_channels, 4)
        x = torch.randn(
            [
                1,
                input_channels,
                height // self.downsampling_factor,
                width // self.downsampling_factor,
            ],
            dtype=self.torch_dtype(),
            device=device,
        )
        if self.perlin > 0.0:
            perlin_noise = self.get_perlin_noise(width // self.downsampling_factor, height // self.downsampling_factor)
            x = (1 - self.perlin) * x + self.perlin * perlin_noise
        return x
