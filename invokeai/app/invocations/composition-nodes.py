# All nodes in this file are originally pulled from https://github.com/dwringer/composition-nodes

import os
from ast import literal_eval as tuple_from_string
from functools import reduce
from io import BytesIO
from math import pi as PI
from typing import Literal, Optional

import cv2
import numpy
import torch
from PIL import Image, ImageChops, ImageCms, ImageColor, ImageDraw, ImageEnhance, ImageOps
from torchvision.transforms.functional import to_pil_image as pil_image_from_tensor

from invokeai.app.invocations.primitives import ImageOutput
from invokeai.backend.image_util.composition import (
    CIELAB_TO_UPLAB_ICC_PATH,
    MAX_FLOAT,
    equivalent_achromatic_lightness,
    gamut_clip_tensor,
    hsl_from_srgb,
    linear_srgb_from_oklab,
    linear_srgb_from_srgb,
    okhsl_from_srgb,
    okhsv_from_srgb,
    oklab_from_linear_srgb,
    remove_nans,
    srgb_from_hsl,
    srgb_from_linear_srgb,
    srgb_from_okhsl,
    srgb_from_okhsv,
    tensor_from_pil_image,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)

HUE_COLOR_SPACES = Literal[
    "HSV / HSL / RGB",
    "Okhsl",
    "Okhsv",
    "*Oklch / Oklab",
    "*LCh / CIELab",
    "*UPLab (w/CIELab_to_UPLab.icc)",
]


@invocation(
    "invokeai_img_hue_adjust_plus",
    title="Adjust Image Hue Plus",
    tags=["image", "hue", "oklab", "cielab", "uplab", "lch", "hsv", "hsl", "lab"],
    category="image",
    version="1.2.0",
)
class InvokeAdjustImageHuePlusInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adjusts the Hue of an image by rotating it in the selected color space. Originally created by @dwringer"""

    image: ImageField = InputField(description="The image to adjust")
    space: HUE_COLOR_SPACES = InputField(
        default="HSV / HSL / RGB",
        description="Color space in which to rotate hue by polar coords (*: non-invertible)",
    )
    degrees: float = InputField(default=0.0, description="Degrees by which to rotate image hue")
    preserve_lightness: bool = InputField(default=False, description="Whether to preserve CIELAB lightness values")
    ok_adaptive_gamut: float = InputField(
        ge=0, default=0.05, description="Higher preserves chroma at the expense of lightness (Oklab)"
    )
    ok_high_precision: bool = InputField(
        default=True, description="Use more steps in computing gamut (Oklab/Okhsv/Okhsl)"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.images.get_pil(self.image.image_name)
        image_out = None
        space = self.space.split()[0].lower().strip("*")

        # Keep the mode and alpha channel for restoration after shifting the hue:
        image_mode = image_in.mode
        original_mode = image_mode
        alpha_channel = None
        if (image_mode == "RGBA") or (image_mode == "LA") or (image_mode == "PA"):
            alpha_channel = image_in.getchannel("A")
        elif (image_mode == "RGBa") or (image_mode == "La") or (image_mode == "Pa"):
            alpha_channel = image_in.getchannel("a")
        if (image_mode == "RGBA") or (image_mode == "RGBa"):
            image_mode = "RGB"
        elif (image_mode == "LA") or (image_mode == "La"):
            image_mode = "L"
        elif image_mode == "PA":
            image_mode = "P"

        image_in = image_in.convert("RGB")

        # Keep the CIELAB L* lightness channel for restoration if Preserve Lightness is selected:
        (channel_l, channel_a, channel_b, profile_srgb, profile_lab, profile_uplab, lab_transform, uplab_transform) = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if self.preserve_lightness or (space == "lch") or (space == "uplab"):
            profile_srgb = ImageCms.createProfile("sRGB")
            if space == "uplab":
                with open(CIELAB_TO_UPLAB_ICC_PATH, "rb") as f:
                    profile_uplab = ImageCms.getOpenProfile(f)
            if profile_uplab is None:
                profile_lab = ImageCms.createProfile("LAB", colorTemp=6500)
            else:
                profile_lab = ImageCms.createProfile("LAB", colorTemp=5000)

            lab_transform = ImageCms.buildTransformFromOpenProfiles(
                profile_srgb, profile_lab, "RGB", "LAB", renderingIntent=2, flags=0x2400
            )
            image_out = ImageCms.applyTransform(image_in, lab_transform)
            if profile_uplab is not None:
                uplab_transform = ImageCms.buildTransformFromOpenProfiles(
                    profile_lab, profile_uplab, "LAB", "LAB", renderingIntent=2, flags=0x2400
                )
                image_out = ImageCms.applyTransform(image_out, uplab_transform)

            channel_l = image_out.getchannel("L")
            channel_a = image_out.getchannel("A")
            channel_b = image_out.getchannel("B")

        if space == "hsv":
            hsv_tensor = image_resized_to_grid_as_tensor(image_in.convert("HSV"), normalize=False, multiple_of=1)
            hsv_tensor[0, :, :] = torch.remainder(torch.add(hsv_tensor[0, :, :], torch.div(self.degrees, 360.0)), 1.0)
            image_out = pil_image_from_tensor(hsv_tensor, mode="HSV").convert("RGB")

        elif space == "okhsl":
            rgb_tensor = image_resized_to_grid_as_tensor(image_in.convert("RGB"), normalize=False, multiple_of=1)
            hsl_tensor = okhsl_from_srgb(rgb_tensor, steps=(3 if self.ok_high_precision else 1))
            hsl_tensor[0, :, :] = torch.remainder(torch.add(hsl_tensor[0, :, :], torch.div(self.degrees, 360.0)), 1.0)
            rgb_tensor = srgb_from_okhsl(hsl_tensor, alpha=0.0)
            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        elif space == "okhsv":
            rgb_tensor = image_resized_to_grid_as_tensor(image_in.convert("RGB"), normalize=False, multiple_of=1)
            hsv_tensor = okhsv_from_srgb(rgb_tensor, steps=(3 if self.ok_high_precision else 1))
            hsv_tensor[0, :, :] = torch.remainder(torch.add(hsv_tensor[0, :, :], torch.div(self.degrees, 360.0)), 1.0)
            rgb_tensor = srgb_from_okhsv(hsv_tensor, alpha=0.0)
            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        elif (space == "lch") or (space == "uplab"):
            # <Channels a and b were already extracted, above.>

            a_tensor = image_resized_to_grid_as_tensor(channel_a, normalize=True, multiple_of=1)
            b_tensor = image_resized_to_grid_as_tensor(channel_b, normalize=True, multiple_of=1)

            # L*a*b* to L*C*h
            c_tensor = torch.sqrt(torch.add(torch.pow(a_tensor, 2.0), torch.pow(b_tensor, 2.0)))
            h_tensor = torch.atan2(b_tensor, a_tensor)

            # Rotate h
            rot_rads = (self.degrees / 180.0) * PI

            h_rot = torch.add(h_tensor, rot_rads)
            h_rot = torch.sub(torch.remainder(torch.add(h_rot, PI), 2 * PI), PI)

            # L*C*h to L*a*b*
            a_tensor = torch.mul(c_tensor, torch.cos(h_rot))
            b_tensor = torch.mul(c_tensor, torch.sin(h_rot))

            # -1..1 -> 0..1 for all elts of a, b
            a_tensor = torch.div(torch.add(a_tensor, 1.0), 2.0)
            b_tensor = torch.div(torch.add(b_tensor, 1.0), 2.0)

            a_img = pil_image_from_tensor(a_tensor)
            b_img = pil_image_from_tensor(b_tensor)

            image_out = Image.merge("LAB", (channel_l, a_img, b_img))

            if profile_uplab is not None:
                deuplab_transform = ImageCms.buildTransformFromOpenProfiles(
                    profile_uplab, profile_lab, "LAB", "LAB", renderingIntent=2, flags=0x2400
                )
                image_out = ImageCms.applyTransform(image_out, deuplab_transform)

            rgb_transform = ImageCms.buildTransformFromOpenProfiles(
                profile_lab, profile_srgb, "LAB", "RGB", renderingIntent=2, flags=0x2400
            )
            image_out = ImageCms.applyTransform(image_out, rgb_transform)

        elif space == "oklch":
            rgb_tensor = image_resized_to_grid_as_tensor(image_in.convert("RGB"), normalize=False, multiple_of=1)

            linear_srgb_tensor = linear_srgb_from_srgb(rgb_tensor)

            lab_tensor = oklab_from_linear_srgb(linear_srgb_tensor)

            # L*a*b* to L*C*h
            c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1, :, :], 2.0), torch.pow(lab_tensor[2, :, :], 2.0)))
            h_tensor = torch.atan2(lab_tensor[2, :, :], lab_tensor[1, :, :])

            # Rotate h
            rot_rads = (self.degrees / 180.0) * PI

            h_rot = torch.add(h_tensor, rot_rads)
            h_rot = torch.remainder(torch.add(h_rot, 2 * PI), 2 * PI)

            # L*C*h to L*a*b*
            lab_tensor[1, :, :] = torch.mul(c_tensor, torch.cos(h_rot))
            lab_tensor[2, :, :] = torch.mul(c_tensor, torch.sin(h_rot))

            linear_srgb_tensor = linear_srgb_from_oklab(lab_tensor)

            rgb_tensor = srgb_from_linear_srgb(
                linear_srgb_tensor, alpha=self.ok_adaptive_gamut, steps=(3 if self.ok_high_precision else 1)
            )

            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        # Not all modes can convert directly to LAB using pillow:
        # image_out = image_out.convert("RGB")

        # Restore the L* channel if required:
        if self.preserve_lightness and (not ((space == "lch") or (space == "uplab"))):
            if profile_uplab is None:
                profile_lab = ImageCms.createProfile("LAB", colorTemp=6500)
            else:
                profile_lab = ImageCms.createProfile("LAB", colorTemp=5000)

            lab_transform = ImageCms.buildTransformFromOpenProfiles(
                profile_srgb, profile_lab, "RGB", "LAB", renderingIntent=2, flags=0x2400
            )

            image_out = ImageCms.applyTransform(image_out, lab_transform)

            if profile_uplab is not None:
                uplab_transform = ImageCms.buildTransformFromOpenProfiles(
                    profile_lab, profile_uplab, "LAB", "LAB", renderingIntent=2, flags=0x2400
                )
                image_out = ImageCms.applyTransform(image_out, uplab_transform)

            image_out = Image.merge("LAB", tuple([channel_l] + [image_out.getchannel(c) for c in "AB"]))

            if profile_uplab is not None:
                deuplab_transform = ImageCms.buildTransformFromOpenProfiles(
                    profile_uplab, profile_lab, "LAB", "LAB", renderingIntent=2, flags=0x2400
                )
                image_out = ImageCms.applyTransform(image_out, deuplab_transform)

            rgb_transform = ImageCms.buildTransformFromOpenProfiles(
                profile_lab, profile_srgb, "LAB", "RGB", renderingIntent=2, flags=0x2400
            )
            image_out = ImageCms.applyTransform(image_out, rgb_transform)

        # Restore the original image mode, with alpha channel if required:
        image_out = image_out.convert(image_mode)
        if "a" in original_mode.lower():
            image_out = Image.merge(
                original_mode, tuple([image_out.getchannel(c) for c in image_mode] + [alpha_channel])
            )

        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)


@invocation(
    "invokeai_img_enhance",
    title="Enhance Image",
    tags=["enhance", "image"],
    category="image",
    version="1.2.0",
)
class InvokeImageEnhanceInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Applies processing from PIL's ImageEnhance module. Originally created by @dwringer"""

    image: ImageField = InputField(default=None, description="The image for which to apply processing")
    invert: bool = InputField(default=False, description="Whether to invert the image colors")
    color: float = InputField(ge=0, default=1.0, description="Color enhancement factor")
    contrast: float = InputField(ge=0, default=1.0, description="Contrast enhancement factor")
    brightness: float = InputField(ge=0, default=1.0, description="Brightness enhancement factor")
    sharpness: float = InputField(ge=0, default=1.0, description="Sharpness enhancement factor")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_out = context.images.get_pil(self.image.image_name)
        if self.invert:
            if image_out.mode not in ("L", "RGB"):
                image_out = image_out.convert("RGB")
            image_out = ImageOps.invert(image_out)
        if self.color != 1.0:
            color_enhancer = ImageEnhance.Color(image_out)
            image_out = color_enhancer.enhance(self.color)
        if self.contrast != 1.0:
            contrast_enhancer = ImageEnhance.Contrast(image_out)
            image_out = contrast_enhancer.enhance(self.contrast)
        if self.brightness != 1.0:
            brightness_enhancer = ImageEnhance.Brightness(image_out)
            image_out = brightness_enhancer.enhance(self.brightness)
        if self.sharpness != 1.0:
            sharpness_enhancer = ImageEnhance.Sharpness(image_out)
            image_out = sharpness_enhancer.enhance(self.sharpness)
        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)


@invocation(
    "invokeai_ealightness",
    title="Equivalent Achromatic Lightness",
    tags=["image", "channel", "mask", "cielab", "lab"],
    category="image",
    version="1.2.0",
)
class InvokeEquivalentAchromaticLightnessInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Calculate Equivalent Achromatic Lightness from image. Originally created by @dwringer"""

    image: ImageField = InputField(description="Image from which to get channel")

    #  The chroma, C*
    # , and the hue, h, in the CIELAB color space are obtained by C*=sqrt((a*)^2+(b*)^2)
    #  and h=arctan(b*/a*)
    # k 0.1644	0.0603	0.1307	0.0060

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.images.get_pil(self.image.image_name)

        if image_in.mode == "L":
            image_in = image_in.convert("RGB")

        image_out = image_in.convert("LAB")
        channel_l = image_out.getchannel("L")
        channel_a = image_out.getchannel("A")
        channel_b = image_out.getchannel("B")

        l_tensor = image_resized_to_grid_as_tensor(channel_l, normalize=False, multiple_of=1)
        l_max = torch.ones(l_tensor.shape)
        l_min = torch.zeros(l_tensor.shape)
        a_tensor = image_resized_to_grid_as_tensor(channel_a, normalize=True, multiple_of=1)
        b_tensor = image_resized_to_grid_as_tensor(channel_b, normalize=True, multiple_of=1)

        c_tensor = torch.sqrt(torch.add(torch.pow(a_tensor, 2.0), torch.pow(b_tensor, 2.0)))
        h_tensor = torch.atan2(b_tensor, a_tensor)

        k = [0.1644, 0.0603, 0.1307, 0.0060]

        h_minus_90 = torch.sub(h_tensor, PI / 2.0)
        h_minus_90 = torch.sub(torch.remainder(torch.add(h_minus_90, 3 * PI), 2 * PI), PI)

        f_by = torch.add(k[0] * torch.abs(torch.sin(torch.div(h_minus_90, 2.0))), k[1])
        f_r_0 = torch.add(k[2] * torch.abs(torch.cos(h_tensor)), k[3])

        f_r = torch.zeros(l_tensor.shape)
        mask_hi = torch.ge(h_tensor, -1 * (PI / 2.0))
        mask_lo = torch.le(h_tensor, PI / 2.0)
        mask = torch.logical_and(mask_hi, mask_lo)
        f_r[mask] = f_r_0[mask]

        l_adjustment = torch.tensordot(torch.add(f_by, f_r), c_tensor, dims=([1, 2], [1, 2]))
        l_max = torch.add(l_max, l_adjustment)
        l_min = torch.add(l_min, l_adjustment)
        image_tensor = torch.add(l_tensor, l_adjustment)

        image_tensor = torch.div(torch.sub(image_tensor, l_min.min()), l_max.max() - l_min.min())

        image_out = pil_image_from_tensor(image_tensor)

        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)


BLEND_MODES = Literal[
    "Normal",
    "Lighten Only",
    "Darken Only",
    "Lighten Only (EAL)",
    "Darken Only (EAL)",
    "Hue",
    "Saturation",
    "Color",
    "Luminosity",
    "Linear Dodge (Add)",
    "Subtract",
    "Multiply",
    "Divide",
    "Screen",
    "Overlay",
    "Linear Burn",
    "Difference",
    "Hard Light",
    "Soft Light",
    "Vivid Light",
    "Linear Light",
    "Color Burn",
    "Color Dodge",
]

BLEND_COLOR_SPACES = Literal[
    "RGB", "Linear RGB", "HSL (RGB)", "HSV (RGB)", "Okhsl", "Okhsv", "Oklch (Oklab)", "LCh (CIELab)"
]


@invocation(
    "invokeai_img_blend",
    title="Image Layer Blend",
    tags=["image", "blend", "layer", "alpha", "composite", "dodge", "burn"],
    category="image",
    version="1.2.0",
)
class InvokeImageBlendInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Blend two images together, with optional opacity, mask, and blend modes. Originally created by @dwringer"""

    layer_upper: ImageField = InputField(description="The top image to blend", ui_order=1)
    blend_mode: BLEND_MODES = InputField(default="Normal", description="Available blend modes", ui_order=2)
    opacity: float = InputField(ge=0, default=1.0, description="Desired opacity of the upper layer", ui_order=3)
    mask: Optional[ImageField] = InputField(
        default=None, description="Optional mask, used to restrict areas from blending", ui_order=4
    )
    fit_to_width: bool = InputField(default=False, description="Scale upper layer to fit base width", ui_order=5)
    fit_to_height: bool = InputField(default=True, description="Scale upper layer to fit base height", ui_order=6)
    layer_base: ImageField = InputField(description="The bottom image to blend", ui_order=7)
    color_space: BLEND_COLOR_SPACES = InputField(
        default="RGB", description="Available color spaces for blend computations", ui_order=8
    )
    adaptive_gamut: float = InputField(
        ge=0,
        default=0.0,
        description="Adaptive gamut clipping (0=off). Higher prioritizes chroma over lightness",
        ui_order=9,
    )
    high_precision: bool = InputField(
        default=True, description="Use more steps in computing gamut when possible", ui_order=10
    )

    def scale_and_pad_or_crop_to_base(self, image_upper: Image.Image, image_base: Image.Image):
        """Rescale upper image based on self.fill_x and self.fill_y params"""

        aspect_base = image_base.width / image_base.height
        aspect_upper = image_upper.width / image_upper.height
        if self.fit_to_width and self.fit_to_height:
            image_upper = image_upper.resize((image_base.width, image_base.height))
        elif (self.fit_to_width and (aspect_base < aspect_upper)) or (
            self.fit_to_height and (aspect_upper <= aspect_base)
        ):
            image_upper = ImageOps.pad(
                image_upper, (image_base.width, image_base.height), color=tuple([0 for band in image_upper.getbands()])
            )
        elif (self.fit_to_width and (aspect_upper <= aspect_base)) or (
            self.fit_to_height and (aspect_base < aspect_upper)
        ):
            image_upper = ImageOps.fit(image_upper, (image_base.width, image_base.height))
        return image_upper

    def image_convert_with_xform(self, image_in: Image.Image, from_mode: str, to_mode: str):
        """Use PIL ImageCms color management to convert 3-channel image from one mode to another"""

        def fixed_mode(mode: str):
            if mode.lower() == "srgb":
                return "rgb"
            elif mode.lower() == "cielab":
                return "lab"
            else:
                return mode.lower()

        from_mode, to_mode = fixed_mode(from_mode), fixed_mode(to_mode)

        profile_srgb = None
        profile_uplab = None
        profile_lab = None
        if (from_mode.lower() == "rgb") or (to_mode.lower() == "rgb"):
            profile_srgb = ImageCms.createProfile("sRGB")
        if (from_mode.lower() == "uplab") or (to_mode.lower() == "uplab"):
            if os.path.isfile("CIELab_to_UPLab.icc"):
                profile_uplab = ImageCms.getOpenProfile("CIELab_to_UPLab.icc")
        if (from_mode.lower() in ["lab", "cielab", "uplab"]) or (to_mode.lower() in ["lab", "cielab", "uplab"]):
            if profile_uplab is None:
                profile_lab = ImageCms.createProfile("LAB", colorTemp=6500)
            else:
                profile_lab = ImageCms.createProfile("LAB", colorTemp=5000)

        xform_rgb_to_lab = None
        xform_uplab_to_lab = None
        xform_lab_to_uplab = None
        xform_lab_to_rgb = None
        if from_mode == "rgb":
            xform_rgb_to_lab = ImageCms.buildTransformFromOpenProfiles(
                profile_srgb, profile_lab, "RGB", "LAB", renderingIntent=2, flags=0x2400
            )
        elif from_mode == "uplab":
            xform_uplab_to_lab = ImageCms.buildTransformFromOpenProfiles(
                profile_uplab, profile_lab, "LAB", "LAB", renderingIntent=2, flags=0x2400
            )
        if to_mode == "uplab":
            xform_lab_to_uplab = ImageCms.buildTransformFromOpenProfiles(
                profile_lab, profile_uplab, "LAB", "LAB", renderingIntent=2, flags=0x2400
            )
        elif to_mode == "rgb":
            xform_lab_to_rgb = ImageCms.buildTransformFromOpenProfiles(
                profile_lab, profile_srgb, "LAB", "RGB", renderingIntent=2, flags=0x2400
            )

        image_out = None
        if (from_mode == "rgb") and (to_mode == "lab"):
            image_out = ImageCms.applyTransform(image_in, xform_rgb_to_lab)
        elif (from_mode == "rgb") and (to_mode == "uplab"):
            image_out = ImageCms.applyTransform(image_in, xform_rgb_to_lab)
            image_out = ImageCms.applyTransform(image_out, xform_lab_to_uplab)
        elif (from_mode == "lab") and (to_mode == "uplab"):
            image_out = ImageCms.applyTransform(image_in, xform_lab_to_uplab)
        elif (from_mode == "lab") and (to_mode == "rgb"):
            image_out = ImageCms.applyTransform(image_in, xform_lab_to_rgb)
        elif (from_mode == "uplab") and (to_mode == "lab"):
            image_out = ImageCms.applyTransform(image_in, xform_uplab_to_lab)
        elif (from_mode == "uplab") and (to_mode == "rgb"):
            image_out = ImageCms.applyTransform(image_in, xform_uplab_to_lab)
            image_out = ImageCms.applyTransform(image_out, xform_lab_to_rgb)

        return image_out

    def prepare_tensors_from_images(
        self,
        image_upper: Image.Image,
        image_lower: Image.Image,
        mask_image: Optional[Image.Image] = None,
        required: Optional[list[str]] = None,
    ):
        """Convert image to the necessary image space representations for blend calculations"""
        required = required or ["hsv", "hsl", "lch", "oklch", "okhsl", "okhsv", "l_eal"]
        alpha_upper, alpha_lower = None, None
        if image_upper.mode == "RGBA":
            # Prepare tensors to compute blend
            image_rgba_upper = image_upper.convert("RGBA")
            alpha_upper = image_rgba_upper.getchannel("A")
            image_upper = image_upper.convert("RGB")
        else:
            if not (image_upper.mode == "RGB"):
                image_upper = image_upper.convert("RGB")
        if image_lower.mode == "RGBA":
            # Prepare tensors to compute blend
            image_rgba_lower = image_lower.convert("RGBA")
            alpha_lower = image_rgba_lower.getchannel("A")
            image_lower = image_lower.convert("RGB")
        else:
            if not (image_lower.mode == "RGB"):
                image_lower = image_lower.convert("RGB")

        image_lab_upper, image_lab_lower = None, None
        upper_lab_tensor, lower_lab_tensor = None, None
        upper_lch_tensor, lower_lch_tensor = None, None
        if "lch" in required:
            image_lab_upper, image_lab_lower = (
                self.image_convert_with_xform(image_upper, "rgb", "lab"),
                self.image_convert_with_xform(image_lower, "rgb", "lab"),
            )

            upper_lab_tensor = torch.stack(
                [
                    tensor_from_pil_image(image_lab_upper.getchannel("L"), normalize=False)[0, :, :],
                    tensor_from_pil_image(image_lab_upper.getchannel("A"), normalize=True)[0, :, :],
                    tensor_from_pil_image(image_lab_upper.getchannel("B"), normalize=True)[0, :, :],
                ]
            )
            lower_lab_tensor = torch.stack(
                [
                    tensor_from_pil_image(image_lab_lower.getchannel("L"), normalize=False)[0, :, :],
                    tensor_from_pil_image(image_lab_lower.getchannel("A"), normalize=True)[0, :, :],
                    tensor_from_pil_image(image_lab_lower.getchannel("B"), normalize=True)[0, :, :],
                ]
            )
            upper_lch_tensor = torch.stack(
                [
                    upper_lab_tensor[0, :, :],
                    torch.sqrt(
                        torch.add(torch.pow(upper_lab_tensor[1, :, :], 2.0), torch.pow(upper_lab_tensor[2, :, :], 2.0))
                    ),
                    torch.atan2(upper_lab_tensor[2, :, :], upper_lab_tensor[1, :, :]),
                ]
            )
            lower_lch_tensor = torch.stack(
                [
                    lower_lab_tensor[0, :, :],
                    torch.sqrt(
                        torch.add(torch.pow(lower_lab_tensor[1, :, :], 2.0), torch.pow(lower_lab_tensor[2, :, :], 2.0))
                    ),
                    torch.atan2(lower_lab_tensor[2, :, :], lower_lab_tensor[1, :, :]),
                ]
            )

        upper_l_eal_tensor, lower_l_eal_tensor = None, None
        if "l_eal" in required:
            upper_l_eal_tensor = equivalent_achromatic_lightness(upper_lch_tensor)
            lower_l_eal_tensor = equivalent_achromatic_lightness(lower_lch_tensor)

        image_hsv_upper, image_hsv_lower = None, None
        upper_hsv_tensor, lower_hsv_tensor = None, None
        if "hsv" in required:
            image_hsv_upper, image_hsv_lower = image_upper.convert("HSV"), image_lower.convert("HSV")
            upper_hsv_tensor = torch.stack(
                [
                    tensor_from_pil_image(image_hsv_upper.getchannel("H"), normalize=False)[0, :, :],
                    tensor_from_pil_image(image_hsv_upper.getchannel("S"), normalize=False)[0, :, :],
                    tensor_from_pil_image(image_hsv_upper.getchannel("V"), normalize=False)[0, :, :],
                ]
            )
            lower_hsv_tensor = torch.stack(
                [
                    tensor_from_pil_image(image_hsv_lower.getchannel("H"), normalize=False)[0, :, :],
                    tensor_from_pil_image(image_hsv_lower.getchannel("S"), normalize=False)[0, :, :],
                    tensor_from_pil_image(image_hsv_lower.getchannel("V"), normalize=False)[0, :, :],
                ]
            )

        upper_rgb_tensor = tensor_from_pil_image(image_upper, normalize=False)
        lower_rgb_tensor = tensor_from_pil_image(image_lower, normalize=False)

        alpha_upper_tensor, alpha_lower_tensor = None, None
        if alpha_upper is None:
            alpha_upper_tensor = torch.ones(upper_rgb_tensor[0, :, :].shape)
        else:
            alpha_upper_tensor = tensor_from_pil_image(alpha_upper, normalize=False)[0, :, :]
        if alpha_lower is None:
            alpha_lower_tensor = torch.ones(lower_rgb_tensor[0, :, :].shape)
        else:
            alpha_lower_tensor = tensor_from_pil_image(alpha_lower, normalize=False)[0, :, :]

        mask_tensor = None
        if mask_image is not None:
            mask_tensor = tensor_from_pil_image(mask_image.convert("L"), normalize=False)[0, :, :]

        upper_hsl_tensor, lower_hsl_tensor = None, None
        if "hsl" in required:
            upper_hsl_tensor = hsl_from_srgb(upper_rgb_tensor)
            lower_hsl_tensor = hsl_from_srgb(lower_rgb_tensor)

        upper_okhsl_tensor, lower_okhsl_tensor = None, None
        if "okhsl" in required:
            upper_okhsl_tensor = okhsl_from_srgb(upper_rgb_tensor, steps=(3 if self.high_precision else 1))
            lower_okhsl_tensor = okhsl_from_srgb(lower_rgb_tensor, steps=(3 if self.high_precision else 1))

        upper_okhsv_tensor, lower_okhsv_tensor = None, None
        if "okhsv" in required:
            upper_okhsv_tensor = okhsv_from_srgb(upper_rgb_tensor, steps=(3 if self.high_precision else 1))
            lower_okhsv_tensor = okhsv_from_srgb(lower_rgb_tensor, steps=(3 if self.high_precision else 1))

        upper_rgb_l_tensor = linear_srgb_from_srgb(upper_rgb_tensor)
        lower_rgb_l_tensor = linear_srgb_from_srgb(lower_rgb_tensor)

        upper_oklab_tensor, lower_oklab_tensor = None, None
        upper_oklch_tensor, lower_oklch_tensor = None, None
        if "oklch" in required:
            upper_oklab_tensor = oklab_from_linear_srgb(upper_rgb_l_tensor)
            lower_oklab_tensor = oklab_from_linear_srgb(lower_rgb_l_tensor)

            upper_oklch_tensor = torch.stack(
                [
                    upper_oklab_tensor[0, :, :],
                    torch.sqrt(
                        torch.add(
                            torch.pow(upper_oklab_tensor[1, :, :], 2.0), torch.pow(upper_oklab_tensor[2, :, :], 2.0)
                        )
                    ),
                    torch.atan2(upper_oklab_tensor[2, :, :], upper_oklab_tensor[1, :, :]),
                ]
            )
            lower_oklch_tensor = torch.stack(
                [
                    lower_oklab_tensor[0, :, :],
                    torch.sqrt(
                        torch.add(
                            torch.pow(lower_oklab_tensor[1, :, :], 2.0), torch.pow(lower_oklab_tensor[2, :, :], 2.0)
                        )
                    ),
                    torch.atan2(lower_oklab_tensor[2, :, :], lower_oklab_tensor[1, :, :]),
                ]
            )

        return (
            upper_rgb_l_tensor,
            lower_rgb_l_tensor,
            upper_rgb_tensor,
            lower_rgb_tensor,
            alpha_upper_tensor,
            alpha_lower_tensor,
            mask_tensor,
            upper_hsv_tensor,
            lower_hsv_tensor,
            upper_hsl_tensor,
            lower_hsl_tensor,
            upper_lab_tensor,
            lower_lab_tensor,
            upper_lch_tensor,
            lower_lch_tensor,
            upper_l_eal_tensor,
            lower_l_eal_tensor,
            upper_oklab_tensor,
            lower_oklab_tensor,
            upper_oklch_tensor,
            lower_oklch_tensor,
            upper_okhsv_tensor,
            lower_okhsv_tensor,
            upper_okhsl_tensor,
            lower_okhsl_tensor,
        )

    def apply_blend(self, image_tensors: torch.Tensor):
        """Apply the selected blend mode using the appropriate color space representations"""

        blend_mode = self.blend_mode
        color_space = self.color_space.split()[0]
        if (color_space in ["RGB", "Linear"]) and (blend_mode in ["Hue", "Saturation", "Luminosity", "Color"]):
            color_space = "HSL"

        def adaptive_clipped(rgb_tensor: torch.Tensor, clamp: bool = True, replace_with: float = MAX_FLOAT):
            """Keep elements of the tensor finite"""

            rgb_tensor = remove_nans(rgb_tensor, replace_with=replace_with)

            if 0 < self.adaptive_gamut:
                rgb_tensor = gamut_clip_tensor(
                    rgb_tensor, alpha=self.adaptive_gamut, steps=(3 if self.high_precision else 1)
                )
                rgb_tensor = remove_nans(rgb_tensor, replace_with=replace_with)
            if clamp:  # Use of MAX_FLOAT seems to lead to NaN's coming back in some cases:
                rgb_tensor = rgb_tensor.clamp(0.0, 1.0)

            return rgb_tensor

        reassembly_function = {
            "RGB": lambda t: linear_srgb_from_srgb(t),
            "Linear": lambda t: t,
            "HSL": lambda t: linear_srgb_from_srgb(srgb_from_hsl(t)),
            "HSV": lambda t: linear_srgb_from_srgb(
                tensor_from_pil_image(
                    pil_image_from_tensor(t.clamp(0.0, 1.0), mode="HSV").convert("RGB"), normalize=False
                )
            ),
            "Okhsl": lambda t: linear_srgb_from_srgb(
                srgb_from_okhsl(t, alpha=self.adaptive_gamut, steps=(3 if self.high_precision else 1))
            ),
            "Okhsv": lambda t: linear_srgb_from_srgb(
                srgb_from_okhsv(t, alpha=self.adaptive_gamut, steps=(3 if self.high_precision else 1))
            ),
            "Oklch": lambda t: linear_srgb_from_oklab(
                torch.stack(
                    [
                        t[0, :, :],
                        torch.mul(t[1, :, :], torch.cos(t[2, :, :])),
                        torch.mul(t[1, :, :], torch.sin(t[2, :, :])),
                    ]
                )
            ),
            "LCh": lambda t: linear_srgb_from_srgb(
                tensor_from_pil_image(
                    self.image_convert_with_xform(
                        Image.merge(
                            "LAB",
                            tuple(
                                pil_image_from_tensor(u)
                                for u in [
                                    t[0, :, :].clamp(0.0, 1.0),
                                    torch.div(torch.add(torch.mul(t[1, :, :], torch.cos(t[2, :, :])), 1.0), 2.0),
                                    torch.div(torch.add(torch.mul(t[1, :, :], torch.sin(t[2, :, :])), 1.0), 2.0),
                                ]
                            ),
                        ),
                        "lab",
                        "rgb",
                    ),
                    normalize=False,
                )
            ),
        }[color_space]

        (
            upper_rgb_l_tensor,  # linear-light sRGB
            lower_rgb_l_tensor,  # linear-light sRGB
            upper_rgb_tensor,
            lower_rgb_tensor,
            alpha_upper_tensor,
            alpha_lower_tensor,
            mask_tensor,
            upper_hsv_tensor,  #   h_rgb,   s_hsv,   v_hsv
            lower_hsv_tensor,
            upper_hsl_tensor,  #        ,   s_hsl,   l_hsl
            lower_hsl_tensor,
            upper_lab_tensor,  #   l_lab,   a_lab,   b_lab
            lower_lab_tensor,
            upper_lch_tensor,  #        ,   c_lab,   h_lab
            lower_lch_tensor,
            upper_l_eal_tensor,  # l_eal
            lower_l_eal_tensor,
            upper_oklab_tensor,  # l_oklab, a_oklab, b_oklab
            lower_oklab_tensor,
            upper_oklch_tensor,  #        , c_oklab, h_oklab
            lower_oklch_tensor,
            upper_okhsv_tensor,  # h_okhsv, s_okhsv, v_okhsv
            lower_okhsv_tensor,
            upper_okhsl_tensor,  # h_okhsl, s_okhsl, l_r_oklab
            lower_okhsl_tensor,
        ) = image_tensors

        current_space_tensors = {
            "RGB": [upper_rgb_tensor, lower_rgb_tensor],
            "Linear": [upper_rgb_l_tensor, lower_rgb_l_tensor],
            "HSL": [upper_hsl_tensor, lower_hsl_tensor],
            "HSV": [upper_hsv_tensor, lower_hsv_tensor],
            "Okhsl": [upper_okhsl_tensor, lower_okhsl_tensor],
            "Okhsv": [upper_okhsv_tensor, lower_okhsv_tensor],
            "Oklch": [upper_oklch_tensor, lower_oklch_tensor],
            "LCh": [upper_lch_tensor, lower_lch_tensor],
        }[color_space]
        upper_space_tensor = current_space_tensors[0]
        lower_space_tensor = current_space_tensors[1]

        lightness_index = {
            "RGB": None,
            "Linear": None,
            "HSL": 2,
            "HSV": 2,
            "Okhsl": 2,
            "Okhsv": 2,
            "Oklch": 0,
            "LCh": 0,
        }[color_space]

        saturation_index = {
            "RGB": None,
            "Linear": None,
            "HSL": 1,
            "HSV": 1,
            "Okhsl": 1,
            "Okhsv": 1,
            "Oklch": 1,
            "LCh": 1,
        }[color_space]

        hue_index = {
            "RGB": None,
            "Linear": None,
            "HSL": 0,
            "HSV": 0,
            "Okhsl": 0,
            "Okhsv": 0,
            "Oklch": 2,
            "LCh": 2,
        }[color_space]

        if blend_mode == "Normal":
            upper_rgb_l_tensor = reassembly_function(upper_space_tensor)

        elif blend_mode == "Multiply":
            upper_rgb_l_tensor = reassembly_function(torch.mul(lower_space_tensor, upper_space_tensor))

        elif blend_mode == "Screen":
            upper_rgb_l_tensor = reassembly_function(
                torch.add(
                    torch.mul(
                        torch.mul(
                            torch.add(torch.mul(upper_space_tensor, -1.0), 1.0),
                            torch.add(torch.mul(lower_space_tensor, -1.0), 1.0),
                        ),
                        -1.0,
                    ),
                    1.0,
                )
            )

        elif (blend_mode == "Overlay") or (blend_mode == "Hard Light"):
            subject_of_cond_tensor = lower_space_tensor if (blend_mode == "Overlay") else upper_space_tensor
            if lightness_index is None:
                upper_space_tensor = torch.where(
                    torch.lt(subject_of_cond_tensor, 0.5),
                    torch.mul(torch.mul(lower_space_tensor, upper_space_tensor), 2.0),
                    torch.add(
                        torch.mul(
                            torch.mul(
                                torch.mul(
                                    torch.add(torch.mul(lower_space_tensor, -1.0), 1.0),
                                    torch.add(torch.mul(upper_space_tensor, -1.0), 1.0),
                                ),
                                2.0,
                            ),
                            -1.0,
                        ),
                        1.0,
                    ),
                )
            else:  # TODO: Currently blending only the lightness channel, not really ideal.
                upper_space_tensor[lightness_index, :, :] = torch.where(
                    torch.lt(subject_of_cond_tensor[lightness_index, :, :], 0.5),
                    torch.mul(
                        torch.mul(lower_space_tensor[lightness_index, :, :], upper_space_tensor[lightness_index, :, :]),
                        2.0,
                    ),
                    torch.add(
                        torch.mul(
                            torch.mul(
                                torch.mul(
                                    torch.add(torch.mul(lower_space_tensor[lightness_index, :, :], -1.0), 1.0),
                                    torch.add(torch.mul(upper_space_tensor[lightness_index, :, :], -1.0), 1.0),
                                ),
                                2.0,
                            ),
                            -1.0,
                        ),
                        1.0,
                    ),
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(upper_space_tensor))

        elif blend_mode == "Soft Light":
            if lightness_index is None:
                g_tensor = torch.where(
                    torch.le(lower_space_tensor, 0.25),
                    torch.mul(
                        torch.add(
                            torch.mul(torch.sub(torch.mul(lower_space_tensor, 16.0), 12.0), lower_space_tensor), 4.0
                        ),
                        lower_space_tensor,
                    ),
                    torch.sqrt(lower_space_tensor),
                )
                lower_space_tensor = torch.where(
                    torch.le(upper_space_tensor, 0.5),
                    torch.sub(
                        lower_space_tensor,
                        torch.mul(
                            torch.mul(torch.add(torch.mul(lower_space_tensor, -1.0), 1.0), lower_space_tensor),
                            torch.add(torch.mul(torch.mul(upper_space_tensor, 2.0), -1.0), 1.0),
                        ),
                    ),
                    torch.add(
                        lower_space_tensor,
                        torch.mul(
                            torch.sub(torch.mul(upper_space_tensor, 2.0), 1.0), torch.sub(g_tensor, lower_space_tensor)
                        ),
                    ),
                )
            else:
                print(
                    "\r\nCOND SHAPE:"
                    + str(torch.le(lower_space_tensor[lightness_index, :, :], 0.25).unsqueeze(0).shape)
                    + "\r\n"
                )
                g_tensor = torch.where(  # Calculates all 3 channels but only one is currently used
                    torch.le(lower_space_tensor[lightness_index, :, :], 0.25).expand(upper_space_tensor.shape),
                    torch.mul(
                        torch.add(
                            torch.mul(torch.sub(torch.mul(lower_space_tensor, 16.0), 12.0), lower_space_tensor), 4.0
                        ),
                        lower_space_tensor,
                    ),
                    torch.sqrt(lower_space_tensor),
                )
                lower_space_tensor[lightness_index, :, :] = torch.where(
                    torch.le(upper_space_tensor[lightness_index, :, :], 0.5),
                    torch.sub(
                        lower_space_tensor[lightness_index, :, :],
                        torch.mul(
                            torch.mul(
                                torch.add(torch.mul(lower_space_tensor[lightness_index, :, :], -1.0), 1.0),
                                lower_space_tensor[lightness_index, :, :],
                            ),
                            torch.add(torch.mul(torch.mul(upper_space_tensor[lightness_index, :, :], 2.0), -1.0), 1.0),
                        ),
                    ),
                    torch.add(
                        lower_space_tensor[lightness_index, :, :],
                        torch.mul(
                            torch.sub(torch.mul(upper_space_tensor[lightness_index, :, :], 2.0), 1.0),
                            torch.sub(g_tensor[lightness_index, :, :], lower_space_tensor[lightness_index, :, :]),
                        ),
                    ),
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Linear Dodge (Add)":
            lower_space_tensor = torch.add(lower_space_tensor, upper_space_tensor)
            if hue_index is not None:
                lower_space_tensor[hue_index, :, :] = torch.remainder(lower_space_tensor[hue_index, :, :], 1.0)
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Color Dodge":
            lower_space_tensor = torch.div(lower_space_tensor, torch.add(torch.mul(upper_space_tensor, -1.0), 1.0))
            if hue_index is not None:
                lower_space_tensor[hue_index, :, :] = torch.remainder(lower_space_tensor[hue_index, :, :], 1.0)
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Divide":
            lower_space_tensor = torch.div(lower_space_tensor, upper_space_tensor)
            if hue_index is not None:
                lower_space_tensor[hue_index, :, :] = torch.remainder(lower_space_tensor[hue_index, :, :], 1.0)
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Linear Burn":
            # We compute the result in the lower image's current space tensor and return that:
            if lightness_index is None:  # Elementwise
                lower_space_tensor = torch.sub(torch.add(lower_space_tensor, upper_space_tensor), 1.0)
            else:  # Operate only on the selected lightness channel
                lower_space_tensor[lightness_index, :, :] = torch.sub(
                    torch.add(lower_space_tensor[lightness_index, :, :], upper_space_tensor[lightness_index, :, :]), 1.0
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Color Burn":
            upper_rgb_l_tensor = adaptive_clipped(
                reassembly_function(
                    torch.add(
                        torch.mul(
                            torch.min(
                                torch.div(torch.add(torch.mul(lower_space_tensor, -1.0), 1.0), upper_space_tensor),
                                torch.ones(lower_space_tensor.shape),
                            ),
                            -1.0,
                        ),
                        1.0,
                    )
                )
            )
        elif blend_mode == "Vivid Light":
            if lightness_index is None:
                lower_space_tensor = adaptive_clipped(
                    reassembly_function(
                        torch.where(
                            torch.lt(upper_space_tensor, 0.5),
                            torch.div(
                                torch.add(
                                    torch.mul(
                                        torch.div(
                                            torch.add(torch.mul(lower_space_tensor, -1.0), 1.0), upper_space_tensor
                                        ),
                                        -1.0,
                                    ),
                                    1.0,
                                ),
                                2.0,
                            ),
                            torch.div(
                                torch.div(lower_space_tensor, torch.add(torch.mul(upper_space_tensor, -1.0), 1.0)), 2.0
                            ),
                        )
                    )
                )
            else:
                lower_space_tensor[lightness_index, :, :] = torch.where(
                    torch.lt(upper_space_tensor[lightness_index, :, :], 0.5),
                    torch.div(
                        torch.add(
                            torch.mul(
                                torch.div(
                                    torch.add(torch.mul(lower_space_tensor[lightness_index, :, :], -1.0), 1.0),
                                    upper_space_tensor[lightness_index, :, :],
                                ),
                                -1.0,
                            ),
                            1.0,
                        ),
                        2.0,
                    ),
                    torch.div(
                        torch.div(
                            lower_space_tensor[lightness_index, :, :],
                            torch.add(torch.mul(upper_space_tensor[lightness_index, :, :], -1.0), 1.0),
                        ),
                        2.0,
                    ),
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Linear Light":
            if lightness_index is None:
                lower_space_tensor = torch.sub(torch.add(lower_space_tensor, torch.mul(upper_space_tensor, 2.0)), 1.0)
            else:
                lower_space_tensor[lightness_index, :, :] = torch.sub(
                    torch.add(
                        lower_space_tensor[lightness_index, :, :],
                        torch.mul(upper_space_tensor[lightness_index, :, :], 2.0),
                    ),
                    1.0,
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Subtract":
            lower_space_tensor = torch.sub(lower_space_tensor, upper_space_tensor)
            if hue_index is not None:
                lower_space_tensor[hue_index, :, :] = torch.remainder(lower_space_tensor[hue_index, :, :], 1.0)
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Difference":
            upper_rgb_l_tensor = adaptive_clipped(
                reassembly_function(torch.abs(torch.sub(lower_space_tensor, upper_space_tensor)))
            )

        elif (blend_mode == "Darken Only") or (blend_mode == "Lighten Only"):
            extrema_fn = torch.min if (blend_mode == "Darken Only") else torch.max
            comparator_fn = torch.ge if (blend_mode == "Darken Only") else torch.lt
            if lightness_index is None:
                upper_space_tensor = torch.stack(
                    [
                        extrema_fn(upper_space_tensor[0, :, :], lower_space_tensor[0, :, :]),
                        extrema_fn(upper_space_tensor[1, :, :], lower_space_tensor[1, :, :]),
                        extrema_fn(upper_space_tensor[2, :, :], lower_space_tensor[2, :, :]),
                    ]
                )
            else:
                upper_space_tensor = torch.where(
                    comparator_fn(
                        upper_space_tensor[lightness_index, :, :], lower_space_tensor[lightness_index, :, :]
                    ).expand(upper_space_tensor.shape),
                    lower_space_tensor,
                    upper_space_tensor,
                )
            upper_rgb_l_tensor = reassembly_function(upper_space_tensor)

        elif blend_mode in [
            "Hue",
            "Saturation",
            "Color",
            "Luminosity",
        ]:
            if blend_mode == "Hue":  # l, c: lower / h: upper
                upper_space_tensor[lightness_index, :, :] = lower_space_tensor[lightness_index, :, :]
                upper_space_tensor[saturation_index, :, :] = lower_space_tensor[saturation_index, :, :]
            elif blend_mode == "Saturation":  # l, h: lower / c: upper
                upper_space_tensor[lightness_index, :, :] = lower_space_tensor[lightness_index, :, :]
                upper_space_tensor[hue_index, :, :] = lower_space_tensor[hue_index, :, :]
            elif blend_mode == "Color":  # l: lower / c, h: upper
                upper_space_tensor[lightness_index, :, :] = lower_space_tensor[lightness_index, :, :]
            elif blend_mode == "Luminosity":  # h, c: lower / l: upper
                upper_space_tensor[saturation_index, :, :] = lower_space_tensor[saturation_index, :, :]
                upper_space_tensor[hue_index, :, :] = lower_space_tensor[hue_index, :, :]
            upper_rgb_l_tensor = reassembly_function(upper_space_tensor)

        elif blend_mode in ["Lighten Only (EAL)", "Darken Only (EAL)"]:
            comparator_fn = torch.lt if (blend_mode == "Lighten Only (EAL)") else torch.ge
            upper_space_tensor = torch.where(
                comparator_fn(upper_l_eal_tensor, lower_l_eal_tensor).expand(upper_space_tensor.shape),
                lower_space_tensor,
                upper_space_tensor,
            )
            upper_rgb_l_tensor = reassembly_function(upper_space_tensor)

        return upper_rgb_l_tensor

    def alpha_composite(
        self,
        upper_tensor: torch.Tensor,
        alpha_upper_tensor: torch.Tensor,
        lower_tensor: torch.Tensor,
        alpha_lower_tensor: torch.Tensor,
        mask_tensor: Optional[torch.Tensor] = None,
    ):
        """Alpha compositing of upper on lower tensor with alpha channels, mask and scalar"""

        upper_tensor = remove_nans(upper_tensor)

        alpha_upper_tensor = torch.mul(alpha_upper_tensor, self.opacity)
        if mask_tensor is not None:
            alpha_upper_tensor = torch.mul(alpha_upper_tensor, torch.add(torch.mul(mask_tensor, -1.0), 1.0))

        alpha_tensor = torch.add(
            alpha_upper_tensor, torch.mul(alpha_lower_tensor, torch.add(torch.mul(alpha_upper_tensor, -1.0), 1.0))
        )

        return (
            torch.div(
                torch.add(
                    torch.mul(upper_tensor, alpha_upper_tensor),
                    torch.mul(
                        torch.mul(lower_tensor, alpha_lower_tensor), torch.add(torch.mul(alpha_upper_tensor, -1.0), 1.0)
                    ),
                ),
                alpha_tensor,
            ),
            alpha_tensor,
        )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        """Main execution of the ImageBlendInvocation node"""

        image_upper = context.images.get_pil(self.layer_upper.image_name)
        image_base = context.images.get_pil(self.layer_base.image_name)

        # Keep the modes for restoration after processing:
        image_mode_base = image_base.mode

        # Get rid of ICC profiles by converting to sRGB, but save for restoration:
        cms_profile_srgb = None
        if "icc_profile" in image_upper.info:
            cms_profile_upper = BytesIO(image_upper.info["icc_profile"])
            cms_profile_srgb = ImageCms.createProfile("sRGB")
            cms_xform = ImageCms.buildTransformFromOpenProfiles(
                cms_profile_upper, cms_profile_srgb, image_upper.mode, "RGBA"
            )
            image_upper = ImageCms.applyTransform(image_upper, cms_xform)

        cms_profile_base = None
        icc_profile_bytes = None
        if "icc_profile" in image_base.info:
            icc_profile_bytes = image_base.info["icc_profile"]
            cms_profile_base = BytesIO(icc_profile_bytes)
            if cms_profile_srgb is None:
                cms_profile_srgb = ImageCms.createProfile("sRGB")
            cms_xform = ImageCms.buildTransformFromOpenProfiles(
                cms_profile_base, cms_profile_srgb, image_base.mode, "RGBA"
            )
            image_base = ImageCms.applyTransform(image_base, cms_xform)

        image_mask = None
        if self.mask is not None:
            image_mask = context.images.get_pil(self.mask.image_name)
        color_space = self.color_space.split()[0]

        image_upper = self.scale_and_pad_or_crop_to_base(image_upper, image_base)
        if image_mask is not None:
            image_mask = self.scale_and_pad_or_crop_to_base(image_mask, image_base)

        tensor_requirements = []

        # Hue, Saturation, Color, and Luminosity won't work in sRGB, require HSL
        if self.blend_mode in ["Hue", "Saturation", "Color", "Luminosity"] and self.color_space in [
            "RGB",
            "Linear RGB",
        ]:
            tensor_requirements = ["hsl"]

        if self.blend_mode in ["Lighten Only (EAL)", "Darken Only (EAL)"]:
            tensor_requirements = tensor_requirements + ["lch", "l_eal"]

        tensor_requirements += {
            "Linear": [],
            "RGB": [],
            "HSL": ["hsl"],
            "HSV": ["hsv"],
            "Okhsl": ["okhsl"],
            "Okhsv": ["okhsv"],
            "Oklch": ["oklch"],
            "LCh": ["lch"],
        }[color_space]

        image_tensors = (
            upper_rgb_l_tensor,  # linear-light sRGB
            lower_rgb_l_tensor,  # linear-light sRGB
            upper_rgb_tensor,
            lower_rgb_tensor,
            alpha_upper_tensor,
            alpha_lower_tensor,
            mask_tensor,
            upper_hsv_tensor,
            lower_hsv_tensor,
            upper_hsl_tensor,
            lower_hsl_tensor,
            upper_lab_tensor,
            lower_lab_tensor,
            upper_lch_tensor,
            lower_lch_tensor,
            upper_l_eal_tensor,
            lower_l_eal_tensor,
            upper_oklab_tensor,
            lower_oklab_tensor,
            upper_oklch_tensor,
            lower_oklch_tensor,
            upper_okhsv_tensor,
            lower_okhsv_tensor,
            upper_okhsl_tensor,
            lower_okhsl_tensor,
        ) = self.prepare_tensors_from_images(
            image_upper, image_base, mask_image=image_mask, required=tensor_requirements
        )

        #        if not (self.blend_mode == "Normal"):
        upper_rgb_l_tensor = self.apply_blend(image_tensors)

        output_tensor, alpha_tensor = self.alpha_composite(
            srgb_from_linear_srgb(
                upper_rgb_l_tensor, alpha=self.adaptive_gamut, steps=(3 if self.high_precision else 1)
            ),
            alpha_upper_tensor,
            lower_rgb_tensor,
            alpha_lower_tensor,
            mask_tensor=mask_tensor,
        )

        # Restore alpha channel and base mode:
        output_tensor = torch.stack(
            [output_tensor[0, :, :], output_tensor[1, :, :], output_tensor[2, :, :], alpha_tensor]
        )
        image_out = pil_image_from_tensor(output_tensor, mode="RGBA")

        # Restore ICC profile if base image had one:
        if cms_profile_base is not None:
            cms_xform = ImageCms.buildTransformFromOpenProfiles(
                cms_profile_srgb, BytesIO(icc_profile_bytes), "RGBA", image_out.mode
            )
            image_out = ImageCms.applyTransform(image_out, cms_xform)
        else:
            image_out = image_out.convert(image_mode_base)

        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)


@invocation(
    "invokeai_img_composite",
    title="Image Compositor",
    tags=["image", "compose", "chroma", "key"],
    category="image",
    version="1.2.0",
)
class InvokeImageCompositorInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Removes backdrop from subject image then overlays subject on background image. Originally created by @dwringer"""

    image_subject: ImageField = InputField(description="Image of the subject on a plain monochrome background")
    image_background: ImageField = InputField(description="Image of a background scene")
    chroma_key: str = InputField(
        default="", description="Can be empty for corner flood select, or CSS-3 color or tuple"
    )
    threshold: int = InputField(ge=0, default=50, description="Subject isolation flood-fill threshold")
    fill_x: bool = InputField(default=False, description="Scale base subject image to fit background width")
    fill_y: bool = InputField(default=True, description="Scale base subject image to fit background height")
    x_offset: int = InputField(default=0, description="x-offset for the subject")
    y_offset: int = InputField(default=0, description="y-offset for the subject")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_background = context.images.get_pil(self.image_background.image_name).convert(mode="RGBA")
        image_subject = context.images.get_pil(self.image_subject.image_name).convert(mode="RGBA")

        if image_subject.height == 0 or image_subject.width == 0:
            raise ValueError("The subject image has zero height or width")
        if image_background.height == 0 or image_background.width == 0:
            raise ValueError("The subject image has zero height or width")

        # Handle backdrop removal:
        chroma_key = self.chroma_key.strip()
        if 0 < len(chroma_key):
            # Remove pixels by chroma key:
            if chroma_key[0] == "(":
                chroma_key = tuple_from_string(chroma_key)
                while len(chroma_key) < 3:
                    chroma_key = tuple(list(chroma_key) + [0])
                if len(chroma_key) == 3:
                    chroma_key = tuple(list(chroma_key) + [255])
            else:
                chroma_key = ImageColor.getcolor(chroma_key, "RGBA")
            threshold = self.threshold**2.0  # to compare vs squared color distance from key
            pixels = image_subject.load()
            if pixels is None:
                raise ValueError("Unable to load pixels from subject image")
            for i in range(image_subject.width):
                for j in range(image_subject.height):
                    if (
                        reduce(
                            lambda a, b: a + b, [(pixels[i, j][k] - chroma_key[k]) ** 2 for k in range(len(chroma_key))]
                        )
                        < threshold
                    ):
                        pixels[i, j] = tuple([0 for k in range(len(chroma_key))])
        else:
            # Remove pixels by flood select from corners:
            ImageDraw.floodfill(image_subject, (0, 0), (0, 0, 0, 0), thresh=self.threshold)
            ImageDraw.floodfill(image_subject, (0, image_subject.height - 1), (0, 0, 0, 0), thresh=self.threshold)
            ImageDraw.floodfill(image_subject, (image_subject.width - 1, 0), (0, 0, 0, 0), thresh=self.threshold)
            ImageDraw.floodfill(
                image_subject, (image_subject.width - 1, image_subject.height - 1), (0, 0, 0, 0), thresh=self.threshold
            )

        # Scale and position the subject:
        aspect_background = image_background.width / image_background.height
        aspect_subject = image_subject.width / image_subject.height
        if self.fill_x and self.fill_y:
            image_subject = image_subject.resize((image_background.width, image_background.height))
        elif (self.fill_x and (aspect_background < aspect_subject)) or (
            self.fill_y and (aspect_subject <= aspect_background)
        ):
            image_subject = ImageOps.pad(
                image_subject, (image_background.width, image_background.height), color=(0, 0, 0, 0)
            )
        elif (self.fill_x and (aspect_subject <= aspect_background)) or (
            self.fill_y and (aspect_background < aspect_subject)
        ):
            image_subject = ImageOps.fit(image_subject, (image_background.width, image_background.height))
        if (self.x_offset != 0) or (self.y_offset != 0):
            image_subject = ImageChops.offset(image_subject, self.x_offset, yoffset=-1 * self.y_offset)

        new_image = Image.alpha_composite(image_background, image_subject)
        new_image.convert(mode="RGB")
        image_dto = context.images.save(new_image)

        return ImageOutput.build(image_dto)


DILATE_ERODE_MODES = Literal[
    "Dilate",
    "Erode",
]


@invocation(
    "invokeai_img_dilate_erode",
    title="Image Dilate or Erode",
    tags=["image", "mask", "dilate", "erode", "expand", "contract", "mask"],
    category="image",
    version="1.3.0",
)
class InvokeImageDilateOrErodeInvocation(BaseInvocation, WithMetadata):
    """Dilate (expand) or erode (contract) an image. Originally created by @dwringer"""

    image: ImageField = InputField(description="The image from which to create a mask")
    lightness_only: bool = InputField(default=False, description="If true, only applies to image lightness (CIELa*b*)")
    radius_w: int = InputField(
        ge=0, default=4, description="Width (in pixels) by which to dilate(expand) or erode (contract) the image"
    )
    radius_h: int = InputField(
        ge=0, default=4, description="Height (in pixels) by which to dilate(expand) or erode (contract) the image"
    )
    mode: DILATE_ERODE_MODES = InputField(default="Dilate", description="How to operate on the image")

    def expand_or_contract(self, image_in: Image.Image):
        image_out = numpy.array(image_in)
        expand_radius_w = self.radius_w
        expand_radius_h = self.radius_h

        expand_fn = None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_radius_w * 2 + 1, expand_radius_h * 2 + 1))
        if self.mode == "Dilate":
            expand_fn = cv2.dilate
        elif self.mode == "Erode":
            expand_fn = cv2.erode
        else:
            raise ValueError("Invalid mode selected")
        image_out = expand_fn(image_out, kernel, iterations=1)
        return Image.fromarray(image_out, mode=image_in.mode)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.images.get_pil(self.image.image_name)
        image_out = image_in

        if self.lightness_only:
            image_mode = image_in.mode
            alpha_channel = None
            if (image_mode == "RGBA") or (image_mode == "LA") or (image_mode == "PA"):
                alpha_channel = image_in.getchannel("A")
            elif (image_mode == "RGBa") or (image_mode == "La") or (image_mode == "Pa"):
                alpha_channel = image_in.getchannel("a")
            if (image_mode == "RGBA") or (image_mode == "RGBa"):
                image_mode = "RGB"
            elif (image_mode == "LA") or (image_mode == "La"):
                image_mode = "L"
            elif image_mode == "PA":
                image_mode = "P"
            image_out = image_out.convert("RGB")
            image_out = image_out.convert("LAB")
            l_channel = self.expand_or_contract(image_out.getchannel("L"))
            image_out = Image.merge("LAB", (l_channel, image_out.getchannel("A"), image_out.getchannel("B")))
            if (image_mode == "L") or (image_mode == "P"):
                image_out = image_out.convert("RGB")
            image_out = image_out.convert(image_mode)
            if "a" in image_in.mode.lower():
                image_out = Image.merge(
                    image_in.mode, tuple([image_out.getchannel(c) for c in image_mode] + [alpha_channel])
                )
        else:
            image_out = self.expand_or_contract(image_out)

        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)


@invocation(
    "invokeai_img_val_thresholds",
    title="Image Value Thresholds",
    tags=["image", "mask", "value", "threshold"],
    category="image",
    version="1.2.0",
)
class InvokeImageValueThresholdsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Clip image to pure black/white past specified thresholds. Originally created by @dwringer"""

    image: ImageField = InputField(description="The image from which to create a mask")
    invert_output: bool = InputField(default=False, description="Make light areas dark and vice versa")
    renormalize_values: bool = InputField(default=False, description="Rescale remaining values from minimum to maximum")
    lightness_only: bool = InputField(default=False, description="If true, only applies to image lightness (CIELa*b*)")
    threshold_upper: float = InputField(default=0.5, description="Threshold above which will be set to full value")
    threshold_lower: float = InputField(default=0.5, description="Threshold below which will be set to minimum value")

    def get_threshold_mask(self, image_tensor: torch.Tensor):
        img_tensor = image_tensor.clone()
        threshold_h, threshold_s = self.threshold_upper, self.threshold_lower
        ones_tensor = torch.ones(img_tensor.shape)
        zeros_tensor = torch.zeros(img_tensor.shape)

        zeros_mask, ones_mask = None, None
        if self.invert_output:
            zeros_mask, ones_mask = torch.ge(img_tensor, threshold_h), torch.lt(img_tensor, threshold_s)
        else:
            ones_mask, zeros_mask = torch.ge(img_tensor, threshold_h), torch.lt(img_tensor, threshold_s)

        if not (threshold_h == threshold_s):
            mask_hi = torch.ge(img_tensor, threshold_s)
            mask_lo = torch.lt(img_tensor, threshold_h)
            mask = torch.logical_and(mask_hi, mask_lo)
            masked = img_tensor[mask]
            if 0 < masked.numel():
                if self.renormalize_values:
                    vmax, vmin = max(threshold_h, threshold_s), min(threshold_h, threshold_s)
                    if vmax == vmin:
                        img_tensor[mask] = vmin * ones_tensor[mask]
                    elif self.invert_output:
                        img_tensor[mask] = torch.sub(1.0, (img_tensor[mask] - vmin) / (vmax - vmin))
                    else:
                        img_tensor[mask] = (img_tensor[mask] - vmin) / (vmax - vmin)

        img_tensor[ones_mask] = ones_tensor[ones_mask]
        img_tensor[zeros_mask] = zeros_tensor[zeros_mask]

        return img_tensor

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.images.get_pil(self.image.image_name)

        if self.lightness_only:
            image_mode = image_in.mode
            alpha_channel = None
            if (image_mode == "RGBA") or (image_mode == "LA") or (image_mode == "PA"):
                alpha_channel = image_in.getchannel("A")
            elif (image_mode == "RGBa") or (image_mode == "La") or (image_mode == "Pa"):
                alpha_channel = image_in.getchannel("a")
            if (image_mode == "RGBA") or (image_mode == "RGBa"):
                image_mode = "RGB"
            elif (image_mode == "LA") or (image_mode == "La"):
                image_mode = "L"
            elif image_mode == "PA":
                image_mode = "P"
            image_out = image_in.convert("RGB")
            image_out = image_out.convert("LAB")

            l_channel = image_resized_to_grid_as_tensor(image_out.getchannel("L"), normalize=False)
            l_channel = self.get_threshold_mask(l_channel)
            l_channel = pil_image_from_tensor(l_channel)

            image_out = Image.merge("LAB", (l_channel, image_out.getchannel("A"), image_out.getchannel("B")))
            if (image_mode == "L") or (image_mode == "P"):
                image_out = image_out.convert("RGB")
            image_out = image_out.convert(image_mode)
            if "a" in image_in.mode.lower():
                image_out = Image.merge(
                    image_in.mode, tuple([image_out.getchannel(c) for c in image_mode] + [alpha_channel])
                )
        else:
            image_out = image_resized_to_grid_as_tensor(image_in, normalize=False)
            image_out = self.get_threshold_mask(image_out)
            image_out = pil_image_from_tensor(image_out)

        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)
