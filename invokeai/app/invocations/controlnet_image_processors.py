# Invocations for ControlNet image preprocessors
# initial implementation by Gregg Helt, 2023
# heavily leverages controlnet_aux package: https://github.com/patrickvonplaten/controlnet_aux
from builtins import bool, float
from typing import Dict, List, Literal, Union

import cv2
import numpy as np
from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LeresDetector,
    LineartAnimeDetector,
    LineartDetector,
    MediapipeFaceDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    PidiNetDetector,
    SamDetector,
    ZoeDetector,
)
from controlnet_aux.util import HWC3, ade_palette
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from invokeai.app.invocations.baseinvocation import WithMetadata
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, Input, InputField, OutputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.invocations.util import validate_begin_end_step, validate_weights
from invokeai.backend.image_util.depth_anything import DepthAnythingDetector
from invokeai.backend.image_util.dw_openpose import DWOpenposeDetector
from invokeai.backend.model_management.models.base import BaseModelType

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)

CONTROLNET_MODE_VALUES = Literal["balanced", "more_prompt", "more_control", "unbalanced"]
CONTROLNET_RESIZE_VALUES = Literal[
    "just_resize",
    "crop_resize",
    "fill_resize",
    "just_resize_simple",
]


class ControlNetModelField(BaseModel):
    """ControlNet model field"""

    model_name: str = Field(description="Name of the ControlNet model")
    base_model: BaseModelType = Field(description="Base model")

    model_config = ConfigDict(protected_namespaces=())


class ControlField(BaseModel):
    image: ImageField = Field(description="The control image")
    control_model: ControlNetModelField = Field(description="The ControlNet model to use")
    control_weight: Union[float, List[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode to use")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")

    @field_validator("control_weight")
    @classmethod
    def validate_control_weight(cls, v):
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self


@invocation_output("control_output")
class ControlOutput(BaseInvocationOutput):
    """node output for ControlNet info"""

    # Outputs
    control: ControlField = OutputField(description=FieldDescriptions.control)


@invocation("controlnet", title="ControlNet", tags=["controlnet"], category="controlnet", version="1.1.1")
class ControlNetInvocation(BaseInvocation):
    """Collects ControlNet info to pass to other nodes"""

    image: ImageField = InputField(description="The control image")
    control_model: ControlNetModelField = InputField(description=FieldDescriptions.controlnet_model, input=Input.Direct)
    control_weight: Union[float, List[float]] = InputField(
        default=1.0, ge=-1, le=2, description="The weight given to the ControlNet"
    )
    begin_step_percent: float = InputField(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = InputField(default="balanced", description="The control mode used")
    resize_mode: CONTROLNET_RESIZE_VALUES = InputField(default="just_resize", description="The resize mode used")

    @field_validator("control_weight")
    @classmethod
    def validate_control_weight(cls, v):
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self) -> "ControlNetInvocation":
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context) -> ControlOutput:
        return ControlOutput(
            control=ControlField(
                image=self.image,
                control_model=self.control_model,
                control_weight=self.control_weight,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                control_mode=self.control_mode,
                resize_mode=self.resize_mode,
            ),
        )


# This invocation exists for other invocations to subclass it - do not register with @invocation!
class ImageProcessorInvocation(BaseInvocation, WithMetadata):
    """Base class for invocations that preprocess images for ControlNet"""

    image: ImageField = InputField(description="The image to process")

    def run_processor(self, image: Image.Image) -> Image.Image:
        # superclass just passes through image without processing
        return image

    def invoke(self, context) -> ImageOutput:
        raw_image = context.images.get_pil(self.image.image_name)
        # image type should be PIL.PngImagePlugin.PngImageFile ?
        processed_image = self.run_processor(raw_image)

        # currently can't see processed image in node UI without a showImage node,
        #    so for now setting image_type to RESULT instead of INTERMEDIATE so will get saved in gallery
        image_dto = context.images.save(image=processed_image)

        """Builds an ImageOutput and its ImageField"""
        processed_image_field = ImageField(image_name=image_dto.image_name)
        return ImageOutput(
            image=processed_image_field,
            # width=processed_image.width,
            width=image_dto.width,
            # height=processed_image.height,
            height=image_dto.height,
            # mode=processed_image.mode,
        )


@invocation(
    "canny_image_processor",
    title="Canny Processor",
    tags=["controlnet", "canny"],
    category="controlnet",
    version="1.2.1",
)
class CannyImageProcessorInvocation(ImageProcessorInvocation):
    """Canny edge detection for ControlNet"""

    low_threshold: int = InputField(
        default=100, ge=0, le=255, description="The low threshold of the Canny pixel gradient (0-255)"
    )
    high_threshold: int = InputField(
        default=200, ge=0, le=255, description="The high threshold of the Canny pixel gradient (0-255)"
    )

    def run_processor(self, image):
        canny_processor = CannyDetector()
        processed_image = canny_processor(image, self.low_threshold, self.high_threshold)
        return processed_image


@invocation(
    "hed_image_processor",
    title="HED (softedge) Processor",
    tags=["controlnet", "hed", "softedge"],
    category="controlnet",
    version="1.2.1",
)
class HedImageProcessorInvocation(ImageProcessorInvocation):
    """Applies HED edge detection to image"""

    detect_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.detect_res)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)
    # safe not supported in controlnet_aux v0.0.3
    # safe: bool = InputField(default=False, description=FieldDescriptions.safe_mode)
    scribble: bool = InputField(default=False, description=FieldDescriptions.scribble_mode)

    def run_processor(self, image):
        hed_processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
        processed_image = hed_processor(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
            # safe not supported in controlnet_aux v0.0.3
            # safe=self.safe,
            scribble=self.scribble,
        )
        return processed_image


@invocation(
    "lineart_image_processor",
    title="Lineart Processor",
    tags=["controlnet", "lineart"],
    category="controlnet",
    version="1.2.1",
)
class LineartImageProcessorInvocation(ImageProcessorInvocation):
    """Applies line art processing to image"""

    detect_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.detect_res)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)
    coarse: bool = InputField(default=False, description="Whether to use coarse mode")

    def run_processor(self, image):
        lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = lineart_processor(
            image, detect_resolution=self.detect_resolution, image_resolution=self.image_resolution, coarse=self.coarse
        )
        return processed_image


@invocation(
    "lineart_anime_image_processor",
    title="Lineart Anime Processor",
    tags=["controlnet", "lineart", "anime"],
    category="controlnet",
    version="1.2.1",
)
class LineartAnimeImageProcessorInvocation(ImageProcessorInvocation):
    """Applies line art anime processing to image"""

    detect_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.detect_res)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)

    def run_processor(self, image):
        processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = processor(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
        )
        return processed_image


@invocation(
    "midas_depth_image_processor",
    title="Midas Depth Processor",
    tags=["controlnet", "midas"],
    category="controlnet",
    version="1.2.1",
)
class MidasDepthImageProcessorInvocation(ImageProcessorInvocation):
    """Applies Midas depth processing to image"""

    a_mult: float = InputField(default=2.0, ge=0, description="Midas parameter `a_mult` (a = a_mult * PI)")
    bg_th: float = InputField(default=0.1, ge=0, description="Midas parameter `bg_th`")
    # depth_and_normal not supported in controlnet_aux v0.0.3
    # depth_and_normal: bool = InputField(default=False, description="whether to use depth and normal mode")

    def run_processor(self, image):
        midas_processor = MidasDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = midas_processor(
            image,
            a=np.pi * self.a_mult,
            bg_th=self.bg_th,
            # dept_and_normal not supported in controlnet_aux v0.0.3
            # depth_and_normal=self.depth_and_normal,
        )
        return processed_image


@invocation(
    "normalbae_image_processor",
    title="Normal BAE Processor",
    tags=["controlnet"],
    category="controlnet",
    version="1.2.1",
)
class NormalbaeImageProcessorInvocation(ImageProcessorInvocation):
    """Applies NormalBae processing to image"""

    detect_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.detect_res)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)

    def run_processor(self, image):
        normalbae_processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = normalbae_processor(
            image, detect_resolution=self.detect_resolution, image_resolution=self.image_resolution
        )
        return processed_image


@invocation(
    "mlsd_image_processor", title="MLSD Processor", tags=["controlnet", "mlsd"], category="controlnet", version="1.2.1"
)
class MlsdImageProcessorInvocation(ImageProcessorInvocation):
    """Applies MLSD processing to image"""

    detect_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.detect_res)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)
    thr_v: float = InputField(default=0.1, ge=0, description="MLSD parameter `thr_v`")
    thr_d: float = InputField(default=0.1, ge=0, description="MLSD parameter `thr_d`")

    def run_processor(self, image):
        mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")
        processed_image = mlsd_processor(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
            thr_v=self.thr_v,
            thr_d=self.thr_d,
        )
        return processed_image


@invocation(
    "pidi_image_processor", title="PIDI Processor", tags=["controlnet", "pidi"], category="controlnet", version="1.2.1"
)
class PidiImageProcessorInvocation(ImageProcessorInvocation):
    """Applies PIDI processing to image"""

    detect_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.detect_res)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)
    safe: bool = InputField(default=False, description=FieldDescriptions.safe_mode)
    scribble: bool = InputField(default=False, description=FieldDescriptions.scribble_mode)

    def run_processor(self, image):
        pidi_processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = pidi_processor(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
            safe=self.safe,
            scribble=self.scribble,
        )
        return processed_image


@invocation(
    "content_shuffle_image_processor",
    title="Content Shuffle Processor",
    tags=["controlnet", "contentshuffle"],
    category="controlnet",
    version="1.2.1",
)
class ContentShuffleImageProcessorInvocation(ImageProcessorInvocation):
    """Applies content shuffle processing to image"""

    detect_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.detect_res)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)
    h: int = InputField(default=512, ge=0, description="Content shuffle `h` parameter")
    w: int = InputField(default=512, ge=0, description="Content shuffle `w` parameter")
    f: int = InputField(default=256, ge=0, description="Content shuffle `f` parameter")

    def run_processor(self, image):
        content_shuffle_processor = ContentShuffleDetector()
        processed_image = content_shuffle_processor(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
            h=self.h,
            w=self.w,
            f=self.f,
        )
        return processed_image


# should work with controlnet_aux >= 0.0.4 and timm <= 0.6.13
@invocation(
    "zoe_depth_image_processor",
    title="Zoe (Depth) Processor",
    tags=["controlnet", "zoe", "depth"],
    category="controlnet",
    version="1.2.1",
)
class ZoeDepthImageProcessorInvocation(ImageProcessorInvocation):
    """Applies Zoe depth processing to image"""

    def run_processor(self, image):
        zoe_depth_processor = ZoeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = zoe_depth_processor(image)
        return processed_image


@invocation(
    "mediapipe_face_processor",
    title="Mediapipe Face Processor",
    tags=["controlnet", "mediapipe", "face"],
    category="controlnet",
    version="1.2.1",
)
class MediapipeFaceProcessorInvocation(ImageProcessorInvocation):
    """Applies mediapipe face processing to image"""

    max_faces: int = InputField(default=1, ge=1, description="Maximum number of faces to detect")
    min_confidence: float = InputField(default=0.5, ge=0, le=1, description="Minimum confidence for face detection")

    def run_processor(self, image):
        # MediaPipeFaceDetector throws an error if image has alpha channel
        #     so convert to RGB if needed
        if image.mode == "RGBA":
            image = image.convert("RGB")
        mediapipe_face_processor = MediapipeFaceDetector()
        processed_image = mediapipe_face_processor(image, max_faces=self.max_faces, min_confidence=self.min_confidence)
        return processed_image


@invocation(
    "leres_image_processor",
    title="Leres (Depth) Processor",
    tags=["controlnet", "leres", "depth"],
    category="controlnet",
    version="1.2.1",
)
class LeresImageProcessorInvocation(ImageProcessorInvocation):
    """Applies leres processing to image"""

    thr_a: float = InputField(default=0, description="Leres parameter `thr_a`")
    thr_b: float = InputField(default=0, description="Leres parameter `thr_b`")
    boost: bool = InputField(default=False, description="Whether to use boost mode")
    detect_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.detect_res)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)

    def run_processor(self, image):
        leres_processor = LeresDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = leres_processor(
            image,
            thr_a=self.thr_a,
            thr_b=self.thr_b,
            boost=self.boost,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
        )
        return processed_image


@invocation(
    "tile_image_processor",
    title="Tile Resample Processor",
    tags=["controlnet", "tile"],
    category="controlnet",
    version="1.2.1",
)
class TileResamplerProcessorInvocation(ImageProcessorInvocation):
    """Tile resampler processor"""

    # res: int = InputField(default=512, ge=0, le=1024, description="The pixel resolution for each tile")
    down_sampling_rate: float = InputField(default=1.0, ge=1.0, le=8.0, description="Down sampling rate")

    # tile_resample copied from sd-webui-controlnet/scripts/processor.py
    def tile_resample(
        self,
        np_img: np.ndarray,
        res=512,  # never used?
        down_sampling_rate=1.0,
    ):
        np_img = HWC3(np_img)
        if down_sampling_rate < 1.1:
            return np_img
        H, W, C = np_img.shape
        H = int(float(H) / float(down_sampling_rate))
        W = int(float(W) / float(down_sampling_rate))
        np_img = cv2.resize(np_img, (W, H), interpolation=cv2.INTER_AREA)
        return np_img

    def run_processor(self, img):
        np_img = np.array(img, dtype=np.uint8)
        processed_np_image = self.tile_resample(
            np_img,
            # res=self.tile_size,
            down_sampling_rate=self.down_sampling_rate,
        )
        processed_image = Image.fromarray(processed_np_image)
        return processed_image


@invocation(
    "segment_anything_processor",
    title="Segment Anything Processor",
    tags=["controlnet", "segmentanything"],
    category="controlnet",
    version="1.2.1",
)
class SegmentAnythingProcessorInvocation(ImageProcessorInvocation):
    """Applies segment anything processing to image"""

    def run_processor(self, image):
        # segment_anything_processor = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
        segment_anything_processor = SamDetectorReproducibleColors.from_pretrained(
            "ybelkada/segment-anything", subfolder="checkpoints"
        )
        np_img = np.array(image, dtype=np.uint8)
        processed_image = segment_anything_processor(np_img)
        return processed_image


class SamDetectorReproducibleColors(SamDetector):
    # overriding SamDetector.show_anns() method to use reproducible colors for segmentation image
    #     base class show_anns() method randomizes colors,
    #     which seems to also lead to non-reproducible image generation
    # so using ADE20k color palette instead
    def show_anns(self, anns: List[Dict]):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        h, w = anns[0]["segmentation"].shape
        final_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        palette = ade_palette()
        for i, ann in enumerate(sorted_anns):
            m = ann["segmentation"]
            img = np.empty((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            # doing modulo just in case number of annotated regions exceeds number of colors in palette
            ann_color = palette[i % len(palette)]
            img[:, :] = ann_color
            final_img.paste(Image.fromarray(img, mode="RGB"), (0, 0), Image.fromarray(np.uint8(m * 255)))
        return np.array(final_img, dtype=np.uint8)


@invocation(
    "color_map_image_processor",
    title="Color Map Processor",
    tags=["controlnet"],
    category="controlnet",
    version="1.2.1",
)
class ColorMapImageProcessorInvocation(ImageProcessorInvocation):
    """Generates a color map from the provided image"""

    color_map_tile_size: int = InputField(default=64, ge=0, description=FieldDescriptions.tile_size)

    def run_processor(self, image: Image.Image):
        image = image.convert("RGB")
        np_image = np.array(image, dtype=np.uint8)
        height, width = np_image.shape[:2]

        width_tile_size = min(self.color_map_tile_size, width)
        height_tile_size = min(self.color_map_tile_size, height)

        color_map = cv2.resize(
            np_image,
            (width // width_tile_size, height // height_tile_size),
            interpolation=cv2.INTER_CUBIC,
        )
        color_map = cv2.resize(color_map, (width, height), interpolation=cv2.INTER_NEAREST)
        color_map = Image.fromarray(color_map)
        return color_map


DEPTH_ANYTHING_MODEL_SIZES = Literal["large", "base", "small"]


@invocation(
    "depth_anything_image_processor",
    title="Depth Anything Processor",
    tags=["controlnet", "depth", "depth anything"],
    category="controlnet",
    version="1.0.0",
)
class DepthAnythingImageProcessorInvocation(ImageProcessorInvocation):
    """Generates a depth map based on the Depth Anything algorithm"""

    model_size: DEPTH_ANYTHING_MODEL_SIZES = InputField(
        default="small", description="The size of the depth model to use"
    )
    resolution: int = InputField(default=512, ge=64, multiple_of=64, description=FieldDescriptions.image_res)
    offload: bool = InputField(default=False)

    def run_processor(self, image: Image.Image):
        depth_anything_detector = DepthAnythingDetector()
        depth_anything_detector.load_model(model_size=self.model_size)

        if image.mode == "RGBA":
            image = image.convert("RGB")

        processed_image = depth_anything_detector(image=image, resolution=self.resolution, offload=self.offload)
        return processed_image


@invocation(
    "dw_openpose_image_processor",
    title="DW Openpose Image Processor",
    tags=["controlnet", "dwpose", "openpose"],
    category="controlnet",
    version="1.0.0",
)
class DWOpenposeImageProcessorInvocation(ImageProcessorInvocation):
    """Generates an openpose pose from an image using DWPose"""

    draw_body: bool = InputField(default=True)
    draw_face: bool = InputField(default=False)
    draw_hands: bool = InputField(default=False)
    image_resolution: int = InputField(default=512, ge=0, description=FieldDescriptions.image_res)

    def run_processor(self, image):
        dw_openpose = DWOpenposeDetector()
        processed_image = dw_openpose(
            image,
            draw_face=self.draw_face,
            draw_hands=self.draw_hands,
            draw_body=self.draw_body,
            resolution=self.image_resolution,
        )
        return processed_image
