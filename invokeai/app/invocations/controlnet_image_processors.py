# Invocations for ControlNet image preprocessors
# initial implementation by Gregg Helt, 2023
# heavily leverages controlnet_aux package: https://github.com/patrickvonplaten/controlnet_aux
from builtins import bool, float
from typing import Dict, List, Literal, Optional, Union

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
    OpenposeDetector,
    PidiNetDetector,
    SamDetector,
    ZoeDetector,
)
from controlnet_aux.util import HWC3, ade_palette
from PIL import Image
from pydantic import BaseModel, Field, validator

from ...backend.model_management import BaseModelType, ModelType
from ..models.image import ImageCategory, ImageField, ResourceOrigin
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationConfig, InvocationContext
from ..models.image import ImageOutput, PILInvocationConfig

CONTROLNET_DEFAULT_MODELS = [
    ###########################################
    # lllyasviel sd v1.5, ControlNet v1.0 models
    ##############################################
    "lllyasviel/sd-controlnet-canny",
    "lllyasviel/sd-controlnet-depth",
    "lllyasviel/sd-controlnet-hed",
    "lllyasviel/sd-controlnet-seg",
    "lllyasviel/sd-controlnet-openpose",
    "lllyasviel/sd-controlnet-scribble",
    "lllyasviel/sd-controlnet-normal",
    "lllyasviel/sd-controlnet-mlsd",
    #############################################
    # lllyasviel sd v1.5, ControlNet v1.1 models
    #############################################
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_seg",
    # "lllyasviel/control_v11p_sd15_depth",  # broken
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_normalbae",
    "lllyasviel/control_v11p_sd15_scribble",
    "lllyasviel/control_v11p_sd15_mlsd",
    "lllyasviel/control_v11p_sd15_softedge",
    "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "lllyasviel/control_v11p_sd15_lineart",
    "lllyasviel/control_v11p_sd15_inpaint",
    # "lllyasviel/control_v11u_sd15_tile",
    # problem (temporary?) with huffingface "lllyasviel/control_v11u_sd15_tile",
    # so for now replace  "lllyasviel/control_v11f1e_sd15_tile",
    "lllyasviel/control_v11e_sd15_shuffle",
    "lllyasviel/control_v11e_sd15_ip2p",
    "lllyasviel/control_v11f1e_sd15_tile",
    #################################################
    #  thibaud sd v2.1 models (ControlNet v1.0? or v1.1?
    ##################################################
    "thibaud/controlnet-sd21-openpose-diffusers",
    "thibaud/controlnet-sd21-canny-diffusers",
    "thibaud/controlnet-sd21-depth-diffusers",
    "thibaud/controlnet-sd21-scribble-diffusers",
    "thibaud/controlnet-sd21-hed-diffusers",
    "thibaud/controlnet-sd21-zoedepth-diffusers",
    "thibaud/controlnet-sd21-color-diffusers",
    "thibaud/controlnet-sd21-openposev2-diffusers",
    "thibaud/controlnet-sd21-lineart-diffusers",
    "thibaud/controlnet-sd21-normalbae-diffusers",
    "thibaud/controlnet-sd21-ade20k-diffusers",
    ##############################################
    #  ControlNetMediaPipeface, ControlNet v1.1
    ##############################################
    # ["CrucibleAI/ControlNetMediaPipeFace", "diffusion_sd15"],  # SD 1.5
    #    diffusion_sd15 needs to be passed to from_pretrained() as subfolder arg
    #    hacked t2l to split to model & subfolder if format is "model,subfolder"
    "CrucibleAI/ControlNetMediaPipeFace,diffusion_sd15",  # SD 1.5
    "CrucibleAI/ControlNetMediaPipeFace",  # SD 2.1?
]

CONTROLNET_NAME_VALUES = Literal[tuple(CONTROLNET_DEFAULT_MODELS)]
CONTROLNET_MODE_VALUES = Literal[tuple(["balanced", "more_prompt", "more_control", "unbalanced"])]
CONTROLNET_RESIZE_VALUES = Literal[
    tuple(
        [
            "just_resize",
            "crop_resize",
            "fill_resize",
            "just_resize_simple",
        ]
    )
]


class ControlNetModelField(BaseModel):
    """ControlNet model field"""

    model_name: str = Field(description="Name of the ControlNet model")
    base_model: BaseModelType = Field(description="Base model")


class ControlField(BaseModel):
    image: ImageField = Field(default=None, description="The control image")
    control_model: Optional[ControlNetModelField] = Field(default=None, description="The ControlNet model to use")
    # control_weight: Optional[float] = Field(default=1, description="weight given to controlnet")
    control_weight: Union[float, List[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode to use")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")

    @validator("control_weight")
    def validate_control_weight(cls, v):
        """Validate that all control weights in the valid range"""
        if isinstance(v, list):
            for i in v:
                if i < -1 or i > 2:
                    raise ValueError("Control weights must be within -1 to 2 range")
        else:
            if v < -1 or v > 2:
                raise ValueError("Control weights must be within -1 to 2 range")
        return v

    class Config:
        schema_extra = {
            "required": ["image", "control_model", "control_weight", "begin_step_percent", "end_step_percent"],
            "ui": {
                "type_hints": {
                    "control_weight": "float",
                    "control_model": "controlnet_model",
                    # "control_weight": "number",
                }
            },
        }


class ControlOutput(BaseInvocationOutput):
    """node output for ControlNet info"""

    # fmt: off
    type: Literal["control_output"] = "control_output"
    control: ControlField = Field(default=None, description="The control info")
    # fmt: on


class ControlNetInvocation(BaseInvocation):
    """Collects ControlNet info to pass to other nodes"""

    # fmt: off
    type: Literal["controlnet"] = "controlnet"
    # Inputs
    image: ImageField = Field(default=None, description="The control image")
    control_model: ControlNetModelField = Field(default="lllyasviel/sd-controlnet-canny",
                                                  description="control model used")
    control_weight: Union[float, List[float]] = Field(default=1.0, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(default=0, ge=-1, le=2,
                                      description="When the ControlNet is first applied (% of total steps)")
    end_step_percent: float = Field(default=1, ge=0, le=1,
                                    description="When the ControlNet is last applied (% of total steps)")
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode used")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode used")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "ControlNet",
                "tags": ["controlnet", "latents"],
                "type_hints": {
                    "model": "model",
                    "control": "control",
                    # "cfg_scale": "float",
                    "cfg_scale": "number",
                    "control_weight": "float",
                },
            },
        }

    def invoke(self, context: InvocationContext) -> ControlOutput:
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


class ImageProcessorInvocation(BaseInvocation, PILInvocationConfig):
    """Base class for invocations that preprocess images for ControlNet"""

    # fmt: off
    type: Literal["image_processor"] = "image_processor"
    # Inputs
    image: ImageField = Field(default=None, description="The image to process")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Image Processor", "tags": ["image", "processor"]},
        }

    def run_processor(self, image):
        # superclass just passes through image without processing
        return image

    def invoke(self, context: InvocationContext) -> ImageOutput:
        raw_image = context.services.images.get_pil_image(self.image.image_name)
        # image type should be PIL.PngImagePlugin.PngImageFile ?
        processed_image = self.run_processor(raw_image)

        # FIXME: what happened to image metadata?
        # metadata = context.services.metadata.build_metadata(
        #     session_id=context.graph_execution_state_id, node=self
        # )

        # currently can't see processed image in node UI without a showImage node,
        #    so for now setting image_type to RESULT instead of INTERMEDIATE so will get saved in gallery
        image_dto = context.services.images.create(
            image=processed_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.CONTROL,
            session_id=context.graph_execution_state_id,
            node_id=self.id,
            is_intermediate=self.is_intermediate,
        )

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


class CannyImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Canny edge detection for ControlNet"""

    # fmt: off
    type: Literal["canny_image_processor"] = "canny_image_processor"
    # Input
    low_threshold: int = Field(default=100, ge=0, le=255, description="The low threshold of the Canny pixel gradient (0-255)")
    high_threshold: int = Field(default=200, ge=0, le=255, description="The high threshold of the Canny pixel gradient (0-255)")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Canny Processor", "tags": ["controlnet", "canny", "image", "processor"]},
        }

    def run_processor(self, image):
        canny_processor = CannyDetector()
        processed_image = canny_processor(image, self.low_threshold, self.high_threshold)
        return processed_image


class HedImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies HED edge detection to image"""

    # fmt: off
    type: Literal["hed_image_processor"] = "hed_image_processor"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    # safe not supported in controlnet_aux v0.0.3
    # safe: bool = Field(default=False, description="whether to use safe mode")
    scribble: bool = Field(default=False, description="Whether to use scribble mode")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Softedge(HED) Processor", "tags": ["controlnet", "softedge", "hed", "image", "processor"]},
        }

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


class LineartImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies line art processing to image"""

    # fmt: off
    type: Literal["lineart_image_processor"] = "lineart_image_processor"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    coarse: bool = Field(default=False, description="Whether to use coarse mode")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Lineart Processor", "tags": ["controlnet", "lineart", "image", "processor"]},
        }

    def run_processor(self, image):
        lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = lineart_processor(
            image, detect_resolution=self.detect_resolution, image_resolution=self.image_resolution, coarse=self.coarse
        )
        return processed_image


class LineartAnimeImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies line art anime processing to image"""

    # fmt: off
    type: Literal["lineart_anime_image_processor"] = "lineart_anime_image_processor"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Lineart Anime Processor",
                "tags": ["controlnet", "lineart", "anime", "image", "processor"],
            },
        }

    def run_processor(self, image):
        processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = processor(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
        )
        return processed_image


class OpenposeImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies Openpose processing to image"""

    # fmt: off
    type: Literal["openpose_image_processor"] = "openpose_image_processor"
    # Inputs
    hand_and_face: bool = Field(default=False, description="Whether to use hands and face mode")
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Openpose Processor", "tags": ["controlnet", "openpose", "image", "processor"]},
        }

    def run_processor(self, image):
        openpose_processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = openpose_processor(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
            hand_and_face=self.hand_and_face,
        )
        return processed_image


class MidasDepthImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies Midas depth processing to image"""

    # fmt: off
    type: Literal["midas_depth_image_processor"] = "midas_depth_image_processor"
    # Inputs
    a_mult: float = Field(default=2.0, ge=0, description="Midas parameter `a_mult` (a = a_mult * PI)")
    bg_th: float = Field(default=0.1, ge=0, description="Midas parameter `bg_th`")
    # depth_and_normal not supported in controlnet_aux v0.0.3
    # depth_and_normal: bool = Field(default=False, description="whether to use depth and normal mode")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Midas (Depth) Processor", "tags": ["controlnet", "midas", "depth", "image", "processor"]},
        }

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


class NormalbaeImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies NormalBae processing to image"""

    # fmt: off
    type: Literal["normalbae_image_processor"] = "normalbae_image_processor"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Normal BAE Processor", "tags": ["controlnet", "normal", "bae", "image", "processor"]},
        }

    def run_processor(self, image):
        normalbae_processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = normalbae_processor(
            image, detect_resolution=self.detect_resolution, image_resolution=self.image_resolution
        )
        return processed_image


class MlsdImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies MLSD processing to image"""

    # fmt: off
    type: Literal["mlsd_image_processor"] = "mlsd_image_processor"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    thr_v: float = Field(default=0.1, ge=0, description="MLSD parameter `thr_v`")
    thr_d: float = Field(default=0.1, ge=0, description="MLSD parameter `thr_d`")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "MLSD Processor", "tags": ["controlnet", "mlsd", "image", "processor"]},
        }

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


class PidiImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies PIDI processing to image"""

    # fmt: off
    type: Literal["pidi_image_processor"] = "pidi_image_processor"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    safe: bool = Field(default=False, description="Whether to use safe mode")
    scribble: bool = Field(default=False, description="Whether to use scribble mode")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "PIDI Processor", "tags": ["controlnet", "pidi", "image", "processor"]},
        }

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


class ContentShuffleImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies content shuffle processing to image"""

    # fmt: off
    type: Literal["content_shuffle_image_processor"] = "content_shuffle_image_processor"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    h: Optional[int] = Field(default=512, ge=0, description="Content shuffle `h` parameter")
    w: Optional[int] = Field(default=512, ge=0, description="Content shuffle `w` parameter")
    f: Optional[int] = Field(default=256, ge=0, description="Content shuffle `f` parameter")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Content Shuffle Processor",
                "tags": ["controlnet", "contentshuffle", "image", "processor"],
            },
        }

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
class ZoeDepthImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies Zoe depth processing to image"""

    # fmt: off
    type: Literal["zoe_depth_image_processor"] = "zoe_depth_image_processor"
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Zoe (Depth) Processor", "tags": ["controlnet", "zoe", "depth", "image", "processor"]},
        }

    def run_processor(self, image):
        zoe_depth_processor = ZoeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = zoe_depth_processor(image)
        return processed_image


class MediapipeFaceProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies mediapipe face processing to image"""

    # fmt: off
    type: Literal["mediapipe_face_processor"] = "mediapipe_face_processor"
    # Inputs
    max_faces: int = Field(default=1, ge=1, description="Maximum number of faces to detect")
    min_confidence: float = Field(default=0.5, ge=0, le=1, description="Minimum confidence for face detection")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Mediapipe Processor", "tags": ["controlnet", "mediapipe", "image", "processor"]},
        }

    def run_processor(self, image):
        # MediaPipeFaceDetector throws an error if image has alpha channel
        #     so convert to RGB if needed
        if image.mode == "RGBA":
            image = image.convert("RGB")
        mediapipe_face_processor = MediapipeFaceDetector()
        processed_image = mediapipe_face_processor(image, max_faces=self.max_faces, min_confidence=self.min_confidence)
        return processed_image


class LeresImageProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies leres processing to image"""

    # fmt: off
    type: Literal["leres_image_processor"] = "leres_image_processor"
    # Inputs
    thr_a: float = Field(default=0, description="Leres parameter `thr_a`")
    thr_b: float = Field(default=0, description="Leres parameter `thr_b`")
    boost: bool = Field(default=False, description="Whether to use boost mode")
    detect_resolution: int = Field(default=512, ge=0, description="The pixel resolution for detection")
    image_resolution: int = Field(default=512, ge=0, description="The pixel resolution for the output image")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Leres (Depth) Processor", "tags": ["controlnet", "leres", "depth", "image", "processor"]},
        }

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


class TileResamplerProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    # fmt: off
    type: Literal["tile_image_processor"] = "tile_image_processor"
    # Inputs
    #res: int = Field(default=512, ge=0, le=1024, description="The pixel resolution for each tile")
    down_sampling_rate: float = Field(default=1.0, ge=1.0, le=8.0, description="Down sampling rate")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Tile Resample Processor",
                "tags": ["controlnet", "tile", "resample", "image", "processor"],
            },
        }

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


class SegmentAnythingProcessorInvocation(ImageProcessorInvocation, PILInvocationConfig):
    """Applies segment anything processing to image"""

    # fmt: off
    type: Literal["segment_anything_processor"] = "segment_anything_processor"
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Segment Anything Processor",
                "tags": ["controlnet", "segment", "anything", "sam", "image", "processor"],
            },
        }

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
