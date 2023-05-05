# InvokeAI nodes for ControlNet image preprocessors
# initial implementation by Gregg Helt, 2023
# heavily leverages controlnet_aux package: https://github.com/patrickvonplaten/controlnet_aux
import numpy as np

from typing import Literal, Optional, Union, List
from pydantic import BaseModel, Field

from ..models.image import ImageField, ImageType
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)

from controlnet_aux import (
    CannyDetector,
    HEDdetector,
    LineartDetector,
    LineartAnimeDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
    ContentShuffleDetector,
    ZoeDetector)

from .image import ImageOutput, build_image_output, PILInvocationConfig


class ControlField(BaseModel):
    image: ImageField = Field(default=None, description="processed image")
    # width: Optional[int] = Field(default=None, description="The width of the image in pixels")
    # height: Optional[int] = Field(default=None, description="The height of the image in pixels")
    # mode: Optional[str] = Field(default=None, description="The mode of the image")
    control_model: Optional[str] = Field(default=None, description="The control model used")
    control_weight: Optional[float] = Field(default=None, description="The control weight used")

    class Config:
        schema_extra = {
            "required": ["image", "control_model", "control_weight"]
            # "required": ["type", "image", "width", "height", "mode"]
        }


class ControlOutput(BaseInvocationOutput):
    """node output for ControlNet info"""
    # fmt: off
    type: Literal["control_output"] = "control_output"
    control: Optional[ControlField] = Field(default=None, description="The control info dict")
    raw_processed_image: ImageField = Field(default=None,
                                            description="outputs just the image info (also included in control output)")
    # fmt: on


# This super class handles invoke() call, which in turn calls run_processor(image)
#     subclasses override run_processor() instead of implementing their own invoke()
class PreprocessedControlNetInvocation(BaseInvocation, PILInvocationConfig):
    """Base class for invocations that preprocess images for ControlNet"""

    # fmt: off
    type: Literal["preprocessed_control"] = "preprocessed_control"
    # Inputs
    image: ImageField = Field(default=None, description="image to process")
    control_model: str = Field(default=None, description="control model to use")
    control_weight: float = Field(default=0.5, ge=0, le=1, description="control weight")
    # TODO: support additional ControlNet parameters (mostly just passthroughs to other nodes with ControlField inputs)
    # begin_step_percent: float = Field(default=0, ge=0, le=1,
    #                                    description="% of total steps at which controlnet is first applied")
    # end_step_percent: float = Field(default=1, ge=0, le=1,
    #                                  description="% of total steps at which controlnet is last applied")
    # guess_mode: bool = Field(default=False, description="use guess mode (controlnet ignores prompt)")
    # fmt: on


    def run_processor(self, image):
        # superclass just passes through image without processing
        return image

    def invoke(self, context: InvocationContext) -> ControlOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )
        # image type should be PIL.PngImagePlugin.PngImageFile ?
        processed_image = self.run_processor(image)
        # currently can't see processed image in node UI without a showImage node,
        #    so for now setting image_type to RESULT instead of INTERMEDIATE so will get saved in gallery
        # image_type = ImageType.INTERMEDIATE
        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        context.services.images.save(image_type, image_name, processed_image, metadata)

        """Builds an ImageOutput and its ImageField"""
        image_field = ImageField(
            image_name=image_name,
            image_type=image_type,
        )
        return ControlOutput(
            control=ControlField(
                image=image_field,
                control_model=self.control_model,
                control_weight=self.control_weight,
            ),
            raw_processed_image=image_field,
        )


class CannyControlInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Canny edge detection for ControlNet"""
    # fmt: off
    type: Literal["cannycontrol"] = "cannycontrol"
    # Input
    low_threshold: float = Field(default=100, ge=0, description="low threshold of Canny pixel gradient")
    high_threshold: float = Field(default=200, ge=0, description="high threshold of Canny pixel gradient")
    # fmt: on

    def run_processor(self, image):
        canny_processor = CannyDetector()
        processed_image = canny_processor(image, self.low_threshold, self.high_threshold)
        return processed_image


class HedControlNetInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies HED edge detection to image"""
    # fmt: off
    type: Literal["hed_control"] = "hed_control"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="pixel resolution for edge detection")
    image_resolution: int = Field(default=512, ge=0, description="pixel resolution for output image")
    safe: bool = Field(default=False, description="whether to use safe mode")
    scribble: bool = Field(default=False, description="whether to use scribble mode")
    # fmt: on

    def run_processor(self, image):
        hed_processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
        processed_image = hed_processor(image,
                                        detect_resolution=self.detect_resolution,
                                        image_resolution=self.image_resolution,
                                        safe=self.safe,
                                        scribble=self.scribble,
                                        )
        return processed_image


class LineartControlInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies line art processing to image"""
    # fmt: off
    type: Literal["lineart_control"] = "lineart_control"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="pixel resolution for edge detection")
    image_resolution: int = Field(default=512, ge=0, description="pixel resolution for output image")
    coarse: bool = Field(default=False, description="whether to use coarse mode")
    # fmt: on

    def run_processor(self, image):
        lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = lineart_processor(image,
                                            detect_resolution=self.detect_resolution,
                                            image_resolution=self.image_resolution,
                                            coarse=self.coarse)
        return processed_image


class LineartAnimeControlInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies line art anime processing to image"""
    # fmt: off
    type: Literal["lineart_anime_control"] = "lineart_anime_control"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="pixel resolution for edge detection")
    image_resolution: int = Field(default=512, ge=0, description="pixel resolution for output image")
    # fmt: on

    def run_processor(self, image):
        processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = processor(image,
                                    detect_resolution=self.detect_resolution,
                                    image_resolution=self.image_resolution,
                                    )
        return processed_image


class OpenposeControlInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies Openpose processing to image"""
    # fmt: off
    type: Literal["openpose_control"] = "openpose_control"
    # Inputs
    hand_and_face: bool = Field(default=False, description="whether to use hands and face mode")
    detect_resolution: int = Field(default=512, ge=0, description="pixel resolution for edge detection")
    image_resolution: int = Field(default=512, ge=0, description="pixel resolution for output image")
    # fmt: on

    def run_processor(self, image):
        openpose_processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = openpose_processor(image,
                                             detect_resolution=self.detect_resolution,
                                             image_resolution=self.image_resolution,
                                             hand_and_face=self.hand_and_face,
                                             )
        return processed_image


class MidasDepthControlInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies Midas depth processing to image"""
    # fmt: off
    type: Literal["midas_control"] = "midas_control"
    # Inputs
    a_mult: float = Field(default=2.0, ge=0, description="Midas parameter a = amult * PI")
    bg_th: float = Field(default=0.1, ge=0, description="Midas parameter bg_th")
    depth_and_normal: bool = Field(default=False, description="whether to use depth and normal mode")
    # fmt: on

    def run_processor(self, image):
        midas_processor = MidasDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = midas_processor(image,
                                          a=np.pi * self.a_mult,
                                          bg_th=self.bg_th,
                                          depth_and_normal=self.depth_and_normal)
        return processed_image


class NormalbaeControlNetInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies NormalBae processing to image"""
    # fmt: off
    type: Literal["normalbae_control"] = "normalbae_control"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="pixel resolution for edge detection")
    image_resolution: int = Field(default=512, ge=0, description="pixel resolution for output image")
    # fmt: on

    def run_processor(self, image):
        normalbae_processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = normalbae_processor(image,
                                              detect_resolution=self.detect_resolution,
                                              image_resolution=self.image_resolution)
        return processed_image


class MLSDControlNetInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies MLSD processing to image"""
    # fmt: off
    type: Literal["mlsd_control"] = "mlsd_control"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="pixel resolution for edge detection")
    image_resolution: int = Field(default=512, ge=0, description="pixel resolution for output image")
    thr_v: float = Field(default=0.1, ge=0, description="MLSD parameter thr_v")
    thr_d: float = Field(default=0.1, ge=0, description="MLSD parameter thr_d")
    # fmt: on

    def run_processor(self, image):
        mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")
        processed_image = mlsd_processor(image,
                                         detect_resolution=self.detect_resolution,
                                         image_resolution=self.image_resolution,
                                         thr_v=self.thr_v,
                                         thr_d=self.thr_d)
        return processed_image


class PidiControlNetInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies PIDI processing to image"""
    # fmt: off
    type: Literal["pidi_control"] = "pidi_control"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="pixel resolution for edge detection")
    image_resolution: int = Field(default=512, ge=0, description="pixel resolution for output image")
    safe: bool = Field(default=False, description="whether to use safe mode")
    scribble: bool = Field(default=False, description="whether to use scribble mode")
    # fmt: on

    def run_processor(self, image):
        pidi_processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = pidi_processor(image,
                                         detect_resolution=self.detect_resolution,
                                         image_resolution=self.image_resolution,
                                         safe=self.safe,
                                         scribble=self.scribble)
        return processed_image


class ContentShuffleControlInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies content shuffle processing to image"""
    # fmt: off
    type: Literal["content_shuffle_control"] = "content_shuffle_control"
    # Inputs
    detect_resolution: int = Field(default=512, ge=0, description="pixel resolution for edge detection")
    image_resolution: int = Field(default=512, ge=0, description="pixel resolution for output image")
    h: Union[int | None] = Field(default=None, ge=0, description="content shuffle h parameter")
    w: Union[int | None] = Field(default=None, ge=0, description="content shuffle w parameter")
    f: Union[int | None] = Field(default=None, ge=0, description="cont")
    # fmt: on

    def run_processor(self, image):
        content_shuffle_processor = ContentShuffleDetector()
        processed_image = content_shuffle_processor(image,
                                                    detect_resolution=self.detect_resolution,
                                                    image_resolution=self.image_resolution,
                                                    h=self.h,
                                                    w=self.w,
                                                    f=self.f
                                                    )
        return processed_image


class ZoeDepthControlInvocation(PreprocessedControlNetInvocation, PILInvocationConfig):
    """Applies Zoe depth processing to image"""
    # fmt: off
    type: Literal["zoe_depth_control"] = "zoe_depth_control"
    # fmt: on

    def run_processor(self, image):
        zoe_depth_processor = ZoeDetector.from_pretrained("lllyasviel/Annotators")
        processed_image = zoe_depth_processor(image)
        return processed_image
