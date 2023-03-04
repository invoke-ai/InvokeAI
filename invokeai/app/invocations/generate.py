# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union

import numpy as np
from PIL import Image
from pydantic import Field
from skimage.exposure.histogram_matching import match_histograms

from ..services.image_storage import ImageType
from ..services.invocation_services import InvocationServices
from .baseinvocation import BaseInvocation, InvocationContext
from .image import ImageField, ImageOutput

SAMPLER_NAME_VALUES = Literal[
    "ddim", "plms", "k_lms", "k_dpm_2", "k_dpm_2_a", "k_euler", "k_euler_a", "k_heun"
]


# Text to image
class TextToImageInvocation(BaseInvocation):
    """Generates an image using text2img."""

    type: Literal["txt2img"] = "txt2img"

    # Inputs
    # TODO: consider making prompt optional to enable providing prompt through a link
    # fmt: off
    prompt: Optional[str] = Field(description="The prompt to generate an image from")
    seed:        int = Field(default=-1,ge=-1, le=np.iinfo(np.uint32).max, description="The seed to use (-1 for a random seed)", )
    steps:       int = Field(default=10, gt=0, description="The number of steps to use to generate the image")
    width:       int = Field(default=512, multiple_of=64, gt=0, description="The width of the resulting image", )
    height:      int = Field(default=512, multiple_of=64, gt=0, description="The height of the resulting image", )
    cfg_scale: float = Field(default=7.5, gt=0, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )
    sampler_name: SAMPLER_NAME_VALUES = Field(default="k_lms", description="The sampler to use" )
    seamless:   bool = Field(default=False, description="Whether or not to generate an image that can tile without seams", )
    model:       str = Field(default="", description="The model to use (currently ignored)")
    progress_images: bool = Field(default=False, description="Whether or not to produce progress images during generation",  )
    # fmt: on

    # TODO: pass this an emitter method or something? or a session for dispatching?
    def dispatch_progress(
        self, context: InvocationContext, sample: Any = None, step: int = 0
    ) -> None:
        context.services.events.emit_generator_progress(
            context.graph_execution_state_id,
            self.id,
            step,
            float(step) / float(self.steps),
        )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        def step_callback(sample, step=0):
            self.dispatch_progress(context, sample, step)

        # Handle invalid model parameter
        # TODO: figure out if this can be done via a validator that uses the model_cache
        # TODO: How to get the default model name now?
        if self.model is None or self.model == "":
            self.model = context.services.generate.model_name

        # Set the model (if already cached, this does nothing)
        context.services.generate.set_model(self.model)

        results = context.services.generate.prompt2image(
            prompt=self.prompt,
            step_callback=step_callback,
            **self.dict(
                exclude={"prompt"}
            ),  # Shorthand for passing all of the parameters above manually
        )

        # Results are image and seed, unwrap for now and ignore the seed
        # TODO: pre-seed?
        # TODO: can this return multiple results? Should it?
        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        context.services.images.save(image_type, image_name, results[0][0])
        return ImageOutput(
            image=ImageField(image_type=image_type, image_name=image_name)
        )


class ImageToImageInvocation(TextToImageInvocation):
    """Generates an image using img2img."""

    type: Literal["img2img"] = "img2img"

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image")
    strength: float = Field(
        default=0.75, gt=0, le=1, description="The strength of the original image"
    )
    fit: bool = Field(
        default=True,
        description="Whether or not the result should be fit to the aspect ratio of the input image",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = (
            None
            if self.image is None
            else context.services.images.get(
                self.image.image_type, self.image.image_name
            )
        )
        mask = None

        def step_callback(sample, step=0):
            self.dispatch_progress(context, sample, step)

        # Handle invalid model parameter
        # TODO: figure out if this can be done via a validator that uses the model_cache
        # TODO: How to get the default model name now?
        if self.model is None or self.model == "":
            self.model = context.services.generate.model_name

        # Set the model (if already cached, this does nothing)
        context.services.generate.set_model(self.model)

        results = context.services.generate.prompt2image(
            prompt=self.prompt,
            init_img=image,
            init_mask=mask,
            step_callback=step_callback,
            **self.dict(
                exclude={"prompt", "image", "mask"}
            ),  # Shorthand for passing all of the parameters above manually
        )

        result_image = results[0][0]

        # Results are image and seed, unwrap for now and ignore the seed
        # TODO: pre-seed?
        # TODO: can this return multiple results? Should it?
        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        context.services.images.save(image_type, image_name, result_image)
        return ImageOutput(
            image=ImageField(image_type=image_type, image_name=image_name)
        )


class InpaintInvocation(ImageToImageInvocation):
    """Generates an image using inpaint."""

    type: Literal["inpaint"] = "inpaint"

    # Inputs
    mask: Union[ImageField, None] = Field(description="The mask")
    inpaint_replace: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The amount by which to replace masked areas with latent noise",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = (
            None
            if self.image is None
            else context.services.images.get(
                self.image.image_type, self.image.image_name
            )
        )
        mask = (
            None
            if self.mask is None
            else context.services.images.get(self.mask.image_type, self.mask.image_name)
        )

        def step_callback(sample, step=0):
            self.dispatch_progress(context, sample, step)

        # Handle invalid model parameter
        # TODO: figure out if this can be done via a validator that uses the model_cache
        # TODO: How to get the default model name now?
        if self.model is None or self.model == "":
            self.model = context.services.generate.model_name

        # Set the model (if already cached, this does nothing)
        context.services.generate.set_model(self.model)

        results = context.services.generate.prompt2image(
            prompt=self.prompt,
            init_img=image,
            init_mask=mask,
            step_callback=step_callback,
            **self.dict(
                exclude={"prompt", "image", "mask"}
            ),  # Shorthand for passing all of the parameters above manually
        )

        result_image = results[0][0]

        # Results are image and seed, unwrap for now and ignore the seed
        # TODO: pre-seed?
        # TODO: can this return multiple results? Should it?
        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        context.services.images.save(image_type, image_name, result_image)
        return ImageOutput(
            image=ImageField(image_type=image_type, image_name=image_name)
        )
