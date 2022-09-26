# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import Field
from .image import BaseImageOutput, ImageField
from .baseinvocation import BaseInvocation
from ..services.invocation_services import InvocationServices


SAMPLER_NAME_VALUES = Literal["ddim","plms","k_lms","k_dpm_2","k_dpm_2_a","k_euler","k_euler_a","k_heun"]

# Text to image
class TextToImageInvocation(BaseInvocation):
    """Generates an image using text2img."""
    type: Literal['txt2img']

    # Inputs
    # TODO: consider making prompt optional to enable providing prompt through a link
    prompt: Optional[str]     = Field(description="The prompt to generate an image from")
    seed: int                 = Field(default=0, description="The seed to use (0 for a random seed)")
    steps: int                = Field(default=10, gt=0, description="The number of steps to use to generate the image")
    width: int                = Field(default=512, gt=0, description="The width of the resulting image")
    height: int               = Field(default=512, gt=0, description="The height of the resulting image")
    cfg_scale: float          = Field(default=7.5, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt")
    sampler_name: SAMPLER_NAME_VALUES = Field(default="k_lms", description="The sampler to use")
    seamless: bool            = Field(default=False, description="Whether or not to generate an image that can tile without seams")
    model: str                = Field(default='', description="The model to use (currently ignored)")
    progress_images: bool     = Field(default=False, description="Whether or not to produce progress images during generation")

    class Outputs(BaseImageOutput):
        ...

    def dispatch_progress(self, services: InvocationServices, sample: Any, step: int) -> None:
        services.events.dispatch('progress', {
            #'context_id': self.get_context_id(), # TODO: figure out how to do this
            'invocation_id': self.id,
#                'sample': sample,
            'step': step,
            'percent': float(step) / float(self.steps)
        })

    def invoke(self, services: InvocationServices) -> Outputs:
        results = services.generate.prompt2image(
            prompt = self.prompt,
            step_callback = lambda sample, step: self.dispatch_progress(services, sample, step),
            **self.dict(exclude = {'prompt'}) # Shorthand for passing all of the parameters above manually
        )

        # TODO: send events on progress

        # Results are image and seed, unwrap for now and ignore the seed
        # TODO: pre-seed?
        # TODO: can this return multiple results? Should it?
        return TextToImageInvocation.Outputs.construct(
            image = ImageField.from_image(results[0][0])
        )





class ImageToImageInvocation(TextToImageInvocation):
    """Generates an image using img2img."""
    type: Literal["img2img"]

    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength of the original image")
    fit: bool                     = Field(default=True, description="Whether or not the result should be fit to the aspect ratio of the input image")

    class Outputs(BaseImageOutput):
        ...

    def invoke(self, services: InvocationServices) -> Outputs:
        results = services.generate.prompt2image(
            prompt   = self.prompt,
            init_img = self.image.get(),
            **self.dict(exclude = {'prompt','image'}) # Shorthand for passing all of the parameters above manually
        )

        # TODO: send events on progress

        # Results are image and seed, unwrap for now and ignore the seed
        # TODO: pre-seed?
        # TODO: can this return multiple results? Should it?
        return ImageToImageInvocation.Outputs.construct(
            image = ImageField.from_image(results[0][0])
        )
