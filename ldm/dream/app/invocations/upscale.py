# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from enum import IntEnum
from typing import Literal, Union
from pydantic import Field
from ldm.dream.app.invocations.image import BaseImageOutput, ImageField
from ldm.dream.app.invocations.baseinvocation import BaseInvocation, InvocationContext


class UpscaleLevel(IntEnum):
    two = 2
    four = 4

class UpscaleInvocation(BaseInvocation):
    """Generates an image using text2img."""
    type: Literal["upscale"]

    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength")
    level: UpscaleLevel           = Field(default=2, description = "The upscale level")

    class Outputs(BaseImageOutput):
        ...

    def invoke(self, context: InvocationContext) -> Outputs: 
        results = context.generate.upscale_and_reconstruct(
            image_list     = [[self.image.image, 0]],
            upscale        = (self.level, self.strength),
            strength       = 0.0, # GFPGAN strength
            save_original  = False,
            image_callback = None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        return UpscaleInvocation.Outputs.construct(
            image = ImageField(image=results[0][0])
        )
