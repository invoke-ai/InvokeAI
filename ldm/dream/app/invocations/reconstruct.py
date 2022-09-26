from typing import Literal, Union
from pydantic import Field
from .image import BaseImageOutput, ImageField
from .baseinvocation import BaseInvocation
from ..services.invocation_services import InvocationServices


class RestoreFaceInvocation(BaseInvocation):
    """Restores faces in an image."""
    type: Literal["restore_face"]

    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength of the restoration")

    class Outputs(BaseImageOutput):
        ...

    def invoke(self, services: InvocationServices) -> Outputs: 
        results = services.generate.upscale_and_reconstruct(
            image_list     = [[self.image.get(), 0]],
            upscale        = None,
            strength       = self.strength, # GFPGAN strength
            save_original  = False,
            image_callback = None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        return RestoreFaceInvocation.Outputs.construct(
            image = ImageField.from_image(results[0][0])
        )
