from datetime import datetime, timezone
from typing import Literal, Union

from pydantic import Field

from ..services.image_storage import ImageType
from ..services.invocation_services import InvocationServices
from .baseinvocation import BaseInvocation, InvocationContext
from .image import ImageField, ImageOutput


class RestoreFaceInvocation(BaseInvocation):
    """Restores faces in an image."""
    #fmt: off
    type:  Literal["restore_face"] = "restore_face"

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image")
    strength:                float = Field(default=0.75, gt=0, le=1, description="The strength of the restoration"  )
    #fmt: on
    
    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )
        results = context.services.generate.upscale_and_reconstruct(
            image_list=[[image, 0]],
            upscale=None,
            strength=self.strength,  # GFPGAN strength
            save_original=False,
            image_callback=None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        context.services.images.save(image_type, image_name, results[0][0])
        return ImageOutput(
            image=ImageField(image_type=image_type, image_name=image_name)
        )
