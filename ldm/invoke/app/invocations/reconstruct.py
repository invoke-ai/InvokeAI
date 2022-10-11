from datetime import datetime, timezone
from typing import Literal, Union
from pydantic import Field
from .image import ImageField, ImageOutput
from .baseinvocation import BaseInvocation
from ..services.image_storage import ImageType
from ..services.invocation_services import InvocationServices


class RestoreFaceInvocation(BaseInvocation):
    """Restores faces in an image."""
    type: Literal['restore_face'] = 'restore_face'

    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength of the restoration")


    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput: 
        image = services.images.get(self.image.image_type, self.image.image_name)
        results = services.generate.upscale_and_reconstruct(
            image_list     = [[image, 0]],
            upscale        = None,
            strength       = self.strength, # GFPGAN strength
            save_original  = False,
            image_callback = None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        image_type = ImageType.RESULT
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, results[0][0])
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )
