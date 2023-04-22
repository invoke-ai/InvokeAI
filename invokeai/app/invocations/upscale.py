# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal, Union

from pydantic import Field

from invokeai.app.models.image import ImageField, ImageType
from .baseinvocation import BaseInvocation, InvocationContext, InvocationConfig
from .image import ImageOutput, build_image_output


class UpscaleInvocation(BaseInvocation):
    """Upscales an image."""
    #fmt: off
    type: Literal["upscale"] = "upscale"

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    strength: float = Field(default=0.75, gt=0, le=1, description="The strength")
    level: Literal[2, 4] = Field(default=2, description="The upscale level")
    #fmt: on


    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["upscaling", "image"],
            },
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )
        results = context.services.restoration.upscale_and_reconstruct(
            image_list=[[image, 0]],
            upscale=(self.level, self.strength),
            strength=0.0,  # GFPGAN strength
            save_original=False,
            image_callback=None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, results[0][0], metadata)
        return build_image_output(
            image_type=image_type,
            image_name=image_name,
            image=results[0][0]
        )