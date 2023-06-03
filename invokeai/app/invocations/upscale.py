# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal, Union

from pydantic import Field

from invokeai.app.models.image import ImageCategory, ImageField, ResourceOrigin
from .baseinvocation import BaseInvocation, InvocationContext, InvocationConfig
from .image import ImageOutput


class UpscaleInvocation(BaseInvocation):
    """Upscales an image."""

    # fmt: off
    type: Literal["upscale"] = "upscale"

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    strength: float = Field(default=0.75, gt=0, le=1, description="The strength")
    level: Literal[2, 4] = Field(default=2, description="The upscale level")
    # fmt: on

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["upscaling", "image"],
            },
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(
            self.image.image_origin, self.image.image_name
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
        image_dto = context.services.images.create(
            image=results[0][0],
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(
                image_name=image_dto.image_name,
                image_origin=image_dto.image_origin,
            ),
            width=image_dto.width,
            height=image_dto.height,
        )
