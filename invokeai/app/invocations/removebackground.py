from typing import Literal
from invokeai.app.models.image import (ImageCategory, ResourceOrigin)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationContext,
    InputField,
    invocation,
    )

rembg_models = Literal[
    "isnet-anime",
    #"isnet-general-use", # on the github page but not shipped with pip it seems
    "silueta",
    "u2net_cloth_seg",
    "u2net_human_seg",
    "u2net",
    "u2netp",
]

@invocation("remove_background", title="Remove Background", tags=["image", "remove", "background", "rembg"], category="image", version="1.0.0")
class RemoveBackgroundInvocation(BaseInvocation):
    """Outputs an image with the background removed behind the subject using rembg."""

    image:       ImageField  = InputField(description="Image to remove background from")
    model_name:  rembg_models = InputField(default="u2net", description="Model to use to remove background")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        try:
            from rembg import remove, new_session
            session = new_session(self.model_name)
            image = remove(image, session=session)
        except:
            context.services.logger.warning("Remove Background --> To use this node, please quit InvokeAI and execute 'pip install rembg' from outside your InvokeAI folder with your InvokeAI virtual environment activated.")
            context.services.logger.warning("Remove Background --> rembg package not found. Passing through unaltered image!")

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
