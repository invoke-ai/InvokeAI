from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.lineart_anime import LineartAnimeEdgeDetector, UnetGenerator


@invocation(
    "lineart_anime_edge_detection",
    title="Lineart Anime Edge Detection",
    tags=["controlnet", "lineart"],
    category="controlnet",
    version="1.0.0",
)
class LineartAnimeEdgeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Geneartes an edge map using the Lineart model."""

    image: ImageField = InputField(description="The image to process")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        model_url = LineartAnimeEdgeDetector.get_model_url()
        loaded_model = context.models.load_remote_model(model_url, LineartAnimeEdgeDetector.load_model)

        with loaded_model as model:
            assert isinstance(model, UnetGenerator)
            detector = LineartAnimeEdgeDetector(model)
            edge_map = detector.run(image=image)

        image_dto = context.images.save(image=edge_map)
        return ImageOutput.build(image_dto)
