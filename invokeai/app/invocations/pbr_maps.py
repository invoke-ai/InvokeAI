import pathlib
from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import ImageField, InputField, OutputField, WithBoard, WithMetadata
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.pbr_maps.architecture.pbr_rrdb_net import PBR_RRDB_Net
from invokeai.backend.image_util.pbr_maps.pbr_maps import NORMAL_MAP_MODEL, OTHER_MAP_MODEL, PBRMapsGenerator
from invokeai.backend.util.devices import TorchDevice


@invocation_output("pbr_maps-output")
class PBRMapsOutput(BaseInvocationOutput):
    normal_map: ImageField = OutputField(default=None, description="The generated normal map")
    roughness_map: ImageField = OutputField(default=None, description="The generated roughness map")
    displacement_map: ImageField = OutputField(default=None, description="The generated displacement map")


@invocation("pbr_maps", title="PBR Maps", tags=["image", "material"], category="image", version="1.0.0")
class PBRMapsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generate Normal, Displacement and Roughness Map from a given image"""

    image: ImageField = InputField(description="Input image")
    tile_size: int = InputField(default=512, description="Tile size")
    border_mode: Literal["none", "seamless", "mirror", "replicate"] = InputField(
        default="none", description="Border mode to apply to eliminate any artifacts or seams"
    )

    def invoke(self, context: InvocationContext) -> PBRMapsOutput:
        image_pil = context.images.get_pil(self.image.image_name, mode="RGB")

        def loader(model_path: pathlib.Path):
            return PBRMapsGenerator.load_model(model_path, TorchDevice.choose_torch_device())

        with (
            context.models.load_remote_model(NORMAL_MAP_MODEL, loader) as normal_map_model,
            context.models.load_remote_model(OTHER_MAP_MODEL, loader) as other_map_model,
        ):
            assert isinstance(normal_map_model, PBR_RRDB_Net)
            assert isinstance(other_map_model, PBR_RRDB_Net)
            pbr_pipeline = PBRMapsGenerator(normal_map_model, other_map_model, TorchDevice.choose_torch_device())
            normal_map, roughness_map, displacement_map = pbr_pipeline.generate_maps(
                image_pil, self.tile_size, self.border_mode
            )

            normal_map = context.images.save(normal_map)
            normal_map_field = ImageField(image_name=normal_map.image_name)

            roughness_map = context.images.save(roughness_map)
            roughness_map_field = ImageField(image_name=roughness_map.image_name)

            displacement_map = context.images.save(displacement_map)
            displacement_map_field = ImageField(image_name=displacement_map.image_name)

        return PBRMapsOutput(
            normal_map=normal_map_field, roughness_map=roughness_map_field, displacement_map=displacement_map_field
        )
