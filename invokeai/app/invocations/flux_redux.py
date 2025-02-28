import torch
from PIL import Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxReduxConditioningField,
    InputField,
    OutputField,
)
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.services.model_records.model_records_base import UnknownModelException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.redux.flux_redux_model import FluxReduxModel
from invokeai.backend.model_manager.config import (
    BaseModelType,
    ModelType,
)
from invokeai.backend.sig_lip.sig_lip_pipeline import SigLipPipeline
from invokeai.backend.util.devices import TorchDevice


@invocation_output("flux_redux_output")
class FluxReduxOutput(BaseInvocationOutput):
    """The conditioning output of a FLUX Redux invocation."""

    redux_cond: FluxReduxConditioningField = OutputField(
        description=FieldDescriptions.flux_redux_conditioning, title="Conditioning"
    )


SIGLIP_STARTER_MODEL_NAME = "SigLIP - google/siglip-so400m-patch14-384"
FLUX_REDUX_STARTER_MODEL_NAME = "FLUX Redux"


@invocation(
    "flux_redux",
    title="FLUX Redux",
    tags=["ip_adapter", "control"],
    category="ip_adapter",
    version="1.0.0",
    classification=Classification.Prototype,
)
class FluxReduxInvocation(BaseInvocation):
    """Runs a FLUX Redux model to generate a conditioning tensor."""

    image: ImageField = InputField(description="The FLUX Redux image prompt.")

    # TODO(ryand): Add support for a mask.

    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")

        encoded_x = self._siglip_encode(context, image)
        redux_conditioning = self._flux_redux_encode(context, encoded_x)

        tensor_name = context.tensors.save(redux_conditioning)
        return FluxReduxOutput(redux_cond=FluxReduxConditioningField(tensor_name=tensor_name))

    @torch.no_grad()
    def _siglip_encode(self, context: InvocationContext, image: Image.Image) -> torch.Tensor:
        try:
            siglip_model = context.models.load_by_attrs(
                name=SIGLIP_STARTER_MODEL_NAME,
                base=BaseModelType.Any,
                type=ModelType.SigLIP,
            )
        except UnknownModelException as e:
            raise RuntimeError(
                f"The SigLIP model required for FLUX Redux is not installed. Install '{SIGLIP_STARTER_MODEL_NAME}' "
                "from the Starter Models tab."
            ) from e

        with siglip_model.model_on_device() as (_, siglip_pipeline):
            assert isinstance(siglip_pipeline, SigLipPipeline)
            return siglip_pipeline.encode_image(
                x=image, device=TorchDevice.choose_torch_device(), dtype=TorchDevice.choose_torch_dtype()
            )

    @torch.no_grad()
    def _flux_redux_encode(self, context: InvocationContext, encoded_x: torch.Tensor) -> torch.Tensor:
        try:
            redux_model = context.models.load_by_attrs(
                name=FLUX_REDUX_STARTER_MODEL_NAME,
                base=BaseModelType.Flux,
                type=ModelType.FluxRedux,
            )
        except UnknownModelException as e:
            raise RuntimeError(
                f"The FLUX Redux model is not installed. Install the '{FLUX_REDUX_STARTER_MODEL_NAME}' model from the "
                " Starter Models tab."
            ) from e

        with redux_model.model_on_device() as (_, flux_redux):
            assert isinstance(flux_redux, FluxReduxModel)
            dtype = next(flux_redux.parameters()).dtype
            encoded_x = encoded_x.to(dtype=dtype)
            return flux_redux(encoded_x)
