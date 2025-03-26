from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxFillConditioningField,
    InputField,
    OutputField,
    TensorField,
)
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("flux_fill_output")
class FluxFillOutput(BaseInvocationOutput):
    """The conditioning output of a FLUX Fill invocation."""

    fill_cond: FluxFillConditioningField = OutputField(
        description=FieldDescriptions.flux_redux_conditioning, title="Conditioning"
    )


@invocation(
    "flux_fill",
    title="FLUX Fill Conditioning",
    tags=["inpaint"],
    category="inpaint",
    version="1.0.0",
    classification=Classification.Beta,
)
class FluxFillInvocation(BaseInvocation):
    """Prepare the FLUX Fill conditioning data."""

    image: ImageField = InputField(description="The FLUX Fill reference image.")
    mask: TensorField = InputField(
        description="The bool inpainting mask. Excluded regions should be set to "
        "False, included regions should be set to True.",
    )

    def invoke(self, context: InvocationContext) -> FluxFillOutput:
        return FluxFillOutput(fill_cond=FluxFillConditioningField(image=self.image, mask=self.mask))
