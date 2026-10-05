from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxKontextConditioningField,
    InputField,
    OutputField,
)
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("flux_kontext_output")
class FluxKontextOutput(BaseInvocationOutput):
    """The conditioning output of a FLUX Kontext invocation."""

    kontext_cond: FluxKontextConditioningField = OutputField(
        description=FieldDescriptions.flux_kontext_conditioning, title="Kontext Conditioning"
    )


@invocation(
    "flux_kontext",
    title="Kontext Conditioning - FLUX",
    tags=["conditioning", "kontext", "flux"],
    category="conditioning",
    version="1.0.0",
)
class FluxKontextInvocation(BaseInvocation):
    """Prepares a reference image for FLUX Kontext conditioning."""

    image: ImageField = InputField(description="The Kontext reference image.")

    def invoke(self, context: InvocationContext) -> FluxKontextOutput:
        """Packages the provided image into a Kontext conditioning field."""
        return FluxKontextOutput(kontext_cond=FluxKontextConditioningField(image=self.image))
