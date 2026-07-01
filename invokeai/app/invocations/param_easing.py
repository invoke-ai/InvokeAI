import numpy as np

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.primitives import FloatCollectionOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "float_range",
    title="Float Range",
    tags=["math", "range"],
    category="math",
    version="1.0.1",
)
class FloatLinearRangeInvocation(BaseInvocation):
    """Creates a range"""

    start: float = InputField(default=5, description="The first value of the range")
    stop: float = InputField(default=10, description="The last value of the range")
    steps: int = InputField(
        default=30,
        description="number of values to interpolate over (including start and stop)",
    )

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        param_list = list(np.linspace(self.start, self.stop, self.steps))
        return FloatCollectionOutput(collection=param_list)
