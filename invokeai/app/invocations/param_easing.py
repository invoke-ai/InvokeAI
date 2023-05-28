from typing import Literal, Optional
from pydantic import BaseModel, Field
import numpy as np

from easing_functions import (
    LinearInOut,
    QuadEaseInOut, QuadEaseIn, QuadEaseOut,
    CubicEaseInOut, CubicEaseIn, CubicEaseOut,
    QuarticEaseInOut, QuarticEaseIn, QuarticEaseOut,
    QuinticEaseInOut, QuinticEaseIn, QuinticEaseOut,
    SineEaseInOut, SineEaseIn, SineEaseOut,
    CircularEaseIn, CircularEaseInOut, CircularEaseOut,
    ExponentialEaseInOut, ExponentialEaseIn, ExponentialEaseOut,
    ElasticEaseIn, ElasticEaseInOut, ElasticEaseOut,
    BackEaseIn, BackEaseInOut, BackEaseOut,
    BounceEaseIn, BounceEaseInOut, BounceEaseOut )

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)
from .collections import FloatCollectionOutput

class FloatLinearRangeInvocation(BaseInvocation):
    """Creates a range"""

    type: Literal["float_range"] = "float_range"

    # Inputs
    start: float = Field(default=5, description="The first value of the range")
    stop: float = Field(default=10, description="The last value of the range")
    steps: int = Field(default=30, description="number of values to interpolate over (including start and stop)")

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        param_list = list(np.linspace(self.start, self.stop, self.steps))
        return FloatCollectionOutput(
            collection=param_list
        )

EASING_FUNCTIONS_MAP = {
    "linear": LinearInOut,
    "quad_in": QuadEaseIn,
    "quad_out": QuadEaseOut,
    "quad_in_out": QuadEaseInOut,
    "cubic_in": CubicEaseIn,
    "cubic_out": CubicEaseOut,
    "cubic_in_out": CubicEaseInOut,
    "quartic_in": QuarticEaseIn,
    "quartic_out": QuarticEaseOut,
    "quartic_in_out": QuarticEaseInOut,
    "quintic_in": QuinticEaseIn,
    "quintic_out": QuinticEaseOut,
    "quintic_in_out": QuinticEaseInOut,
    "sine_in": SineEaseIn,
    "sine_out": SineEaseOut,
    "sine_in_out": SineEaseInOut,
    "circular_in": CircularEaseIn,
    "circular_out": CircularEaseOut,
    "circular_in_out": CircularEaseInOut,
    "exponential_in": ExponentialEaseIn,
    "exponential_out": ExponentialEaseOut,
    "exponential_in_out": ExponentialEaseInOut,
    "elastic_in": ElasticEaseIn,
    "elastic_out": ElasticEaseOut,
    "elastic_in_out": ElasticEaseInOut,
    "back_in": BackEaseIn,
    "back_out": BackEaseOut,
    "back_in_out": BackEaseInOut,
    "bounce_in": BounceEaseIn,
    "bounce_out": BounceEaseOut,
    "bounce_in_out": BounceEaseInOut,
}

EASING_FUNCTION_KEYS = Literal[
    tuple(list(EASING_FUNCTIONS_MAP.keys()))
]

# actually I think for now could just use CollectionOutput (which is list[Any]
class StepParamEasingInvocation(BaseInvocation):
    """Experimental per-step parameter easing for denoising steps"""

    type: Literal["step_param_easing"] = "step_param_easing"

    # Inputs
    # fmt: off
    easing: EASING_FUNCTION_KEYS = Field(default="linear", description="The easing function to use" )
    num_steps: int = Field(description="number of denoising steps")
    start_value: float = Field(default=0.0, description="easing starting value")
    end_value: float = Field(default=1.0, description="easing ending value")
    start_step_percent: float = Field(default=0.0, description="fraction of steps at which to start easing")
    end_step_percent: float = Field(default=1.0, description="fraction of steps after which to end easing")
    # if None, then start_value is used prior to easing start
    pre_start_value: Optional[float] = Field(default=None, description="value before easing start")
    # if None, then end value is used prior to easing end
    post_end_value: Optional[float] = Field(default=None, description="value after easing end")
    # fmt: on

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        # convert from start_step_percent to nearest step <= (steps * start_step_percent)
        start_step = int(np.floor(self.num_steps * self.start_step_percent))
        # convert from end_step_percent to nearest step >= (steps * end_step_percent)
        end_step =   int(np.ceil((self.num_steps-1) * self.end_step_percent))
        num_easing_steps = end_step - start_step + 1
        # print("start_step", start_step)
        # print("end_step", end_step)
        # print("num_easing_steps", num_easing_steps)
        num_presteps = start_step - 1
        num_poststeps = self.num_steps - (num_presteps + num_easing_steps)
        prelist = list(num_presteps * [self.pre_start_value])
        postlist = list(num_poststeps * [self.post_end_value])

        easing_class = EASING_FUNCTIONS_MAP[self.easing]
        # print(easing_class)
        easing_list = list()
        easing_function = easing_class(start=self.start_value,
                                       end=self.end_value,
                                       duration=num_easing_steps-1)
        for step_index in range(num_easing_steps):
            step_val = easing_function.ease(step_index)
            print(step_index, step_val)
            easing_list.append(step_val)

        # linspace_list = list(np.linspace(self.start_value, self.end_value, num_easing_steps))
        # print("linspace easing param_list size", len(easing_list))
        # print(linspace_list)

        # print("prelist size", len(prelist))
        # print("easing_list size", len(easing_list))
        # print("postlist size", len(postlist))

        param_list = prelist + easing_list + postlist
        # print("easing param_list size", len(param_list))
        # print(param_list)

        # output array of size steps, each entry list[i] is param value for step i
        return FloatCollectionOutput(
            collection=param_list
        )


