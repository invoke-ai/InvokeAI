import io
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from easing_functions import (
    BackEaseIn,
    BackEaseInOut,
    BackEaseOut,
    BounceEaseIn,
    BounceEaseInOut,
    BounceEaseOut,
    CircularEaseIn,
    CircularEaseInOut,
    CircularEaseOut,
    CubicEaseIn,
    CubicEaseInOut,
    CubicEaseOut,
    ElasticEaseIn,
    ElasticEaseInOut,
    ElasticEaseOut,
    ExponentialEaseIn,
    ExponentialEaseInOut,
    ExponentialEaseOut,
    LinearInOut,
    QuadEaseIn,
    QuadEaseInOut,
    QuadEaseOut,
    QuarticEaseIn,
    QuarticEaseInOut,
    QuarticEaseOut,
    QuinticEaseIn,
    QuinticEaseInOut,
    QuinticEaseOut,
    SineEaseIn,
    SineEaseInOut,
    SineEaseOut,
)
from matplotlib.ticker import MaxNLocator

from invokeai.app.invocations.primitives import FloatCollectionOutput
from invokeai.app.services.shared.invocation_context import InvocationContext

from .baseinvocation import BaseInvocation, invocation
from .fields import InputField


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


EASING_FUNCTIONS_MAP = {
    "Linear": LinearInOut,
    "QuadIn": QuadEaseIn,
    "QuadOut": QuadEaseOut,
    "QuadInOut": QuadEaseInOut,
    "CubicIn": CubicEaseIn,
    "CubicOut": CubicEaseOut,
    "CubicInOut": CubicEaseInOut,
    "QuarticIn": QuarticEaseIn,
    "QuarticOut": QuarticEaseOut,
    "QuarticInOut": QuarticEaseInOut,
    "QuinticIn": QuinticEaseIn,
    "QuinticOut": QuinticEaseOut,
    "QuinticInOut": QuinticEaseInOut,
    "SineIn": SineEaseIn,
    "SineOut": SineEaseOut,
    "SineInOut": SineEaseInOut,
    "CircularIn": CircularEaseIn,
    "CircularOut": CircularEaseOut,
    "CircularInOut": CircularEaseInOut,
    "ExponentialIn": ExponentialEaseIn,
    "ExponentialOut": ExponentialEaseOut,
    "ExponentialInOut": ExponentialEaseInOut,
    "ElasticIn": ElasticEaseIn,
    "ElasticOut": ElasticEaseOut,
    "ElasticInOut": ElasticEaseInOut,
    "BackIn": BackEaseIn,
    "BackOut": BackEaseOut,
    "BackInOut": BackEaseInOut,
    "BounceIn": BounceEaseIn,
    "BounceOut": BounceEaseOut,
    "BounceInOut": BounceEaseInOut,
}

EASING_FUNCTION_KEYS = Literal[tuple(EASING_FUNCTIONS_MAP.keys())]


# actually I think for now could just use CollectionOutput (which is list[Any]
@invocation(
    "step_param_easing",
    title="Step Param Easing",
    tags=["step", "easing"],
    category="step",
    version="1.0.2",
)
class StepParamEasingInvocation(BaseInvocation):
    """Experimental per-step parameter easing for denoising steps"""

    easing: EASING_FUNCTION_KEYS = InputField(default="Linear", description="The easing function to use")
    num_steps: int = InputField(default=20, description="number of denoising steps")
    start_value: float = InputField(default=0.0, description="easing starting value")
    end_value: float = InputField(default=1.0, description="easing ending value")
    start_step_percent: float = InputField(default=0.0, description="fraction of steps at which to start easing")
    end_step_percent: float = InputField(default=1.0, description="fraction of steps after which to end easing")
    # if None, then start_value is used prior to easing start
    pre_start_value: Optional[float] = InputField(default=None, description="value before easing start")
    # if None, then end value is used prior to easing end
    post_end_value: Optional[float] = InputField(default=None, description="value after easing end")
    mirror: bool = InputField(default=False, description="include mirror of easing function")
    # FIXME: add alt_mirror option (alternative to default or mirror), or remove entirely
    # alt_mirror: bool = InputField(default=False, description="alternative mirroring by dual easing")
    show_easing_plot: bool = InputField(default=False, description="show easing plot")

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        log_diagnostics = False
        # convert from start_step_percent to nearest step <= (steps * start_step_percent)
        # start_step = int(np.floor(self.num_steps * self.start_step_percent))
        start_step = int(np.round(self.num_steps * self.start_step_percent))
        # convert from end_step_percent to nearest step >= (steps * end_step_percent)
        # end_step = int(np.ceil((self.num_steps - 1) * self.end_step_percent))
        end_step = int(np.round((self.num_steps - 1) * self.end_step_percent))

        # end_step = int(np.ceil(self.num_steps * self.end_step_percent))
        num_easing_steps = end_step - start_step + 1

        # num_presteps = max(start_step - 1, 0)
        num_presteps = start_step
        num_poststeps = self.num_steps - (num_presteps + num_easing_steps)
        prelist = list(num_presteps * [self.pre_start_value])
        postlist = list(num_poststeps * [self.post_end_value])

        if log_diagnostics:
            context.logger.debug("start_step: " + str(start_step))
            context.logger.debug("end_step: " + str(end_step))
            context.logger.debug("num_easing_steps: " + str(num_easing_steps))
            context.logger.debug("num_presteps: " + str(num_presteps))
            context.logger.debug("num_poststeps: " + str(num_poststeps))
            context.logger.debug("prelist size: " + str(len(prelist)))
            context.logger.debug("postlist size: " + str(len(postlist)))
            context.logger.debug("prelist: " + str(prelist))
            context.logger.debug("postlist: " + str(postlist))

        easing_class = EASING_FUNCTIONS_MAP[self.easing]
        if log_diagnostics:
            context.logger.debug("easing class: " + str(easing_class))
        easing_list = []
        if self.mirror:  # "expected" mirroring
            # if number of steps is even, squeeze duration down to (number_of_steps)/2
            # and create reverse copy of list to append
            # if number of steps is odd, squeeze duration down to ceil(number_of_steps/2)
            # and create reverse copy of list[1:end-1]
            # but if even then number_of_steps/2 === ceil(number_of_steps/2), so can just use ceil always

            base_easing_duration = int(np.ceil(num_easing_steps / 2.0))
            if log_diagnostics:
                context.logger.debug("base easing duration: " + str(base_easing_duration))
            even_num_steps = num_easing_steps % 2 == 0  # even number of steps
            easing_function = easing_class(
                start=self.start_value,
                end=self.end_value,
                duration=base_easing_duration - 1,
            )
            base_easing_vals = []
            for step_index in range(base_easing_duration):
                easing_val = easing_function.ease(step_index)
                base_easing_vals.append(easing_val)
                if log_diagnostics:
                    context.logger.debug("step_index: " + str(step_index) + ", easing_val: " + str(easing_val))
            if even_num_steps:
                mirror_easing_vals = list(reversed(base_easing_vals))
            else:
                mirror_easing_vals = list(reversed(base_easing_vals[0:-1]))
            if log_diagnostics:
                context.logger.debug("base easing vals: " + str(base_easing_vals))
                context.logger.debug("mirror easing vals: " + str(mirror_easing_vals))
            easing_list = base_easing_vals + mirror_easing_vals

        # FIXME: add alt_mirror option (alternative to default or mirror), or remove entirely
        # elif self.alt_mirror:  # function mirroring (unintuitive behavior (at least to me))
        #     # half_ease_duration = round(num_easing_steps - 1 / 2)
        #     half_ease_duration = round((num_easing_steps - 1) / 2)
        #     easing_function = easing_class(start=self.start_value,
        #                                    end=self.end_value,
        #                                    duration=half_ease_duration,
        #                                    )
        #
        #     mirror_function = easing_class(start=self.end_value,
        #                                    end=self.start_value,
        #                                    duration=half_ease_duration,
        #                                    )
        #     for step_index in range(num_easing_steps):
        #         if step_index <= half_ease_duration:
        #             step_val = easing_function.ease(step_index)
        #         else:
        #             step_val = mirror_function.ease(step_index - half_ease_duration)
        #         easing_list.append(step_val)
        #         if log_diagnostics: logger.debug(step_index, step_val)
        #

        else:  # no mirroring (default)
            easing_function = easing_class(
                start=self.start_value,
                end=self.end_value,
                duration=num_easing_steps - 1,
            )
            for step_index in range(num_easing_steps):
                step_val = easing_function.ease(step_index)
                easing_list.append(step_val)
                if log_diagnostics:
                    context.logger.debug("step_index: " + str(step_index) + ", easing_val: " + str(step_val))

        if log_diagnostics:
            context.logger.debug("prelist size: " + str(len(prelist)))
            context.logger.debug("easing_list size: " + str(len(easing_list)))
            context.logger.debug("postlist size: " + str(len(postlist)))

        param_list = prelist + easing_list + postlist

        if self.show_easing_plot:
            plt.figure()
            plt.xlabel("Step")
            plt.ylabel("Param Value")
            plt.title("Per-Step Values Based On Easing: " + self.easing)
            plt.bar(range(len(param_list)), param_list)
            # plt.plot(param_list)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            im = PIL.Image.open(buf)
            im.show()
            buf.close()

        # output array of size steps, each entry list[i] is param value for step i
        return FloatCollectionOutput(collection=param_list)
