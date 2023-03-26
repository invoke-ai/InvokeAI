import torch
from PIL import Image
from ..invocations.baseinvocation import InvocationContext
from ...backend.util.util import image_to_dataURL
from ...backend.generator.base import Generator
from ...backend.stable_diffusion import PipelineIntermediateState

class CanceledException(Exception):
    pass

def fast_latents_step_callback(sample: torch.Tensor, step: int, steps: int, id: str, context: InvocationContext, ):
    # TODO: only output a preview image when requested
    image = Generator.sample_to_lowres_estimated_image(sample)

    (width, height) = image.size
    width *= 8
    height *= 8

    dataURL = image_to_dataURL(image, image_format="JPEG")

    context.services.events.emit_generator_progress(
        context.graph_execution_state_id,
        id,
        {
            "width": width,
            "height": height,
            "dataURL": dataURL
        },
        step,
        steps,
    )

def diffusers_step_callback_adapter(*cb_args, **kwargs):
    """
    txt2img gives us a Tensor in the step_callbak, while img2img gives us a PipelineIntermediateState.
    This adapter grabs the needed data and passes it along to the callback function.
    """
    if isinstance(cb_args[0], PipelineIntermediateState):
        progress_state: PipelineIntermediateState = cb_args[0]
        return fast_latents_step_callback(progress_state.latents, progress_state.step, **kwargs)
    else:
        return fast_latents_step_callback(*cb_args, **kwargs)
