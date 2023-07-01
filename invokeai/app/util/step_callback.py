from invokeai.app.models.exceptions import CanceledException
from invokeai.app.models.image import ProgressImage
from ..invocations.baseinvocation import InvocationContext
from ...backend.util.util import image_to_dataURL
from ...backend.generator.base import Generator
from ...backend.stable_diffusion import PipelineIntermediateState


def stable_diffusion_step_callback(
    context: InvocationContext,
    intermediate_state: PipelineIntermediateState,
    node: dict,
    source_node_id: str,
):
    if context.services.queue.is_canceled(context.graph_execution_state_id):
        raise CanceledException

    # Some schedulers report not only the noisy latents at the current timestep,
    # but also their estimate so far of what the de-noised latents will be. Use
    # that estimate if it is available.
    if intermediate_state.predicted_original is not None:
        sample = intermediate_state.predicted_original
    else:
        sample = intermediate_state.latents

    # TODO: This does not seem to be needed any more?
    # # txt2img provides a Tensor in the step_callback
    # # img2img provides a PipelineIntermediateState
    # if isinstance(sample, PipelineIntermediateState):
    #     # this was an img2img
    #     print('img2img')
    #     latents = sample.latents
    #     step = sample.step
    # else:
    #     print('txt2img')
    #     latents = sample
    #     step = intermediate_state.step

    # TODO: only output a preview image when requested
    image = Generator.sample_to_lowres_estimated_image(sample)

    (width, height) = image.size
    width *= 8
    height *= 8

    dataURL = image_to_dataURL(image, image_format="JPEG")

    context.services.events.emit_generator_progress(
        graph_execution_state_id=context.graph_execution_state_id,
        node=node,
        source_node_id=source_node_id,
        progress_image=ProgressImage(width=width, height=height, dataURL=dataURL),
        step=intermediate_state.step,
        total_steps=node["steps"],
    )
