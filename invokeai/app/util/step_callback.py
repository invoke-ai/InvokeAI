import torch
from PIL import Image
from invokeai.app.models.exceptions import CanceledException
from invokeai.app.models.image import ProgressImage
from ..invocations.baseinvocation import InvocationContext
from ...backend.util.util import image_to_dataURL
from ...backend.stable_diffusion import PipelineIntermediateState


def sample_to_lowres_estimated_image(samples):
    # origingally adapted from code by @erucipe and @keturn here:
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7

    # these updated numbers for v1.5 are from @torridgristle
    v1_5_latent_rgb_factors = torch.tensor(
        [
            #    R        G        B
            [0.3444, 0.1385, 0.0670],  # L1
            [0.1247, 0.4027, 0.1494],  # L2
            [-0.3192, 0.2513, 0.2103],  # L3
            [-0.1307, -0.1874, -0.7445],  # L4
        ],
        dtype=samples.dtype,
        device=samples.device,
    )

    latent_image = samples[0].permute(1, 2, 0) @ v1_5_latent_rgb_factors
    latents_ubyte = (
        ((latent_image + 1) / 2)
        .clamp(0, 1)  # change scale from -1..1 to 0..1
        .mul(0xFF)  # to 0..255
        .byte()
    ).cpu()

    return Image.fromarray(latents_ubyte.numpy())

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
    image = sample_to_lowres_estimated_image(sample)

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
