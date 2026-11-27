from contextlib import ExitStack

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import Qwen3VLEncoderField
from invokeai.app.invocations.primitives import Flux3ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux3.text_encoding import encode_flux3_prompt
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, Flux3ConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux3_text_encoder",
    title="Prompt - FLUX.3",
    tags=["prompt", "conditioning", "flux3", "flux"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class Flux3TextEncoderInvocation(BaseInvocation):
    """Encodes a prompt for FLUX.3 with the Qwen3-VL 4B text encoder.

    Eight decoder layers are stacked per token. Write the prompt as a full description of the image; when
    editing, describe the result and refer to reference images as <ref_image_0>, <ref_image_1>, ... in the
    order they are connected.
    """

    prompt: str = InputField(description="Text prompt describing the desired image.", ui_component=UIComponent.Textarea)
    qwen3_vl_encoder: Qwen3VLEncoderField = InputField(
        title="Qwen3-VL Encoder",
        description=FieldDescriptions.qwen3_vl_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Flux3ConditioningOutput:
        tokenizer_info = context.models.load(self.qwen3_vl_encoder.tokenizer)
        text_encoder_info = context.models.load(self.qwen3_vl_encoder.text_encoder)
        # A GGUF or nvfp4 build dequantizes each packed Linear per forward, a transient its resident size does
        # not cover. Read from the unlocked model, before the lock the reservation applies to; zero otherwise.
        dequant_bytes = peak_dequant_transient_bytes(
            text_encoder_info.model, TorchDevice.choose_bfloat16_safe_dtype(text_encoder_info.compute_device)
        )

        context.util.signal_progress("Running Qwen3-VL text encoder")

        with ExitStack() as exit_stack:
            tokenizer = exit_stack.enter_context(tokenizer_info)
            (_, text_encoder) = exit_stack.enter_context(
                text_encoder_info.model_on_device(working_mem_bytes=dequant_bytes)
            )
            prompt_embeds = encode_flux3_prompt(
                self.prompt, tokenizer, text_encoder, get_effective_device(text_encoder)
            )

        conditioning_data = ConditioningFieldData(
            conditionings=[Flux3ConditioningInfo(prompt_embeds=prompt_embeds.detach().to("cpu"))]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return Flux3ConditioningOutput.build(conditioning_name)
