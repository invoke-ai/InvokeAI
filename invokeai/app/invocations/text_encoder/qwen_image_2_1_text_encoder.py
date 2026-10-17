import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import Qwen3VLEncoderField
from invokeai.app.invocations.primitives import QwenImage21ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.qwen_image_2_1.text_encoding import encode_prompt
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    QwenImage21ConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "qwen_image_2_1_text_encoder",
    title="Prompt - Qwen-Image-2.1",
    tags=["prompt", "conditioning", "qwen_image_2_1", "qwen-image-2.1"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class QwenImage21TextEncoderInvocation(BaseInvocation):
    """Encodes a text prompt for Qwen-Image-2.1 with its Qwen3-VL-8B encoder."""

    prompt: str = InputField(description="Text prompt describing the desired image.", ui_component=UIComponent.Textarea)
    qwen3_vl_encoder: Qwen3VLEncoderField = InputField(
        title="Qwen3-VL Encoder", description=FieldDescriptions.qwen3_vl_encoder, input=Input.Connection
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> QwenImage21ConditioningOutput:
        tokenizer_info = context.models.load(self.qwen3_vl_encoder.tokenizer)
        text_encoder_info = context.models.load(self.qwen3_vl_encoder.text_encoder)
        # A GGUF or int8 encoder dequantizes one layer at a time; that transient is the working memory.
        dequant_bytes = peak_dequant_transient_bytes(
            text_encoder_info.model, TorchDevice.choose_bfloat16_safe_dtype(text_encoder_info.compute_device)
        )

        context.util.signal_progress("Running Qwen3-VL text encoder")
        with (
            tokenizer_info as tokenizer,
            text_encoder_info.model_on_device(working_mem_bytes=dequant_bytes) as (
                _,
                text_encoder,
            ),
        ):
            embeds = encode_prompt(text_encoder, tokenizer, self.prompt, get_effective_device(text_encoder))

        conditioning = QwenImage21ConditioningInfo(prompt_embeds=embeds.detach().cpu())
        name = context.conditioning.save(ConditioningFieldData(conditionings=[conditioning]))
        return QwenImage21ConditioningOutput.build(name)
