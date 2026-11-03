from contextlib import ExitStack

import torch
from transformers import PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import Qwen35EncoderField
from invokeai.app.invocations.primitives import MossImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, Qwen35VariantType
from invokeai.backend.moss_image.text_encoding import encode_moss_prompt
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    MossImageConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "moss_image_text_encoder",
    title="Prompt - MOSS-Image",
    tags=["prompt", "conditioning", "moss_image", "moss-image"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class MossImageTextEncoderInvocation(BaseInvocation):
    """Encodes a prompt for MOSS-Image with its Qwen3.5-9B text encoder.

    The prompt is the user turn of a chat, read after the encoder's final norm, as the reference pipeline encodes it.
    Natural language, tag sequences and MOSS layout JSON all go in as they are.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_5_encoder: Qwen35EncoderField = InputField(
        title="Qwen3.5 Encoder",
        description=FieldDescriptions.qwen3_5_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MossImageConditioningOutput:
        # Detached and on the CPU: this node may run on a borrowed idle GPU, released as soon as it returns.
        prompt_embeds = self._encode(context).detach().to("cpu")
        conditioning_data = ConditioningFieldData(
            conditionings=[MossImageConditioningInfo(prompt_embeds=prompt_embeds)]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return MossImageConditioningOutput.build(conditioning_name)

    def _encode(self, context: InvocationContext) -> torch.Tensor:
        # Imported here: transformers' qwen3_5 modeling module costs ~0.25 s, which every app start would otherwise
        # pay through this node module.
        from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder

        self._raise_unless_qwen3_5_9b(context)
        tokenizer_info = context.models.load(self.qwen3_5_encoder.tokenizer)
        encoder_info = context.models.load(self.qwen3_5_encoder.text_encoder)
        # A quantized build dequantizes each packed Linear per forward, a transient its resident size does not cover.
        # Read from the unlocked model, before the lock the reservation applies to; zero for a dense build.
        dequant_bytes = peak_dequant_transient_bytes(
            encoder_info.model, TorchDevice.choose_bfloat16_safe_dtype(encoder_info.compute_device)
        )

        context.util.signal_progress("Running Qwen3.5 text encoder")
        with ExitStack() as exit_stack:
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())
            (_, encoder) = exit_stack.enter_context(encoder_info.model_on_device(working_mem_bytes=dequant_bytes))
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"Expected a tokenizer, got {type(tokenizer).__name__}.")
            if not isinstance(encoder, Qwen35Encoder):
                raise TypeError(f"Expected a Qwen3.5 encoder, got {type(encoder).__name__}.")
            # The compute device, not where the weights sit: partial loading may have left them all in RAM.
            return encode_moss_prompt(self.prompt, tokenizer, encoder, device=encoder_info.compute_device)

    def _raise_unless_qwen3_5_9b(self, context: InvocationContext) -> None:
        """Refuse any encoder but the 9B, from its record, before its weights are loaded.

        The model loader checks this, but a workflow can wire this node to another loader's encoder output -- Anima's
        Qwen3.5 4B, say. That one loads (~8 GB) and then fails without its final norm, or later on a 2560-against-4096
        shape, neither of which names the encoder the user picked.
        """
        config = context.models.get_config(self.qwen3_5_encoder.text_encoder)
        from_moss_pipeline = config.type is ModelType.Main and config.base is BaseModelType.MossImage
        is_9b = (
            config.type is ModelType.Qwen35Encoder and getattr(config, "variant", None) == Qwen35VariantType.Qwen35_9B
        )
        if not (from_moss_pipeline or is_9b):
            raise ValueError(
                f"'{config.name}' is not a Qwen3.5 9B encoder, which MOSS-Image conditions on (hidden size 4096). "
                "Select a Qwen3.5 9B encoder, or the MOSS-Image diffusers model's own, on the MOSS-Image model loader."
            )
