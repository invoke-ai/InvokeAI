from contextlib import ExitStack
from typing import Iterator, Literal, Tuple

import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField
from invokeai.app.invocations.model import CLIPField, T5EncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.modules.conditioner import HFEncoder
from invokeai.backend.lora.conversions.flux_kohya_lora_conversion_utils import FLUX_KOHYA_T5_PREFIX
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw
from invokeai.backend.lora.lora_patcher import LoRAPatcher
from invokeai.backend.model_manager.config import ModelFormat
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo


@invocation(
    "flux_text_encoder",
    title="FLUX Text Encoding",
    tags=["prompt", "conditioning", "flux"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class FluxTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for a flux image."""

    clip: CLIPField = InputField(
        title="CLIP",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )
    t5_encoder: T5EncoderField = InputField(
        title="T5Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )
    t5_max_seq_len: Literal[256, 512] = InputField(
        description="Max sequence length for the T5 encoder. Expected to be 256 for FLUX schnell models and 512 for FLUX dev models."
    )
    prompt: str = InputField(description="Text prompt to encode.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        # Note: The T5 and CLIP encoding are done in separate functions to ensure that all model references are locally
        # scoped. This ensures that the T5 model can be freed and gc'd before loading the CLIP model (if necessary).
        t5_embeddings = self._t5_encode(context)
        clip_embeddings = self._clip_encode(context)
        conditioning_data = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=clip_embeddings, t5_embeds=t5_embeddings)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput.build(conditioning_name)

    def _t5_encode(self, context: InvocationContext) -> torch.Tensor:
        t5_tokenizer_info = context.models.load(self.t5_encoder.tokenizer)
        t5_text_encoder_info = context.models.load(self.t5_encoder.text_encoder)

        prompt = [self.prompt]

        with (
            t5_text_encoder_info.model_on_device() as (cached_weights, t5_text_encoder),
            t5_tokenizer_info as t5_tokenizer,
            ExitStack() as exit_stack,
        ):
            assert isinstance(t5_text_encoder, T5EncoderModel)
            assert isinstance(t5_tokenizer, T5Tokenizer)

            t5_text_encoder_config = t5_text_encoder_info.config
            assert t5_text_encoder_config is not None

            # Apply LoRA models to the T5 encoder.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            if t5_text_encoder_config.format in [ModelFormat.T5Encoder]:
                # The model is non-quantized, so we can apply the LoRA weights directly into the model.
                exit_stack.enter_context(
                    LoRAPatcher.apply_lora_patches(
                        model=t5_text_encoder,
                        patches=self._lora_iterator(context),
                        prefix=FLUX_KOHYA_T5_PREFIX,
                        cached_weights=cached_weights,
                    )
                )
            elif t5_text_encoder_config.format in [ModelFormat.BnbQuantizedLlmInt8b, ModelFormat.BnbQuantizednf4b]:
                # The model is quantized, so apply the LoRA weights as sidecar layers. This results in slower inference,
                # than directly patching the weights, but is agnostic to the quantization format.
                exit_stack.enter_context(
                    LoRAPatcher.apply_lora_sidecar_patches(
                        model=t5_text_encoder,
                        patches=self._lora_iterator(context),
                        prefix=FLUX_KOHYA_T5_PREFIX,
                        dtype=t5_text_encoder.dtype,
                    )
                )
            elif t5_text_encoder_config.format in [ModelFormat.BnbQuantizedLlmInt8b]:
                pass
            else:
                raise ValueError(f"Unsupported model format: {t5_text_encoder_config.format}")

            t5_encoder = HFEncoder(t5_text_encoder, t5_tokenizer, False, self.t5_max_seq_len)

            prompt_embeds = t5_encoder(prompt)

        assert isinstance(prompt_embeds, torch.Tensor)
        return prompt_embeds

    def _clip_encode(self, context: InvocationContext) -> torch.Tensor:
        clip_tokenizer_info = context.models.load(self.clip.tokenizer)
        clip_text_encoder_info = context.models.load(self.clip.text_encoder)

        prompt = [self.prompt]

        with (
            clip_text_encoder_info as clip_text_encoder,
            clip_tokenizer_info as clip_tokenizer,
        ):
            assert isinstance(clip_text_encoder, CLIPTextModel)
            assert isinstance(clip_tokenizer, CLIPTokenizer)

            clip_encoder = HFEncoder(clip_text_encoder, clip_tokenizer, True, 77)

            pooled_prompt_embeds = clip_encoder(prompt)

        assert isinstance(pooled_prompt_embeds, torch.Tensor)
        return pooled_prompt_embeds

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[LoRAModelRaw, float]]:
        for lora in self.t5_encoder.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, LoRAModelRaw)
            yield (lora_info.model, lora.weight)
            del lora_info
