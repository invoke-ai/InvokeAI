from contextlib import ExitStack
from typing import Iterator, Tuple

import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField
from invokeai.app.invocations.model import CLIPField, T5EncoderField
from invokeai.app.invocations.primitives import SD3ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_CLIP_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, SD3ConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# The SD3 T5 Max Sequence Length set based on the default in diffusers.
SD3_T5_MAX_SEQ_LEN = 256


@invocation(
    "sd3_text_encoder",
    title="Prompt - SD3",
    tags=["prompt", "conditioning", "sd3"],
    category="conditioning",
    version="1.0.1",
)
class Sd3TextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for a SD3 image."""

    clip_l: CLIPField = InputField(
        title="CLIP L",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )
    clip_g: CLIPField = InputField(
        title="CLIP G",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )

    # The SD3 models were trained with text encoder dropout, so the T5 encoder can be omitted to save time/memory.
    t5_encoder: T5EncoderField | None = InputField(
        title="T5Encoder",
        default=None,
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )
    prompt: str = InputField(description="Text prompt to encode.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> SD3ConditioningOutput:
        # Note: The text encoding model are run in separate functions to ensure that all model references are locally
        # scoped. This ensures that earlier models can be freed and gc'd before loading later models (if necessary).

        clip_l_embeddings, clip_l_pooled_embeddings = self._clip_encode(context, self.clip_l)
        clip_g_embeddings, clip_g_pooled_embeddings = self._clip_encode(context, self.clip_g)

        t5_embeddings: torch.Tensor | None = None
        if self.t5_encoder is not None:
            t5_embeddings = self._t5_encode(context, SD3_T5_MAX_SEQ_LEN)

        conditioning_data = ConditioningFieldData(
            conditionings=[
                SD3ConditioningInfo(
                    clip_l_embeds=clip_l_embeddings,
                    clip_l_pooled_embeds=clip_l_pooled_embeddings,
                    clip_g_embeds=clip_g_embeddings,
                    clip_g_pooled_embeds=clip_g_pooled_embeddings,
                    t5_embeds=t5_embeddings,
                )
            ]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return SD3ConditioningOutput.build(conditioning_name)

    def _t5_encode(self, context: InvocationContext, max_seq_len: int) -> torch.Tensor:
        assert self.t5_encoder is not None
        prompt = [self.prompt]

        with (
            context.models.load(self.t5_encoder.text_encoder) as t5_text_encoder,
            context.models.load(self.t5_encoder.tokenizer) as t5_tokenizer,
        ):
            context.util.signal_progress("Running T5 encoder")
            assert isinstance(t5_text_encoder, T5EncoderModel)
            assert isinstance(t5_tokenizer, (T5Tokenizer, T5TokenizerFast))

            text_inputs = t5_tokenizer(
                prompt,
                padding="max_length",
                max_length=max_seq_len,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = t5_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            assert isinstance(text_input_ids, torch.Tensor)
            assert isinstance(untruncated_ids, torch.Tensor)
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = t5_tokenizer.batch_decode(untruncated_ids[:, max_seq_len - 1 : -1])
                context.logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_seq_len} tokens: {removed_text}"
                )

            prompt_embeds = t5_text_encoder(text_input_ids.to(TorchDevice.choose_torch_device()))[0]

        assert isinstance(prompt_embeds, torch.Tensor)
        return prompt_embeds

    def _clip_encode(
        self, context: InvocationContext, clip_model: CLIPField, tokenizer_max_length: int = 77
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = [self.prompt]

        clip_text_encoder_info = context.models.load(clip_model.text_encoder)
        with (
            clip_text_encoder_info.model_on_device() as (cached_weights, clip_text_encoder),
            context.models.load(clip_model.tokenizer) as clip_tokenizer,
            ExitStack() as exit_stack,
        ):
            context.util.signal_progress("Running CLIP encoder")
            assert isinstance(clip_text_encoder, (CLIPTextModel, CLIPTextModelWithProjection))
            assert isinstance(clip_tokenizer, CLIPTokenizer)

            clip_text_encoder_config = clip_text_encoder_info.config
            assert clip_text_encoder_config is not None

            # Apply LoRA models to the CLIP encoder.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            if clip_text_encoder_config.format in [ModelFormat.Diffusers]:
                # The model is non-quantized, so we can apply the LoRA weights directly into the model.
                exit_stack.enter_context(
                    LayerPatcher.apply_smart_model_patches(
                        model=clip_text_encoder,
                        patches=self._clip_lora_iterator(context, clip_model),
                        prefix=FLUX_LORA_CLIP_PREFIX,
                        dtype=clip_text_encoder.dtype,
                        cached_weights=cached_weights,
                    )
                )
            else:
                # There are currently no supported CLIP quantized models. Add support here if needed.
                raise ValueError(f"Unsupported model format: {clip_text_encoder_config.format}")

            clip_text_encoder = clip_text_encoder.eval().requires_grad_(False)

            text_inputs = clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = clip_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            assert isinstance(text_input_ids, torch.Tensor)
            assert isinstance(untruncated_ids, torch.Tensor)
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = clip_tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
                context.logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer_max_length} tokens: {removed_text}"
                )
            prompt_embeds = clip_text_encoder(
                input_ids=text_input_ids.to(TorchDevice.choose_torch_device()), output_hidden_states=True
            )
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]

            return prompt_embeds, pooled_prompt_embeds

    def _clip_lora_iterator(
        self, context: InvocationContext, clip_model: CLIPField
    ) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in clip_model.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info
