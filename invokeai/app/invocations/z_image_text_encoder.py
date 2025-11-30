import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.invocations.primitives import ZImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    ZImageConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

# Z-Image max sequence length based on diffusers default
Z_IMAGE_MAX_SEQ_LEN = 512


@invocation(
    "z_image_text_encoder",
    title="Prompt - Z-Image",
    tags=["prompt", "conditioning", "z-image"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ZImageTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for a Z-Image image."""

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ZImageConditioningOutput:
        prompt_embeds = self._encode_prompt(context, max_seq_len=Z_IMAGE_MAX_SEQ_LEN)
        conditioning_data = ConditioningFieldData(conditionings=[ZImageConditioningInfo(prompt_embeds=prompt_embeds)])
        conditioning_name = context.conditioning.save(conditioning_data)
        return ZImageConditioningOutput.build(conditioning_name)

    def _encode_prompt(self, context: InvocationContext, max_seq_len: int) -> torch.Tensor:
        """Encode prompt using Qwen3 text encoder.

        Based on the ZImagePipeline._encode_prompt method from diffusers.
        """
        prompt = self.prompt

        with (
            context.models.load(self.qwen3_encoder.text_encoder).model_on_device() as (_, text_encoder),
            context.models.load(self.qwen3_encoder.tokenizer).model_on_device() as (_, tokenizer),
        ):
            context.util.signal_progress("Running Qwen3 text encoder")
            assert isinstance(text_encoder, PreTrainedModel)
            assert isinstance(tokenizer, PreTrainedTokenizerBase)

            # Apply chat template similar to diffusers ZImagePipeline
            # The chat template formats the prompt for the Qwen3 model
            prompt_formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            # Tokenize the formatted prompt
            text_inputs = tokenizer(
                prompt_formatted,
                padding="max_length",
                max_length=max_seq_len,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            assert isinstance(text_input_ids, torch.Tensor)
            assert isinstance(attention_mask, torch.Tensor)

            # Check for truncation
            untruncated_ids = tokenizer(prompt_formatted, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, max_seq_len - 1 : -1])
                context.logger.warning(
                    f"The following part of your input was truncated because `max_sequence_length` is set to "
                    f"{max_seq_len} tokens: {removed_text}"
                )

            device = TorchDevice.choose_torch_device()

            # Get hidden states from the text encoder
            # Use the second-to-last hidden state like diffusers does
            prompt_mask = attention_mask.to(device).bool()
            outputs = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_mask,
                output_hidden_states=True,
            )
            prompt_embeds = outputs.hidden_states[-2]

            # Z-Image expects a 2D tensor [seq_len, hidden_dim] with only valid tokens
            # Based on diffusers ZImagePipeline implementation:
            # embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
            # Since batch_size=1, we take the first item and filter by mask
            prompt_embeds = prompt_embeds[0][prompt_mask[0]]

        assert isinstance(prompt_embeds, torch.Tensor)
        return prompt_embeds
