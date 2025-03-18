import torch
from transformers import GlmModel, PreTrainedTokenizerFast

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import GlmEncoderField
from invokeai.app.invocations.primitives import CogView4ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    CogView4ConditioningInfo,
    ConditioningFieldData,
)
from invokeai.backend.util.devices import TorchDevice

# The CogView4 GLM Text Encoder max sequence length set based on the default in diffusers.
COGVIEW4_GLM_MAX_SEQ_LEN = 1024


@invocation(
    "cogview4_text_encoder",
    title="Prompt - CogView4",
    tags=["prompt", "conditioning", "cogview4"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class CogView4TextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for a cogview4 image."""

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    glm_encoder: GlmEncoderField = InputField(
        title="GLM Encoder",
        description=FieldDescriptions.glm_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> CogView4ConditioningOutput:
        glm_embeds = self._glm_encode(context, max_seq_len=COGVIEW4_GLM_MAX_SEQ_LEN)
        conditioning_data = ConditioningFieldData(conditionings=[CogView4ConditioningInfo(glm_embeds=glm_embeds)])
        conditioning_name = context.conditioning.save(conditioning_data)
        return CogView4ConditioningOutput.build(conditioning_name)

    def _glm_encode(self, context: InvocationContext, max_seq_len: int) -> torch.Tensor:
        prompt = [self.prompt]

        # TODO(ryand): Add model inputs to the invocation rather than hard-coding.
        with (
            context.models.load(self.glm_encoder.text_encoder).model_on_device() as (_, glm_text_encoder),
            context.models.load(self.glm_encoder.tokenizer).model_on_device() as (_, glm_tokenizer),
        ):
            context.util.signal_progress("Running GLM text encoder")
            assert isinstance(glm_text_encoder, GlmModel)
            assert isinstance(glm_tokenizer, PreTrainedTokenizerFast)

            text_inputs = glm_tokenizer(
                prompt,
                padding="longest",
                max_length=max_seq_len,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = glm_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            assert isinstance(text_input_ids, torch.Tensor)
            assert isinstance(untruncated_ids, torch.Tensor)
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = glm_tokenizer.batch_decode(untruncated_ids[:, max_seq_len - 1 : -1])
                context.logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_seq_len} tokens: {removed_text}"
                )

            current_length = text_input_ids.shape[1]
            pad_length = (16 - (current_length % 16)) % 16
            if pad_length > 0:
                pad_ids = torch.full(
                    (text_input_ids.shape[0], pad_length),
                    fill_value=glm_tokenizer.pad_token_id,
                    dtype=text_input_ids.dtype,
                    device=text_input_ids.device,
                )
                text_input_ids = torch.cat([pad_ids, text_input_ids], dim=1)
            prompt_embeds = glm_text_encoder(
                text_input_ids.to(TorchDevice.choose_torch_device()), output_hidden_states=True
            ).hidden_states[-2]

        assert isinstance(prompt_embeds, torch.Tensor)
        return prompt_embeds
