from contextlib import ExitStack

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    ErnieImageConditioningField,
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import Mistral3EncoderField
from invokeai.app.invocations.primitives import ErnieImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    ErnieImageConditioningInfo,
)


@invocation(
    "ernie_image_text_encoder",
    title="Prompt - ERNIE-Image",
    tags=["prompt", "conditioning", "ernie-image"],
    category="conditioning",
    version="2.0.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class ErnieImageTextEncoderInvocation(BaseInvocation):
    """Encodes a prompt for ERNIE-Image generation.

    Rewriting a prompt with the pipeline's bundled prompt-enhancer is the separate
    `ernie_image_prompt_enhancer` node; connect its output to `prompt` to enhance. Keeping the two
    apart is what lets this node stay `idle_gpu_offloadable` — see that node's decorator for why the
    enhancer must not be.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)

    text_encoder: Mistral3EncoderField = InputField(
        title="Text Encoder",
        description="Mistral3 text encoder + tokenizer",
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ErnieImageConditioningOutput:
        prompt_embeds = self._encode_prompt(context, self.prompt)
        prompt_embeds = prompt_embeds.detach().to("cpu")
        conditioning_data = ConditioningFieldData(
            conditionings=[ErnieImageConditioningInfo(prompt_embeds=prompt_embeds)]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return ErnieImageConditioningOutput(
            conditioning=ErnieImageConditioningField(conditioning_name=conditioning_name)
        )

    def _encode_prompt(self, context: InvocationContext, prompt: str) -> torch.Tensor:
        text_encoder_info = context.models.load(self.text_encoder.text_encoder)
        tokenizer_info = context.models.load(self.text_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (_, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            if not isinstance(text_encoder, PreTrainedModel):
                raise TypeError(f"Expected PreTrainedModel, got {type(text_encoder).__name__}")
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"Expected tokenizer, got {type(tokenizer).__name__}")

            ids = tokenizer(prompt, add_special_tokens=True, truncation=True, padding=False)["input_ids"]
            if not ids:
                ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0]

            input_ids = torch.tensor([ids], device=text_encoder.device)
            outputs = text_encoder(input_ids=input_ids, output_hidden_states=True)
            if not getattr(outputs, "hidden_states", None) or len(outputs.hidden_states) < 2:
                raise RuntimeError("Mistral3 encoder did not return enough hidden states")

            # Match upstream pipeline: second-to-last hidden state, single batch -> [T, H]
            return outputs.hidden_states[-2][0]
