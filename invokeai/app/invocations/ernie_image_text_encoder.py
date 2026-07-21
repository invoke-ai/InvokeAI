import json
from contextlib import ExitStack
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    ErnieImageConditioningField,
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import Mistral3EncoderField, PromptEnhancerField
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
    version="1.0.0",
    classification=Classification.Prototype,
)
class ErnieImageTextEncoderInvocation(BaseInvocation):
    """Encodes a prompt for ERNIE-Image generation, optionally rewriting it via the
    bundled prompt-enhancer (Ministral3ForCausalLM) before tokenization.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)

    text_encoder: Mistral3EncoderField = InputField(
        title="Text Encoder",
        description="Mistral3 text encoder + tokenizer",
        input=Input.Connection,
    )

    prompt_enhancer: Optional[PromptEnhancerField] = InputField(
        default=None,
        title="Prompt Enhancer",
        description="If connected and `use_prompt_enhancer` is true, the PE model rewrites the prompt before encoding.",
        input=Input.Connection,
    )

    use_prompt_enhancer: bool = InputField(
        default=True,
        description="Whether to run the prompt-enhancer (no-op if no PE field is connected).",
        title="Use Prompt Enhancer",
    )

    pe_width: int = InputField(default=1024, description="Target width passed to the prompt enhancer.")
    pe_height: int = InputField(default=1024, description="Target height passed to the prompt enhancer.")
    pe_temperature: float = InputField(default=0.6, ge=0.0, le=2.0)
    pe_top_p: float = InputField(default=0.95, ge=0.0, le=1.0)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ErnieImageConditioningOutput:
        prompt = self.prompt
        if self.use_prompt_enhancer and self.prompt_enhancer is not None:
            prompt = self._enhance_prompt(context, prompt)
            context.logger.info(f"ERNIE-Image PE rewrote prompt -> {prompt!r}")

        prompt_embeds = self._encode_prompt(context, prompt)
        prompt_embeds = prompt_embeds.detach().to("cpu")
        conditioning_data = ConditioningFieldData(
            conditionings=[ErnieImageConditioningInfo(prompt_embeds=prompt_embeds)]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return ErnieImageConditioningOutput(
            conditioning=ErnieImageConditioningField(conditioning_name=conditioning_name)
        )

    def _enhance_prompt(self, context: InvocationContext, prompt: str) -> str:
        assert self.prompt_enhancer is not None  # checked by caller

        tokenizer_info = context.models.load(self.prompt_enhancer.tokenizer)
        lm_info = context.models.load(self.prompt_enhancer.text_encoder)

        with ExitStack() as exit_stack:
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())
            (_, lm) = exit_stack.enter_context(lm_info.model_on_device())

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"Expected tokenizer, got {type(tokenizer).__name__}")
            if not isinstance(lm, PreTrainedModel):
                raise TypeError(f"Expected PreTrainedModel for PE, got {type(lm).__name__}")

            user_content = json.dumps(
                {"prompt": prompt, "width": self.pe_width, "height": self.pe_height},
                ensure_ascii=False,
            )
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(lm.device)
            output_ids = lm.generate(
                **inputs,
                max_new_tokens=tokenizer.model_max_length,
                do_sample=self.pe_temperature != 1.0 or self.pe_top_p != 1.0,
                temperature=self.pe_temperature,
                top_p=self.pe_top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
