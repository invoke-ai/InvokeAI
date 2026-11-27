import json
from contextlib import ExitStack
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import PromptEnhancerField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext

# Hard ceiling on the prompt-enhancer's generation length. Upstream drives `max_new_tokens` off the
# PE tokenizer's `model_max_length`, but that is unreliable as a bound: if the tokenizer config omits
# it, transformers substitutes a sentinel (int(1e30)), and a rewrite that never emits EOS would hang
# the graph. A rewritten image prompt is a few hundred tokens at most, so cap it.
PE_MAX_NEW_TOKENS = 1024


class _CancelStoppingCriteria(StoppingCriteria):
    """Halts `generate()` when the session's cancel event fires.

    `generate()` is a single opaque call from the graph's point of view, so without this a cancel
    only takes effect once the whole rewrite has been sampled. The caller re-checks the cancel flag
    afterwards and raises, so the truncated sequence this returns is never used.
    """

    def __init__(self, context: InvocationContext) -> None:
        super().__init__()
        self._context = context

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: object) -> bool:
        return self._context.util.is_canceled()


@invocation(
    "ernie_image_prompt_enhancer",
    title="Prompt Enhancer - ERNIE-Image",
    tags=["prompt", "ernie-image"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    # Deliberately NOT idle_gpu_offloadable, unlike `ernie_image_text_encoder` which this node was
    # split out of. Offloading holds the lent GPU's exclusive-use lock for the whole node, and this
    # one runs an autoregressive `generate()` of up to PE_MAX_NEW_TOKENS on *every* execution — a
    # cost that never amortizes into the borrowed device's model cache the way an encoder forward
    # does. Borrowing here would re-stall a GPU that another session may be waiting to start on.
)
class ErnieImagePromptEnhancerInvocation(BaseInvocation):
    """Rewrites a prompt for ERNIE-Image generation using the pipeline's bundled prompt-enhancer
    (Ministral3ForCausalLM), sized for the intended output dimensions.

    If no prompt-enhancer is connected — the pipeline may not ship one — the prompt is passed
    through unchanged.
    """

    prompt: str = InputField(description="Text prompt to rewrite.", ui_component=UIComponent.Textarea)

    prompt_enhancer: Optional[PromptEnhancerField] = InputField(
        default=None,
        title="Prompt Enhancer",
        description="The prompt-enhancer model. If not connected, the prompt is passed through unchanged.",
        input=Input.Connection,
    )

    width: int = InputField(default=1024, description="Target width the prompt is rewritten for.")
    height: int = InputField(default=1024, description="Target height the prompt is rewritten for.")
    temperature: float = InputField(default=0.6, ge=0.0, le=2.0)
    top_p: float = InputField(default=0.95, ge=0.0, le=1.0)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        if self.prompt_enhancer is None:
            return StringOutput(value=self.prompt)

        enhanced = self._enhance_prompt(context, self.prompt)
        context.logger.info(f"ERNIE-Image PE rewrote prompt -> {enhanced!r}")
        return StringOutput(value=enhanced)

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
                {"prompt": prompt, "width": self.width, "height": self.height},
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
                max_new_tokens=min(tokenizer.model_max_length, PE_MAX_NEW_TOKENS),
                do_sample=self.temperature != 1.0 or self.top_p != 1.0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_CancelStoppingCriteria(context)]),
            )
            # The stopping criterion above cuts generation short on cancel, leaving a partial
            # rewrite; discard it rather than encoding a truncated prompt.
            if context.util.is_canceled():
                raise CanceledException

            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
