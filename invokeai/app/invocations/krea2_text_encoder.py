import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3VLEncoderField
from invokeai.app.invocations.primitives import Krea2ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.sampling_utils import (
    KREA2_MAX_SEQ_LEN,
    KREA2_NUM_SUFFIX_TOKENS,
    KREA2_SELECT_LAYERS,
    KREA2_START_IDX,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Krea2ConditioningInfo,
)

# Prompt template from diffusers Krea2Pipeline.get_text_hidden_states. The prefix (a system turn that
# instructs the model to describe the image) is the same "generate" template used by Qwen-Image, which
# is why the first KREA2_START_IDX (34) tokens are dropped from the encoder output.
_KREA2_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
_KREA2_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


@invocation(
    "krea2_text_encoder",
    title="Prompt - Krea-2",
    tags=["prompt", "conditioning", "krea2", "krea-2"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Krea2TextEncoderInvocation(BaseInvocation):
    """Encodes a text prompt for Krea-2 using the Qwen3-VL text encoder.

    The encoder taps 12 decoder hidden-state layers and stacks them per token, producing a 4D
    conditioning tensor (B, seq, 12, hidden) that the Krea-2 transformer's text-fusion stage consumes.
    """

    prompt: str = InputField(description="Text prompt describing the desired image.", ui_component=UIComponent.Textarea)
    qwen3_vl_encoder: Qwen3VLEncoderField = InputField(
        title="Qwen3-VL Encoder",
        description=FieldDescriptions.qwen3_vl_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Krea2ConditioningOutput:
        prompt_embeds, prompt_mask = self._encode(context)
        prompt_embeds = prompt_embeds.detach().to("cpu")
        prompt_mask = prompt_mask.detach().to("cpu") if prompt_mask is not None else None

        conditioning_data = ConditioningFieldData(
            conditionings=[Krea2ConditioningInfo(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_mask)]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return Krea2ConditioningOutput.build(conditioning_name)

    def _encode(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokenizer_info = context.models.load(self.qwen3_vl_encoder.tokenizer)
        text_encoder_info = context.models.load(self.qwen3_vl_encoder.text_encoder)

        text = _KREA2_PREFIX + self.prompt + _KREA2_SUFFIX
        # diffusers caps the tokenizer length at max_sequence_length + start_idx - num_suffix_tokens.
        max_length = KREA2_MAX_SEQ_LEN + KREA2_START_IDX - KREA2_NUM_SUFFIX_TOKENS

        context.util.signal_progress("Running Qwen3-VL text encoder")

        with tokenizer_info as tokenizer, text_encoder_info.model_on_device() as (_, text_encoder):
            device = get_effective_device(text_encoder)

            model_inputs = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = model_inputs.input_ids.to(device=device)
            attention_mask = model_inputs.attention_mask.to(device=device)

            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            # Some VL models nest the language-model output; fall back to that if needed.
            hidden_states_tuple = getattr(outputs, "hidden_states", None)
            if hidden_states_tuple is None:
                lm_output = getattr(outputs, "language_model_outputs", None)
                hidden_states_tuple = getattr(lm_output, "hidden_states", None)
            if hidden_states_tuple is None:
                raise RuntimeError("Qwen3-VL encoder did not return hidden_states; cannot build Krea-2 conditioning.")

            # Stack the selected layers along a new layer axis: (B, seq, 12, hidden).
            stacked = torch.stack([hidden_states_tuple[i] for i in KREA2_SELECT_LAYERS], dim=2)

            # Drop the system-prompt prefix tokens.
            prompt_embeds = stacked[:, KREA2_START_IDX:]
            prompt_mask = attention_mask[:, KREA2_START_IDX:].bool()

            prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)

        # If every token is valid (no padding), the mask is unnecessary.
        if prompt_mask is not None and bool(prompt_mask.all()):
            prompt_mask = None

        return prompt_embeds, prompt_mask
