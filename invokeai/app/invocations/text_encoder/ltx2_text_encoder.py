"""LTX-2 prompt encoding (Gemma-4 tower + text connectors)."""

import torch

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LTX2ConditioningField,
    OutputField,
    UIComponent,
)
from invokeai.app.invocations.model import LTX2TextEncoderField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ltx2.constants import LTX2_DEFAULT_NEGATIVE_PROMPT, LTX2_MAX_SEQUENCE_LENGTH
from invokeai.backend.ltx2.text_conditioning import (
    apply_connectors,
    encode_hidden_states,
    estimate_connector_working_memory,
    estimate_tower_working_memory,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData


@invocation_output("ltx2_text_encoder_output")
class LTX2TextEncoderOutput(BaseInvocationOutput):
    """The positive and (when classifier-free guidance will run) negative LTX-2 conditioning."""

    conditioning: LTX2ConditioningField = OutputField(description=FieldDescriptions.cond)
    negative_conditioning: LTX2ConditioningField | None = OutputField(
        default=None, description="Conditioning for the unconditional guidance pass, if one was encoded."
    )


@invocation(
    "ltx2_text_encoder",
    title="Prompt - LTX-2",
    tags=["prompt", "conditioning", "ltx", "ltx2", "video"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class LTX2TextEncoderInvocation(BaseInvocation):
    """Encodes a prompt, and optionally a negative prompt, for LTX-2.

    Both prompts go through the same two stages: the Lightricks-tuned Gemma-4 tower, whose hidden
    states are taken from every layer rather than just the last, and the text connectors, which
    project them into the separate streams the video and audio halves of the transformer read.

    The negative prompt only reaches the model through classifier-free guidance, so it is skipped
    entirely for a distilled checkpoint (which runs none) rather than paying a 12B encode for a
    tensor nothing will read.
    """

    prompt: str = InputField(description="Text prompt for LTX-2.", ui_component=UIComponent.Textarea)
    text_encoder: LTX2TextEncoderField = InputField(
        title="Gemma-4 Encoder",
        description=FieldDescriptions.ltx2_text_encoder,
        input=Input.Connection,
    )
    negative_prompt: str = InputField(
        default=LTX2_DEFAULT_NEGATIVE_PROMPT,
        description="What to steer away from. Reaches the model only through classifier-free guidance.",
        ui_component=UIComponent.Textarea,
    )
    encode_negative: bool = InputField(
        default=True,
        description="Whether to encode the negative prompt. Turn off for a distilled checkpoint, which "
        "runs no unconditional pass.",
    )
    max_sequence_length: int = InputField(
        default=LTX2_MAX_SEQUENCE_LENGTH,
        gt=0,
        le=LTX2_MAX_SEQUENCE_LENGTH,
        description="Prompt tokens to keep. Every prompt is padded to this length, which is the length "
        "the connectors' learned registers are defined against.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LTX2TextEncoderOutput:
        prompts = [self.prompt]
        if self.encode_negative:
            prompts.append(self.negative_prompt)

        # The 12B tower and the 6 GB connectors run in two phases with the states parked in RAM in
        # between, so only one of them is resident at a time. Loading the tower first also keeps the
        # cache from evicting the small tokenizer record to make room for it (issue #7513).
        text_encoder_info = context.models.load(self.text_encoder.text_encoder)
        tokenizer_info = context.models.load(self.text_encoder.tokenizer)
        # Read off the unlocked model, before the VRAM lock the reservation applies to.
        tower_memory = estimate_tower_working_memory(text_encoder_info.model, self.max_sequence_length)
        hidden_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        with (
            tokenizer_info.model_on_device() as (_, tokenizer),
            text_encoder_info.model_on_device(working_mem_bytes=tower_memory) as (_, text_encoder),
        ):
            device = get_effective_device(text_encoder)
            context.util.signal_progress("Running the LTX-2 Gemma-4 text encoder")
            for prompt in prompts:
                states, mask = encode_hidden_states(
                    text_encoder,
                    tokenizer,
                    prompt,
                    max_sequence_length=self.max_sequence_length,
                    device=device,
                )
                hidden_states.append((states.cpu(), mask.cpu()))

        connectors_info = context.models.load(self.text_encoder.connectors)
        connector_memory = estimate_connector_working_memory(hidden_states[0][0])
        with connectors_info.model_on_device(working_mem_bytes=connector_memory) as (_, connectors):
            context.util.signal_progress("Running the LTX-2 text connectors")
            names = [
                context.conditioning.save(ConditioningFieldData(conditionings=[apply_connectors(connectors, *state)]))
                for state in hidden_states
            ]

        return LTX2TextEncoderOutput(
            conditioning=LTX2ConditioningField(conditioning_name=names[0]),
            negative_conditioning=LTX2ConditioningField(conditioning_name=names[1]) if len(names) > 1 else None,
        )
