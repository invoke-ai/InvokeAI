from typing import Literal, Optional, Union
from pydantic import BaseModel, Field

from invokeai.app.invocations.util.choose_model import choose_model
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig

from ...backend.util.devices import choose_torch_device, torch_dtype
from ...backend.stable_diffusion.diffusion import InvokeAIDiffuserComponent
from ...backend.stable_diffusion.textual_inversion_manager import TextualInversionManager

from compel import Compel
from compel.prompt_parser import (
    Blend,
    CrossAttentionControlSubstitute,
    FlattenedPrompt,
    Fragment,
)


class ConditioningField(BaseModel):
    conditioning_name: Optional[str] = Field(default=None, description="The name of conditioning data")
    class Config:
        schema_extra = {"required": ["conditioning_name"]}


class CompelOutput(BaseInvocationOutput):
    """Compel parser output"""

    #fmt: off
    type: Literal["compel_output"] = "compel_output"

    conditioning: ConditioningField = Field(default=None, description="Conditioning")
    #fmt: on


class CompelInvocation(BaseInvocation):
    """Parse prompt using compel package to conditioning."""

    type: Literal["compel"] = "compel"

    prompt: str = Field(default="", description="Prompt")
    model: str = Field(default="", description="Model to use")

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Prompt (Compel)",
                "tags": ["prompt", "compel"],
                "type_hints": {
                  "model": "model"
                }
            },
        }

    def invoke(self, context: InvocationContext) -> CompelOutput:

        # TODO: load without model
        model = choose_model(context.services.model_manager, self.model)
        pipeline = model["model"]
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder

        # TODO: global? input?
        #use_full_precision = precision == "float32" or precision == "autocast"
        #use_full_precision = False

        # TODO: redo TI when separate model loding implemented
        #textual_inversion_manager = TextualInversionManager(
        #    tokenizer=tokenizer,
        #    text_encoder=text_encoder,
        #    full_precision=use_full_precision,
        #)

        def load_huggingface_concepts(concepts: list[str]):
            pipeline.textual_inversion_manager.load_huggingface_concepts(concepts)

        # apply the concepts library to the prompt
        prompt_str = pipeline.textual_inversion_manager.hf_concepts_library.replace_concepts_with_triggers(
            self.prompt,
            lambda concepts: load_huggingface_concepts(concepts),
            pipeline.textual_inversion_manager.get_all_trigger_strings(),
        )

        # lazy-load any deferred textual inversions.
        # this might take a couple of seconds the first time a textual inversion is used.
        pipeline.textual_inversion_manager.create_deferred_token_ids_for_any_trigger_terms(
            prompt_str
        )

        compel = Compel(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            textual_inversion_manager=pipeline.textual_inversion_manager,
            dtype_for_device_getter=torch_dtype,
            truncate_long_prompts=True, # TODO:
        )

        # TODO: support legacy blend?

        conjunction = Compel.parse_prompt_string(prompt_str)
        prompt: Union[FlattenedPrompt, Blend] = conjunction.prompts[0]

        if context.services.configuration.log_tokenization:
            log_tokenization_for_prompt_object(prompt, tokenizer)

        c, options = compel.build_conditioning_tensor_for_prompt_object(prompt)

        # TODO: long prompt support
        #if not self.truncate_long_prompts:
        #    [c, uc] = compel.pad_conditioning_tensors_to_same_length([c, uc])

        ec = InvokeAIDiffuserComponent.ExtraConditioningInfo(
            tokens_count_including_eos_bos=get_max_token_count(tokenizer, prompt),
            cross_attention_control_args=options.get("cross_attention_control", None),
        )

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"

        # TODO: hacky but works ;D maybe rename latents somehow?
        context.services.latents.save(conditioning_name, (c, ec))

        return CompelOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
        )


def get_max_token_count(
    tokenizer, prompt: Union[FlattenedPrompt, Blend], truncate_if_too_long=False
) -> int:
    if type(prompt) is Blend:
        blend: Blend = prompt
        return max(
            [
                get_max_token_count(tokenizer, c, truncate_if_too_long)
                for c in blend.prompts
            ]
        )
    else:
        return len(
            get_tokens_for_prompt_object(tokenizer, prompt, truncate_if_too_long)
        )


def get_tokens_for_prompt_object(
    tokenizer, parsed_prompt: FlattenedPrompt, truncate_if_too_long=True
) -> [str]:
    if type(parsed_prompt) is Blend:
        raise ValueError(
            "Blend is not supported here - you need to get tokens for each of its .children"
        )

    text_fragments = [
        x.text
        if type(x) is Fragment
        else (
            " ".join([f.text for f in x.original])
            if type(x) is CrossAttentionControlSubstitute
            else str(x)
        )
        for x in parsed_prompt.children
    ]
    text = " ".join(text_fragments)
    tokens = tokenizer.tokenize(text)
    if truncate_if_too_long:
        max_tokens_length = tokenizer.model_max_length - 2  # typically 75
        tokens = tokens[0:max_tokens_length]
    return tokens


def log_tokenization_for_prompt_object(
    p: Union[Blend, FlattenedPrompt], tokenizer, display_label_prefix=None
):
    display_label_prefix = display_label_prefix or ""
    if type(p) is Blend:
        blend: Blend = p
        for i, c in enumerate(blend.prompts):
            log_tokenization_for_prompt_object(
                c,
                tokenizer,
                display_label_prefix=f"{display_label_prefix}(blend part {i + 1}, weight={blend.weights[i]})",
            )
    elif type(p) is FlattenedPrompt:
        flattened_prompt: FlattenedPrompt = p
        if flattened_prompt.wants_cross_attention_control:
            original_fragments = []
            edited_fragments = []
            for f in flattened_prompt.children:
                if type(f) is CrossAttentionControlSubstitute:
                    original_fragments += f.original
                    edited_fragments += f.edited
                else:
                    original_fragments.append(f)
                    edited_fragments.append(f)

            original_text = " ".join([x.text for x in original_fragments])
            log_tokenization_for_text(
                original_text,
                tokenizer,
                display_label=f"{display_label_prefix}(.swap originals)",
            )
            edited_text = " ".join([x.text for x in edited_fragments])
            log_tokenization_for_text(
                edited_text,
                tokenizer,
                display_label=f"{display_label_prefix}(.swap replacements)",
            )
        else:
            text = " ".join([x.text for x in flattened_prompt.children])
            log_tokenization_for_text(
                text, tokenizer, display_label=display_label_prefix
            )


def log_tokenization_for_text(text, tokenizer, display_label=None, truncate_if_too_long=False):
    """shows how the prompt is tokenized
    # usually tokens have '</w>' to indicate end-of-word,
    # but for readability it has been replaced with ' '
    """
    tokens = tokenizer.tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)

    for i in range(0, totalTokens):
        token = tokens[i].replace("</w>", " ")
        # alternate color
        s = (usedTokens % 6) + 1
        if truncate_if_too_long and i >= tokenizer.model_max_length:
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
        else:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1

    if usedTokens > 0:
        print(f'\n>> [TOKENLOG] Tokens {display_label or ""} ({usedTokens}):')
        print(f"{tokenized}\x1b[0m")

    if discarded != "":
        print(f"\n>> [TOKENLOG] Tokens Discarded ({totalTokens - usedTokens}):")
        print(f"{discarded}\x1b[0m")
