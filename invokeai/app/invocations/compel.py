from typing import Iterator, List, Optional, Tuple, Union, cast

import torch
from compel import Compel, ReturnedEmbeddingsType
from compel.prompt_parser import Blend, Conjunction, CrossAttentionControlSubstitute, FlattenedPrompt, Fragment
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIComponent
from invokeai.app.invocations.primitives import ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.ti_utils import generate_ti_list
from invokeai.backend.lora.lora_model import LoRAModelRaw
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    ConditioningFieldData,
    ExtraConditioningInfo,
    SDXLConditioningInfo,
)
from invokeai.backend.util.devices import torch_dtype

from .baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from .model import CLIPField

# unconditioned: Optional[torch.Tensor]


# class ConditioningAlgo(str, Enum):
#    Compose = "compose"
#    ComposeEx = "compose_ex"
#    PerpNeg = "perp_neg"


@invocation(
    "compel",
    title="Prompt",
    tags=["prompt", "compel"],
    category="conditioning",
    version="1.1.1",
)
class CompelInvocation(BaseInvocation):
    """Parse prompt using compel package to conditioning."""

    prompt: str = InputField(
        default="",
        description=FieldDescriptions.compel_prompt,
        ui_component=UIComponent.Textarea,
    )
    clip: CLIPField = InputField(
        title="CLIP",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        tokenizer_info = context.models.load(self.clip.tokenizer)
        tokenizer_model = tokenizer_info.model
        assert isinstance(tokenizer_model, CLIPTokenizer)
        text_encoder_info = context.models.load(self.clip.text_encoder)
        text_encoder_model = text_encoder_info.model
        assert isinstance(text_encoder_model, CLIPTextModel)

        def _lora_loader() -> Iterator[Tuple[LoRAModelRaw, float]]:
            for lora in self.clip.loras:
                lora_info = context.models.load(lora.lora)
                assert isinstance(lora_info.model, LoRAModelRaw)
                yield (lora_info.model, lora.weight)
                del lora_info
            return

        # loras = [(context.models.get(**lora.dict(exclude={"weight"})).context.model, lora.weight) for lora in self.clip.loras]

        ti_list = generate_ti_list(self.prompt, text_encoder_info.config.base, context)

        with (
            ModelPatcher.apply_ti(tokenizer_model, text_encoder_model, ti_list) as (
                tokenizer,
                ti_manager,
            ),
            text_encoder_info as text_encoder,
            # Apply the LoRA after text_encoder has been moved to its target device for faster patching.
            ModelPatcher.apply_lora_text_encoder(text_encoder, _lora_loader()),
            # Apply CLIP Skip after LoRA to prevent LoRA application from failing on skipped layers.
            ModelPatcher.apply_clip_skip(text_encoder_model, self.clip.skipped_layers),
        ):
            assert isinstance(text_encoder, CLIPTextModel)
            compel = Compel(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=ti_manager,
                dtype_for_device_getter=torch_dtype,
                truncate_long_prompts=False,
            )

            conjunction = Compel.parse_prompt_string(self.prompt)

            if context.config.get().log_tokenization:
                log_tokenization_for_conjunction(conjunction, tokenizer)

            c, options = compel.build_conditioning_tensor_for_conjunction(conjunction)

            ec = ExtraConditioningInfo(
                tokens_count_including_eos_bos=get_max_token_count(tokenizer, conjunction),
                cross_attention_control_args=options.get("cross_attention_control", None),
            )

        c = c.detach().to("cpu")

        conditioning_data = ConditioningFieldData(
            conditionings=[
                BasicConditioningInfo(
                    embeds=c,
                    extra_conditioning=ec,
                )
            ]
        )

        conditioning_name = context.conditioning.save(conditioning_data)

        return ConditioningOutput.build(conditioning_name)


class SDXLPromptInvocationBase:
    """Prompt processor for SDXL models."""

    def run_clip_compel(
        self,
        context: InvocationContext,
        clip_field: CLIPField,
        prompt: str,
        get_pooled: bool,
        lora_prefix: str,
        zero_on_empty: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[ExtraConditioningInfo]]:
        tokenizer_info = context.models.load(clip_field.tokenizer)
        tokenizer_model = tokenizer_info.model
        assert isinstance(tokenizer_model, CLIPTokenizer)
        text_encoder_info = context.models.load(clip_field.text_encoder)
        text_encoder_model = text_encoder_info.model
        assert isinstance(text_encoder_model, (CLIPTextModel, CLIPTextModelWithProjection))

        # return zero on empty
        if prompt == "" and zero_on_empty:
            cpu_text_encoder = text_encoder_info.model
            assert isinstance(cpu_text_encoder, torch.nn.Module)
            c = torch.zeros(
                (
                    1,
                    cpu_text_encoder.config.max_position_embeddings,
                    cpu_text_encoder.config.hidden_size,
                ),
                dtype=cpu_text_encoder.dtype,
            )
            if get_pooled:
                c_pooled = torch.zeros(
                    (1, cpu_text_encoder.config.hidden_size),
                    dtype=c.dtype,
                )
            else:
                c_pooled = None
            return c, c_pooled, None

        def _lora_loader() -> Iterator[Tuple[LoRAModelRaw, float]]:
            for lora in clip_field.loras:
                lora_info = context.models.load(lora.lora)
                lora_model = lora_info.model
                assert isinstance(lora_model, LoRAModelRaw)
                yield (lora_model, lora.weight)
                del lora_info
            return

        # loras = [(context.models.get(**lora.dict(exclude={"weight"})).context.model, lora.weight) for lora in self.clip.loras]

        ti_list = generate_ti_list(prompt, text_encoder_info.config.base, context)

        with (
            ModelPatcher.apply_ti(tokenizer_model, text_encoder_model, ti_list) as (
                tokenizer,
                ti_manager,
            ),
            text_encoder_info as text_encoder,
            # Apply the LoRA after text_encoder has been moved to its target device for faster patching.
            ModelPatcher.apply_lora(text_encoder, _lora_loader(), lora_prefix),
            # Apply CLIP Skip after LoRA to prevent LoRA application from failing on skipped layers.
            ModelPatcher.apply_clip_skip(text_encoder_model, clip_field.skipped_layers),
        ):
            assert isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection))
            text_encoder = cast(CLIPTextModel, text_encoder)
            compel = Compel(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=ti_manager,
                dtype_for_device_getter=torch_dtype,
                truncate_long_prompts=False,  # TODO:
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,  # TODO: clip skip
                requires_pooled=get_pooled,
            )

            conjunction = Compel.parse_prompt_string(prompt)

            if context.config.get().log_tokenization:
                # TODO: better logging for and syntax
                log_tokenization_for_conjunction(conjunction, tokenizer)

            # TODO: ask for optimizations? to not run text_encoder twice
            c, options = compel.build_conditioning_tensor_for_conjunction(conjunction)
            if get_pooled:
                c_pooled = compel.conditioning_provider.get_pooled_embeddings([prompt])
            else:
                c_pooled = None

            ec = ExtraConditioningInfo(
                tokens_count_including_eos_bos=get_max_token_count(tokenizer, conjunction),
                cross_attention_control_args=options.get("cross_attention_control", None),
            )

        del tokenizer
        del text_encoder
        del tokenizer_info
        del text_encoder_info

        c = c.detach().to("cpu")
        if c_pooled is not None:
            c_pooled = c_pooled.detach().to("cpu")

        return c, c_pooled, ec


@invocation(
    "sdxl_compel_prompt",
    title="SDXL Prompt",
    tags=["sdxl", "compel", "prompt"],
    category="conditioning",
    version="1.1.1",
)
class SDXLCompelPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    """Parse prompt using compel package to conditioning."""

    prompt: str = InputField(
        default="",
        description=FieldDescriptions.compel_prompt,
        ui_component=UIComponent.Textarea,
    )
    style: str = InputField(
        default="",
        description=FieldDescriptions.compel_prompt,
        ui_component=UIComponent.Textarea,
    )
    original_width: int = InputField(default=1024, description="")
    original_height: int = InputField(default=1024, description="")
    crop_top: int = InputField(default=0, description="")
    crop_left: int = InputField(default=0, description="")
    target_width: int = InputField(default=1024, description="")
    target_height: int = InputField(default=1024, description="")
    clip: CLIPField = InputField(description=FieldDescriptions.clip, input=Input.Connection, title="CLIP 1")
    clip2: CLIPField = InputField(description=FieldDescriptions.clip, input=Input.Connection, title="CLIP 2")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        c1, c1_pooled, ec1 = self.run_clip_compel(
            context, self.clip, self.prompt, False, "lora_te1_", zero_on_empty=True
        )
        if self.style.strip() == "":
            c2, c2_pooled, ec2 = self.run_clip_compel(
                context, self.clip2, self.prompt, True, "lora_te2_", zero_on_empty=True
            )
        else:
            c2, c2_pooled, ec2 = self.run_clip_compel(
                context, self.clip2, self.style, True, "lora_te2_", zero_on_empty=True
            )

        original_size = (self.original_height, self.original_width)
        crop_coords = (self.crop_top, self.crop_left)
        target_size = (self.target_height, self.target_width)

        add_time_ids = torch.tensor([original_size + crop_coords + target_size])

        # [1, 77, 768], [1, 154, 1280]
        if c1.shape[1] < c2.shape[1]:
            c1 = torch.cat(
                [
                    c1,
                    torch.zeros(
                        (c1.shape[0], c2.shape[1] - c1.shape[1], c1.shape[2]),
                        device=c1.device,
                        dtype=c1.dtype,
                    ),
                ],
                dim=1,
            )

        elif c1.shape[1] > c2.shape[1]:
            c2 = torch.cat(
                [
                    c2,
                    torch.zeros(
                        (c2.shape[0], c1.shape[1] - c2.shape[1], c2.shape[2]),
                        device=c2.device,
                        dtype=c2.dtype,
                    ),
                ],
                dim=1,
            )

        assert c2_pooled is not None
        conditioning_data = ConditioningFieldData(
            conditionings=[
                SDXLConditioningInfo(
                    embeds=torch.cat([c1, c2], dim=-1),
                    pooled_embeds=c2_pooled,
                    add_time_ids=add_time_ids,
                    extra_conditioning=ec1,
                )
            ]
        )

        conditioning_name = context.conditioning.save(conditioning_data)

        return ConditioningOutput.build(conditioning_name)


@invocation(
    "sdxl_refiner_compel_prompt",
    title="SDXL Refiner Prompt",
    tags=["sdxl", "compel", "prompt"],
    category="conditioning",
    version="1.1.1",
)
class SDXLRefinerCompelPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    """Parse prompt using compel package to conditioning."""

    style: str = InputField(
        default="",
        description=FieldDescriptions.compel_prompt,
        ui_component=UIComponent.Textarea,
    )  # TODO: ?
    original_width: int = InputField(default=1024, description="")
    original_height: int = InputField(default=1024, description="")
    crop_top: int = InputField(default=0, description="")
    crop_left: int = InputField(default=0, description="")
    aesthetic_score: float = InputField(default=6.0, description=FieldDescriptions.sdxl_aesthetic)
    clip2: CLIPField = InputField(description=FieldDescriptions.clip, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        # TODO: if there will appear lora for refiner - write proper prefix
        c2, c2_pooled, ec2 = self.run_clip_compel(context, self.clip2, self.style, True, "<NONE>", zero_on_empty=False)

        original_size = (self.original_height, self.original_width)
        crop_coords = (self.crop_top, self.crop_left)

        add_time_ids = torch.tensor([original_size + crop_coords + (self.aesthetic_score,)])

        assert c2_pooled is not None
        conditioning_data = ConditioningFieldData(
            conditionings=[
                SDXLConditioningInfo(
                    embeds=c2,
                    pooled_embeds=c2_pooled,
                    add_time_ids=add_time_ids,
                    extra_conditioning=ec2,  # or None
                )
            ]
        )

        conditioning_name = context.conditioning.save(conditioning_data)

        return ConditioningOutput.build(conditioning_name)


@invocation_output("clip_skip_output")
class CLIPSkipInvocationOutput(BaseInvocationOutput):
    """CLIP skip node output"""

    clip: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP")


@invocation(
    "clip_skip",
    title="CLIP Skip",
    tags=["clipskip", "clip", "skip"],
    category="conditioning",
    version="1.1.0",
)
class CLIPSkipInvocation(BaseInvocation):
    """Skip layers in clip text_encoder model."""

    clip: CLIPField = InputField(description=FieldDescriptions.clip, input=Input.Connection, title="CLIP")
    skipped_layers: int = InputField(default=0, ge=0, description=FieldDescriptions.skipped_layers)

    def invoke(self, context: InvocationContext) -> CLIPSkipInvocationOutput:
        self.clip.skipped_layers += self.skipped_layers
        return CLIPSkipInvocationOutput(
            clip=self.clip,
        )


def get_max_token_count(
    tokenizer: CLIPTokenizer,
    prompt: Union[FlattenedPrompt, Blend, Conjunction],
    truncate_if_too_long: bool = False,
) -> int:
    if type(prompt) is Blend:
        blend: Blend = prompt
        return max([get_max_token_count(tokenizer, p, truncate_if_too_long) for p in blend.prompts])
    elif type(prompt) is Conjunction:
        conjunction: Conjunction = prompt
        return sum([get_max_token_count(tokenizer, p, truncate_if_too_long) for p in conjunction.prompts])
    else:
        return len(get_tokens_for_prompt_object(tokenizer, prompt, truncate_if_too_long))


def get_tokens_for_prompt_object(
    tokenizer: CLIPTokenizer, parsed_prompt: FlattenedPrompt, truncate_if_too_long: bool = True
) -> List[str]:
    if type(parsed_prompt) is Blend:
        raise ValueError("Blend is not supported here - you need to get tokens for each of its .children")

    text_fragments = [
        (
            x.text
            if type(x) is Fragment
            else (" ".join([f.text for f in x.original]) if type(x) is CrossAttentionControlSubstitute else str(x))
        )
        for x in parsed_prompt.children
    ]
    text = " ".join(text_fragments)
    tokens: List[str] = tokenizer.tokenize(text)
    if truncate_if_too_long:
        max_tokens_length = tokenizer.model_max_length - 2  # typically 75
        tokens = tokens[0:max_tokens_length]
    return tokens


def log_tokenization_for_conjunction(
    c: Conjunction, tokenizer: CLIPTokenizer, display_label_prefix: Optional[str] = None
) -> None:
    display_label_prefix = display_label_prefix or ""
    for i, p in enumerate(c.prompts):
        if len(c.prompts) > 1:
            this_display_label_prefix = f"{display_label_prefix}(conjunction part {i + 1}, weight={c.weights[i]})"
        else:
            assert display_label_prefix is not None
            this_display_label_prefix = display_label_prefix
        log_tokenization_for_prompt_object(p, tokenizer, display_label_prefix=this_display_label_prefix)


def log_tokenization_for_prompt_object(
    p: Union[Blend, FlattenedPrompt], tokenizer: CLIPTokenizer, display_label_prefix: Optional[str] = None
) -> None:
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
            log_tokenization_for_text(text, tokenizer, display_label=display_label_prefix)


def log_tokenization_for_text(
    text: str,
    tokenizer: CLIPTokenizer,
    display_label: Optional[str] = None,
    truncate_if_too_long: Optional[bool] = False,
) -> None:
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
