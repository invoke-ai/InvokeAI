from typing import Literal, Optional, Union, List, Annotated
from pydantic import BaseModel, Field
import re
import torch
from compel import Compel, ReturnedEmbeddingsType
from compel.prompt_parser import Blend, Conjunction, CrossAttentionControlSubstitute, FlattenedPrompt, Fragment
from ...backend.util.devices import torch_dtype
from ...backend.model_management import ModelType
from ...backend.model_management.models import ModelNotFoundException
from ...backend.model_management.lora import ModelPatcher
from ...backend.stable_diffusion.diffusion import InvokeAIDiffuserComponent
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationConfig, InvocationContext
from .model import ClipField
from dataclasses import dataclass


class ConditioningField(BaseModel):
    conditioning_name: Optional[str] = Field(default=None, description="The name of conditioning data")

    class Config:
        schema_extra = {"required": ["conditioning_name"]}


@dataclass
class BasicConditioningInfo:
    # type: Literal["basic_conditioning"] = "basic_conditioning"
    embeds: torch.Tensor
    extra_conditioning: Optional[InvokeAIDiffuserComponent.ExtraConditioningInfo]
    # weight: float
    # mode: ConditioningAlgo


@dataclass
class SDXLConditioningInfo(BasicConditioningInfo):
    # type: Literal["sdxl_conditioning"] = "sdxl_conditioning"
    pooled_embeds: torch.Tensor
    add_time_ids: torch.Tensor


ConditioningInfoType = Annotated[Union[BasicConditioningInfo, SDXLConditioningInfo], Field(discriminator="type")]


@dataclass
class ConditioningFieldData:
    conditionings: List[Union[BasicConditioningInfo, SDXLConditioningInfo]]
    # unconditioned: Optional[torch.Tensor]


# class ConditioningAlgo(str, Enum):
#    Compose = "compose"
#    ComposeEx = "compose_ex"
#    PerpNeg = "perp_neg"


class CompelOutput(BaseInvocationOutput):
    """Compel parser output"""

    # fmt: off
    type: Literal["compel_output"] = "compel_output"

    conditioning: ConditioningField = Field(default=None, description="Conditioning")
    # fmt: on


class CompelInvocation(BaseInvocation):
    """Parse prompt using compel package to conditioning."""

    type: Literal["compel"] = "compel"

    prompt: str = Field(default="", description="Prompt")
    clip: ClipField = Field(None, description="Clip to use")

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Prompt (Compel)", "tags": ["prompt", "compel"], "type_hints": {"model": "model"}},
        }

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> CompelOutput:
        tokenizer_info = context.services.model_manager.get_model(
            **self.clip.tokenizer.dict(),
            context=context,
        )
        text_encoder_info = context.services.model_manager.get_model(
            **self.clip.text_encoder.dict(),
            context=context,
        )

        def _lora_loader():
            for lora in self.clip.loras:
                lora_info = context.services.model_manager.get_model(**lora.dict(exclude={"weight"}), context=context)
                yield (lora_info.context.model, lora.weight)
                del lora_info
            return

        # loras = [(context.services.model_manager.get_model(**lora.dict(exclude={"weight"})).context.model, lora.weight) for lora in self.clip.loras]

        ti_list = []
        for trigger in re.findall(r"<[a-zA-Z0-9., _-]+>", self.prompt):
            name = trigger[1:-1]
            try:
                ti_list.append(
                    context.services.model_manager.get_model(
                        model_name=name,
                        base_model=self.clip.text_encoder.base_model,
                        model_type=ModelType.TextualInversion,
                        context=context,
                    ).context.model
                )
            except ModelNotFoundException:
                # print(e)
                # import traceback
                # print(traceback.format_exc())
                print(f'Warn: trigger: "{trigger}" not found')

        with ModelPatcher.apply_lora_text_encoder(
            text_encoder_info.context.model, _lora_loader()
        ), ModelPatcher.apply_ti(tokenizer_info.context.model, text_encoder_info.context.model, ti_list) as (
            tokenizer,
            ti_manager,
        ), ModelPatcher.apply_clip_skip(
            text_encoder_info.context.model, self.clip.skipped_layers
        ), text_encoder_info as text_encoder:
            compel = Compel(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=ti_manager,
                dtype_for_device_getter=torch_dtype,
                truncate_long_prompts=True,
            )

            conjunction = Compel.parse_prompt_string(self.prompt)
            prompt: Union[FlattenedPrompt, Blend] = conjunction.prompts[0]

            if context.services.configuration.log_tokenization:
                log_tokenization_for_prompt_object(prompt, tokenizer)

            c, options = compel.build_conditioning_tensor_for_prompt_object(prompt)

            ec = InvokeAIDiffuserComponent.ExtraConditioningInfo(
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

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"
        context.services.latents.save(conditioning_name, conditioning_data)

        return CompelOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
        )


class SDXLPromptInvocationBase:
    def run_clip_raw(self, context, clip_field, prompt, get_pooled):
        tokenizer_info = context.services.model_manager.get_model(
            **clip_field.tokenizer.dict(),
            context=context,
        )
        text_encoder_info = context.services.model_manager.get_model(
            **clip_field.text_encoder.dict(),
            context=context,
        )

        def _lora_loader():
            for lora in clip_field.loras:
                lora_info = context.services.model_manager.get_model(**lora.dict(exclude={"weight"}), context=context)
                yield (lora_info.context.model, lora.weight)
                del lora_info
            return

        # loras = [(context.services.model_manager.get_model(**lora.dict(exclude={"weight"})).context.model, lora.weight) for lora in self.clip.loras]

        ti_list = []
        for trigger in re.findall(r"<[a-zA-Z0-9., _-]+>", prompt):
            name = trigger[1:-1]
            try:
                ti_list.append(
                    context.services.model_manager.get_model(
                        model_name=name,
                        base_model=clip_field.text_encoder.base_model,
                        model_type=ModelType.TextualInversion,
                        context=context,
                    ).context.model
                )
            except ModelNotFoundException:
                # print(e)
                # import traceback
                # print(traceback.format_exc())
                print(f'Warn: trigger: "{trigger}" not found')

        with ModelPatcher.apply_lora_text_encoder(
            text_encoder_info.context.model, _lora_loader()
        ), ModelPatcher.apply_ti(tokenizer_info.context.model, text_encoder_info.context.model, ti_list) as (
            tokenizer,
            ti_manager,
        ), ModelPatcher.apply_clip_skip(
            text_encoder_info.context.model, clip_field.skipped_layers
        ), text_encoder_info as text_encoder:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )
            if get_pooled:
                c_pooled = prompt_embeds[0]
            else:
                c_pooled = None
            c = prompt_embeds.hidden_states[-2]

        del tokenizer
        del text_encoder
        del tokenizer_info
        del text_encoder_info

        c = c.detach().to("cpu")
        if c_pooled is not None:
            c_pooled = c_pooled.detach().to("cpu")

        return c, c_pooled, None

    def run_clip_compel(self, context, clip_field, prompt, get_pooled):
        tokenizer_info = context.services.model_manager.get_model(
            **clip_field.tokenizer.dict(),
            context=context,
        )
        text_encoder_info = context.services.model_manager.get_model(
            **clip_field.text_encoder.dict(),
            context=context,
        )

        def _lora_loader():
            for lora in clip_field.loras:
                lora_info = context.services.model_manager.get_model(**lora.dict(exclude={"weight"}), context=context)
                yield (lora_info.context.model, lora.weight)
                del lora_info
            return

        # loras = [(context.services.model_manager.get_model(**lora.dict(exclude={"weight"})).context.model, lora.weight) for lora in self.clip.loras]

        ti_list = []
        for trigger in re.findall(r"<[a-zA-Z0-9., _-]+>", prompt):
            name = trigger[1:-1]
            try:
                ti_list.append(
                    context.services.model_manager.get_model(
                        model_name=name,
                        base_model=clip_field.text_encoder.base_model,
                        model_type=ModelType.TextualInversion,
                        context=context,
                    ).context.model
                )
            except ModelNotFoundException:
                # print(e)
                # import traceback
                # print(traceback.format_exc())
                print(f'Warn: trigger: "{trigger}" not found')

        with ModelPatcher.apply_lora_text_encoder(
            text_encoder_info.context.model, _lora_loader()
        ), ModelPatcher.apply_ti(tokenizer_info.context.model, text_encoder_info.context.model, ti_list) as (
            tokenizer,
            ti_manager,
        ), ModelPatcher.apply_clip_skip(
            text_encoder_info.context.model, clip_field.skipped_layers
        ), text_encoder_info as text_encoder:
            compel = Compel(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=ti_manager,
                dtype_for_device_getter=torch_dtype,
                truncate_long_prompts=True,  # TODO:
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,  # TODO: clip skip
                requires_pooled=True,
            )

            conjunction = Compel.parse_prompt_string(prompt)

            if context.services.configuration.log_tokenization:
                # TODO: better logging for and syntax
                for prompt_obj in conjunction.prompts:
                    log_tokenization_for_prompt_object(prompt_obj, tokenizer)

            # TODO: ask for optimizations? to not run text_encoder twice
            c, options = compel.build_conditioning_tensor_for_conjunction(conjunction)
            if get_pooled:
                c_pooled = compel.conditioning_provider.get_pooled_embeddings([prompt])
            else:
                c_pooled = None

            ec = InvokeAIDiffuserComponent.ExtraConditioningInfo(
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


class SDXLCompelPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    """Parse prompt using compel package to conditioning."""

    type: Literal["sdxl_compel_prompt"] = "sdxl_compel_prompt"

    prompt: str = Field(default="", description="Prompt")
    style: str = Field(default="", description="Style prompt")
    original_width: int = Field(1024, description="")
    original_height: int = Field(1024, description="")
    crop_top: int = Field(0, description="")
    crop_left: int = Field(0, description="")
    target_width: int = Field(1024, description="")
    target_height: int = Field(1024, description="")
    clip: ClipField = Field(None, description="Clip to use")
    clip2: ClipField = Field(None, description="Clip2 to use")

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "SDXL Prompt (Compel)", "tags": ["prompt", "compel"], "type_hints": {"model": "model"}},
        }

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> CompelOutput:
        c1, c1_pooled, ec1 = self.run_clip_compel(context, self.clip, self.prompt, False)
        if self.style.strip() == "":
            c2, c2_pooled, ec2 = self.run_clip_compel(context, self.clip2, self.prompt, True)
        else:
            c2, c2_pooled, ec2 = self.run_clip_compel(context, self.clip2, self.style, True)

        original_size = (self.original_height, self.original_width)
        crop_coords = (self.crop_top, self.crop_left)
        target_size = (self.target_height, self.target_width)

        add_time_ids = torch.tensor([original_size + crop_coords + target_size])

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

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"
        context.services.latents.save(conditioning_name, conditioning_data)

        return CompelOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
        )


class SDXLRefinerCompelPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    """Parse prompt using compel package to conditioning."""

    type: Literal["sdxl_refiner_compel_prompt"] = "sdxl_refiner_compel_prompt"

    style: str = Field(default="", description="Style prompt")  # TODO: ?
    original_width: int = Field(1024, description="")
    original_height: int = Field(1024, description="")
    crop_top: int = Field(0, description="")
    crop_left: int = Field(0, description="")
    aesthetic_score: float = Field(6.0, description="")
    clip2: ClipField = Field(None, description="Clip to use")

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "SDXL Refiner Prompt (Compel)",
                "tags": ["prompt", "compel"],
                "type_hints": {"model": "model"},
            },
        }

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> CompelOutput:
        c2, c2_pooled, ec2 = self.run_clip_compel(context, self.clip2, self.style, True)

        original_size = (self.original_height, self.original_width)
        crop_coords = (self.crop_top, self.crop_left)

        add_time_ids = torch.tensor([original_size + crop_coords + (self.aesthetic_score,)])

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

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"
        context.services.latents.save(conditioning_name, conditioning_data)

        return CompelOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
        )


class SDXLRawPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    """Pass unmodified prompt to conditioning without compel processing."""

    type: Literal["sdxl_raw_prompt"] = "sdxl_raw_prompt"

    prompt: str = Field(default="", description="Prompt")
    style: str = Field(default="", description="Style prompt")
    original_width: int = Field(1024, description="")
    original_height: int = Field(1024, description="")
    crop_top: int = Field(0, description="")
    crop_left: int = Field(0, description="")
    target_width: int = Field(1024, description="")
    target_height: int = Field(1024, description="")
    clip: ClipField = Field(None, description="Clip to use")
    clip2: ClipField = Field(None, description="Clip2 to use")

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "SDXL Prompt (Raw)", "tags": ["prompt", "compel"], "type_hints": {"model": "model"}},
        }

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> CompelOutput:
        c1, c1_pooled, ec1 = self.run_clip_raw(context, self.clip, self.prompt, False)
        if self.style.strip() == "":
            c2, c2_pooled, ec2 = self.run_clip_raw(context, self.clip2, self.prompt, True)
        else:
            c2, c2_pooled, ec2 = self.run_clip_raw(context, self.clip2, self.style, True)

        original_size = (self.original_height, self.original_width)
        crop_coords = (self.crop_top, self.crop_left)
        target_size = (self.target_height, self.target_width)

        add_time_ids = torch.tensor([original_size + crop_coords + target_size])

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

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"
        context.services.latents.save(conditioning_name, conditioning_data)

        return CompelOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
        )


class SDXLRefinerRawPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    """Parse prompt using compel package to conditioning."""

    type: Literal["sdxl_refiner_raw_prompt"] = "sdxl_refiner_raw_prompt"

    style: str = Field(default="", description="Style prompt")  # TODO: ?
    original_width: int = Field(1024, description="")
    original_height: int = Field(1024, description="")
    crop_top: int = Field(0, description="")
    crop_left: int = Field(0, description="")
    aesthetic_score: float = Field(6.0, description="")
    clip2: ClipField = Field(None, description="Clip to use")

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "SDXL Refiner Prompt (Raw)",
                "tags": ["prompt", "compel"],
                "type_hints": {"model": "model"},
            },
        }

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> CompelOutput:
        c2, c2_pooled, ec2 = self.run_clip_raw(context, self.clip2, self.style, True)

        original_size = (self.original_height, self.original_width)
        crop_coords = (self.crop_top, self.crop_left)

        add_time_ids = torch.tensor([original_size + crop_coords + (self.aesthetic_score,)])

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

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"
        context.services.latents.save(conditioning_name, conditioning_data)

        return CompelOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
        )


class ClipSkipInvocationOutput(BaseInvocationOutput):
    """Clip skip node output"""

    type: Literal["clip_skip_output"] = "clip_skip_output"
    clip: ClipField = Field(None, description="Clip with skipped layers")


class ClipSkipInvocation(BaseInvocation):
    """Skip layers in clip text_encoder model."""

    type: Literal["clip_skip"] = "clip_skip"

    clip: ClipField = Field(None, description="Clip to use")
    skipped_layers: int = Field(0, description="Number of layers to skip in text_encoder")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "CLIP Skip", "tags": ["clip", "skip"]},
        }

    def invoke(self, context: InvocationContext) -> ClipSkipInvocationOutput:
        self.clip.skipped_layers += self.skipped_layers
        return ClipSkipInvocationOutput(
            clip=self.clip,
        )


def get_max_token_count(
    tokenizer, prompt: Union[FlattenedPrompt, Blend, Conjunction], truncate_if_too_long=False
) -> int:
    if type(prompt) is Blend:
        blend: Blend = prompt
        return max([get_max_token_count(tokenizer, p, truncate_if_too_long) for p in blend.prompts])
    elif type(prompt) is Conjunction:
        conjunction: Conjunction = prompt
        return sum([get_max_token_count(tokenizer, p, truncate_if_too_long) for p in conjunction.prompts])
    else:
        return len(get_tokens_for_prompt_object(tokenizer, prompt, truncate_if_too_long))


def get_tokens_for_prompt_object(tokenizer, parsed_prompt: FlattenedPrompt, truncate_if_too_long=True) -> List[str]:
    if type(parsed_prompt) is Blend:
        raise ValueError("Blend is not supported here - you need to get tokens for each of its .children")

    text_fragments = [
        x.text
        if type(x) is Fragment
        else (" ".join([f.text for f in x.original]) if type(x) is CrossAttentionControlSubstitute else str(x))
        for x in parsed_prompt.children
    ]
    text = " ".join(text_fragments)
    tokens = tokenizer.tokenize(text)
    if truncate_if_too_long:
        max_tokens_length = tokenizer.model_max_length - 2  # typically 75
        tokens = tokens[0:max_tokens_length]
    return tokens


def log_tokenization_for_conjunction(c: Conjunction, tokenizer, display_label_prefix=None):
    display_label_prefix = display_label_prefix or ""
    for i, p in enumerate(c.prompts):
        if len(c.prompts) > 1:
            this_display_label_prefix = f"{display_label_prefix}(conjunction part {i + 1}, weight={c.weights[i]})"
        else:
            this_display_label_prefix = display_label_prefix
        log_tokenization_for_prompt_object(p, tokenizer, display_label_prefix=this_display_label_prefix)


def log_tokenization_for_prompt_object(p: Union[Blend, FlattenedPrompt], tokenizer, display_label_prefix=None):
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
