from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union

import torch
import torchvision
from diffusers import UNet2DConditionModel
from typing_extensions import TypeAlias

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    ConditioningData,
    ExtraConditioningInfo,
    IPAdapterConditioningInfo,
    SDXLConditioningInfo,
)
from invokeai.backend.stable_diffusion.diffusion.regional_prompt_attention import Range, RegionalPromptData

from .cross_attention_control import (
    CrossAttentionType,
    CrossAttnControlContext,
    SwapCrossAttnContext,
    setup_cross_attention_control_attention_processors,
)

ModelForwardCallback: TypeAlias = Union[
    # x, t, conditioning, Optional[cross-attention kwargs]
    Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, Any]]],
        torch.Tensor,
    ],
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
]


class RegionalTextConditioningInfo:
    def __init__(
        self,
        text_conditioning: Union[BasicConditioningInfo, SDXLConditioningInfo],
        masks: Optional[torch.Tensor] = None,
        embedding_ranges: Optional[list[Range]] = None,
    ):
        """Initialize a RegionalTextConditioningInfo.

        Args:
            text_conditioning (Union[BasicConditioningInfo, SDXLConditioningInfo]): The text conditioning embeddings
                after concatenating the embeddings for all regions.
            masks (Optional[torch.Tensor], optional): Shape: (1, num_regions, h, w).
            embedding_ranges (Optional[list[Range]], optional): The embedding range for each region.
        """
        self.text_conditioning = text_conditioning
        self.masks = masks
        self.embedding_ranges = embedding_ranges

        assert (self.masks is None) == (self.embedding_ranges is None)
        if self.masks is not None:
            assert self.masks.shape[1] == len(self.embedding_ranges)

    def has_region_masks(self):
        if self.masks is None:
            return False
        return any(mask is not None for mask in self.masks)

    def is_sdxl(self):
        return isinstance(self.text_conditioning, SDXLConditioningInfo)

    @classmethod
    def _preprocess_regional_prompt_mask(
        cls, mask: Optional[torch.Tensor], target_height: int, target_width: int
    ) -> torch.Tensor:
        """Preprocess a regional prompt mask to match the target height and width.

        If mask is None, returns a mask of all ones with the target height and width.
        If mask is not None, resizes the mask to the target height and width using nearest neighbor interpolation.

        Returns:
            torch.Tensor: The processed mask. dtype: torch.bool, shape: (1, 1, target_height, target_width).
        """
        if mask is None:
            return torch.ones((1, 1, target_height, target_width), dtype=torch.bool)

        tf = torchvision.transforms.Resize(
            (target_height, target_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        mask = mask.unsqueeze(0)  # Shape: (1, h, w) -> (1, 1, h, w)
        mask = tf(mask)

        return mask

    @classmethod
    def from_text_conditioning_and_masks(
        cls,
        text_conditionings: list[Union[BasicConditioningInfo, SDXLConditioningInfo]],
        masks: Optional[list[Optional[torch.Tensor]]],
        latent_height: int,
        latent_width: int,
    ):
        if masks is None:
            masks = [None] * len(text_conditionings)
        assert len(text_conditionings) == len(masks)

        is_sdxl = type(text_conditionings[0]) is SDXLConditioningInfo

        all_masks_are_none = all(mask is None for mask in masks)

        text_embedding = []
        pooled_embedding = None
        add_time_ids = None
        processed_masks = []
        cur_text_embedding_len = 0
        embedding_ranges: list[Range] = []

        for text_embedding_info, mask in zip(text_conditionings, masks, strict=True):
            # HACK(ryand): Figure out the intended relationship between CAC and other conditioning features.
            assert (
                text_embedding_info.extra_conditioning is None
                or not text_embedding_info.extra_conditioning.wants_cross_attention_control
            )

            if is_sdxl:
                # HACK(ryand): We just use the the first SDXLConditioningInfo's pooled_embeds and add_time_ids. This is
                # fundamentally an interface issue, as the SDXL Compel nodes are not designed to be used in the way that
                # we use them for regional prompting. Ideally, the DenoiseLatents invocation should accept a single
                # pooled_embeds tensor and a list of standard text embeds with region masks. This change would be a
                # pretty major breaking change to a popular node, so for now we use this hack.
                #
                # An improvement could be to use the pooled embeds from the prompt with the largest region, as this is
                # most likely to be a global prompt.
                if pooled_embedding is None:
                    pooled_embedding = text_embedding_info.pooled_embeds
                if add_time_ids is None:
                    add_time_ids = text_embedding_info.add_time_ids

            text_embedding.append(text_embedding_info.embeds)
            embedding_ranges.append(
                Range(start=cur_text_embedding_len, end=cur_text_embedding_len + text_embedding_info.embeds.shape[1])
            )
            cur_text_embedding_len += text_embedding_info.embeds.shape[1]

            if not all_masks_are_none:
                processed_masks.append(cls._preprocess_regional_prompt_mask(mask, latent_height, latent_width))

        text_embedding = torch.cat(text_embedding, dim=1)
        assert len(text_embedding.shape) == 3  # batch_size, seq_len, token_len

        if not all_masks_are_none:
            processed_masks = torch.cat(processed_masks, dim=1)
        else:
            processed_masks = None
            embedding_ranges = None

        if is_sdxl:
            return cls(
                text_conditioning=SDXLConditioningInfo(
                    embeds=text_embedding,
                    extra_conditioning=None,
                    pooled_embeds=pooled_embedding,
                    add_time_ids=add_time_ids,
                ),
                masks=processed_masks,
                embedding_ranges=embedding_ranges,
            )
        return cls(
            text_conditioning=BasicConditioningInfo(
                embeds=text_embedding,
                extra_conditioning=None,
            ),
            masks=processed_masks,
            embedding_ranges=embedding_ranges,
        )


class InvokeAIDiffuserComponent:
    """
    The aim of this component is to provide a single place for code that can be applied identically to
    all InvokeAI diffusion procedures.

    At the moment it includes the following features:
    * Cross attention control ("prompt2prompt")
    * Hybrid conditioning (used for inpainting)
    """

    debug_thresholding = False
    sequential_guidance = False

    def __init__(
        self,
        model,
        model_forward_callback: ModelForwardCallback,
    ):
        """
        :param model: the unet model to pass through to cross attention control
        :param model_forward_callback: a lambda with arguments (x, sigma, conditioning_to_apply). will be called repeatedly. most likely, this should simply call model.forward(x, sigma, conditioning)
        """
        config = InvokeAIAppConfig.get_config()
        self.model = model
        self.model_forward_callback = model_forward_callback
        self.cross_attention_control_context = None
        self.sequential_guidance = config.sequential_guidance

    @contextmanager
    def custom_attention_context(
        self,
        unet: UNet2DConditionModel,
        extra_conditioning_info: Optional[ExtraConditioningInfo],
    ):
        old_attn_processors = unet.attn_processors

        try:
            self.cross_attention_control_context = CrossAttnControlContext(
                arguments=extra_conditioning_info.cross_attention_control_args,
            )
            setup_cross_attention_control_attention_processors(
                unet,
                self.cross_attention_control_context,
            )

            yield None
        finally:
            self.cross_attention_control_context = None
            unet.set_attn_processor(old_attn_processors)

    def do_controlnet_step(
        self,
        control_data,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
        total_step_count: int,
        conditioning_data: ConditioningData,
    ):
        down_block_res_samples, mid_block_res_sample = None, None
        # HACK(ryan): Currently, we just take the first text embedding if there's more than one. We should probably
        # concatenate all of the embeddings for the ControlNet, but not apply embedding masks.
        uncond_text_embeddings = conditioning_data.uncond_text_embeddings[0]
        cond_text_embeddings = conditioning_data.cond_text_embeddings[0]

        # control_data should be type List[ControlNetData]
        # this loop covers both ControlNet (one ControlNetData in list)
        #      and MultiControlNet (multiple ControlNetData in list)
        for _i, control_datum in enumerate(control_data):
            control_mode = control_datum.control_mode
            # soft_injection and cfg_injection are the two ControlNet control_mode booleans
            #     that are combined at higher level to make control_mode enum
            #  soft_injection determines whether to do per-layer re-weighting adjustment (if True)
            #     or default weighting (if False)
            soft_injection = control_mode == "more_prompt" or control_mode == "more_control"
            #  cfg_injection = determines whether to apply ControlNet to only the conditional (if True)
            #      or the default both conditional and unconditional (if False)
            cfg_injection = control_mode == "more_control" or control_mode == "unbalanced"

            first_control_step = math.floor(control_datum.begin_step_percent * total_step_count)
            last_control_step = math.ceil(control_datum.end_step_percent * total_step_count)
            # only apply controlnet if current step is within the controlnet's begin/end step range
            if step_index >= first_control_step and step_index <= last_control_step:
                if cfg_injection:
                    sample_model_input = sample
                else:
                    # expand the latents input to control model if doing classifier free guidance
                    #    (which I think for now is always true, there is conditional elsewhere that stops execution if
                    #     classifier_free_guidance is <= 1.0 ?)
                    sample_model_input = torch.cat([sample] * 2)

                added_cond_kwargs = None

                if cfg_injection:  # only applying ControlNet to conditional instead of in unconditioned
                    if type(cond_text_embeddings) is SDXLConditioningInfo:
                        added_cond_kwargs = {
                            "text_embeds": cond_text_embeddings.pooled_embeds,
                            "time_ids": cond_text_embeddings.add_time_ids,
                        }
                    encoder_hidden_states = cond_text_embeddings.embeds
                    encoder_attention_mask = None
                else:
                    if type(cond_text_embeddings) is SDXLConditioningInfo:
                        added_cond_kwargs = {
                            "text_embeds": torch.cat(
                                [uncond_text_embeddings.pooled_embeds, cond_text_embeddings.pooled_embeds], dim=0
                            ),
                            "time_ids": torch.cat(
                                [uncond_text_embeddings.add_time_ids, cond_text_embeddings.add_time_ids], dim=0
                            ),
                        }
                    (encoder_hidden_states, encoder_attention_mask) = self._concat_conditionings_for_batch(
                        uncond_text_embeddings.embeds, cond_text_embeddings.embeds
                    )
                if isinstance(control_datum.weight, list):
                    # if controlnet has multiple weights, use the weight for the current step
                    controlnet_weight = control_datum.weight[step_index]
                else:
                    # if controlnet has a single weight, use it for all steps
                    controlnet_weight = control_datum.weight

                # controlnet(s) inference
                down_samples, mid_sample = control_datum.model(
                    sample=sample_model_input,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_datum.image_tensor,
                    conditioning_scale=controlnet_weight,  # controlnet specific, NOT the guidance scale
                    encoder_attention_mask=encoder_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                    guess_mode=soft_injection,  # this is still called guess_mode in diffusers ControlNetModel
                    return_dict=False,
                )
                if cfg_injection:
                    # Inferred ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    #    prepend zeros for unconditional batch
                    down_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_samples]
                    mid_sample = torch.cat([torch.zeros_like(mid_sample), mid_sample])

                if down_block_res_samples is None and mid_block_res_sample is None:
                    down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
                else:
                    # add controlnet outputs together if have multiple controlnets
                    down_block_res_samples = [
                        samples_prev + samples_curr
                        for samples_prev, samples_curr in zip(down_block_res_samples, down_samples, strict=True)
                    ]
                    mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample

    def do_unet_step(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_data: ConditioningData,
        ip_adapter_conditioning: Optional[list[IPAdapterConditioningInfo]],
        step_index: int,
        total_step_count: int,
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ):
        cross_attention_control_types_to_do = []
        if self.cross_attention_control_context is not None:
            percent_through = step_index / total_step_count
            cross_attention_control_types_to_do = (
                self.cross_attention_control_context.get_active_cross_attention_control_types_for_step(percent_through)
            )
        wants_cross_attention_control = len(cross_attention_control_types_to_do) > 0

        if wants_cross_attention_control or self.sequential_guidance:
            # If wants_cross_attention_control is True, we force the sequential mode to be used, because cross-attention
            # control is currently only supported in sequential mode.
            (
                unconditioned_next_x,
                conditioned_next_x,
            ) = self._apply_standard_conditioning_sequentially(
                x=sample,
                sigma=timestep,
                conditioning_data=conditioning_data,
                ip_adapter_conditioning=ip_adapter_conditioning,
                cross_attention_control_types_to_do=cross_attention_control_types_to_do,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )
        else:
            (
                unconditioned_next_x,
                conditioned_next_x,
            ) = self._apply_standard_conditioning(
                x=sample,
                sigma=timestep,
                conditioning_data=conditioning_data,
                ip_adapter_conditioning=ip_adapter_conditioning,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )

        return unconditioned_next_x, conditioned_next_x

    def _concat_conditionings_for_batch(self, unconditioning, conditioning):
        def _pad_conditioning(cond, target_len, encoder_attention_mask):
            conditioning_attention_mask = torch.ones(
                (cond.shape[0], cond.shape[1]), device=cond.device, dtype=cond.dtype
            )

            if cond.shape[1] < max_len:
                conditioning_attention_mask = torch.cat(
                    [
                        conditioning_attention_mask,
                        torch.zeros((cond.shape[0], max_len - cond.shape[1]), device=cond.device, dtype=cond.dtype),
                    ],
                    dim=1,
                )

                cond = torch.cat(
                    [
                        cond,
                        torch.zeros(
                            (cond.shape[0], max_len - cond.shape[1], cond.shape[2]),
                            device=cond.device,
                            dtype=cond.dtype,
                        ),
                    ],
                    dim=1,
                )

            if encoder_attention_mask is None:
                encoder_attention_mask = conditioning_attention_mask
            else:
                encoder_attention_mask = torch.cat(
                    [
                        encoder_attention_mask,
                        conditioning_attention_mask,
                    ]
                )

            return cond, encoder_attention_mask

        encoder_attention_mask = None
        if unconditioning.shape[1] != conditioning.shape[1]:
            max_len = max(unconditioning.shape[1], conditioning.shape[1])
            unconditioning, encoder_attention_mask = _pad_conditioning(unconditioning, max_len, encoder_attention_mask)
            conditioning, encoder_attention_mask = _pad_conditioning(conditioning, max_len, encoder_attention_mask)

        return torch.cat([unconditioning, conditioning]), encoder_attention_mask

    def _apply_standard_conditioning(
        self,
        x,
        sigma,
        conditioning_data: ConditioningData,
        ip_adapter_conditioning: Optional[list[IPAdapterConditioningInfo]],
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ):
        """Runs the conditioned and unconditioned UNet forward passes in a single batch for faster inference speed at
        the cost of higher memory usage.
        """
        x_twice = torch.cat([x] * 2)
        sigma_twice = torch.cat([sigma] * 2)

        # TODO(ryand): We currently call from_text_conditioning_and_masks(...) and from_masks_and_ranges(...) for every
        # denoising step. The text conditionings and masks are not changing from step-to-step, so this really only needs
        # to be done once. While this seems painfully inefficient, the time spent is typically negligible compared to
        # the forward inference pass of the UNet. The main reason that this hasn't been moved up to eliminate redundancy
        # is that it is slightly awkward to handle both standard conditioning and sequential conditioning further up the
        # stack.
        cross_attention_kwargs = None
        _, _, h, w = x.shape
        cond_text = RegionalTextConditioningInfo.from_text_conditioning_and_masks(
            text_conditionings=conditioning_data.cond_text_embeddings,
            masks=conditioning_data.cond_text_embedding_masks,
            latent_height=h,
            latent_width=w,
        )
        uncond_text = RegionalTextConditioningInfo.from_text_conditioning_and_masks(
            text_conditionings=conditioning_data.uncond_text_embeddings,
            masks=conditioning_data.uncond_text_embedding_masks,
            latent_height=h,
            latent_width=w,
        )

        if cond_text.has_region_masks() or uncond_text.has_region_masks():
            masks = []
            embedding_ranges = []
            for c in [uncond_text, cond_text]:
                if c.has_region_masks():
                    masks.append(c.masks)
                    embedding_ranges.append(c.embedding_ranges)
                else:
                    # Create a dummy mask and range for text conditioning that doesn't have region masks.
                    masks.append(torch.ones((1, 1, h, w), dtype=torch.bool))
                    embedding_ranges.append([Range(start=0, end=c.text_conditioning.embeds.shape[1])])

            # The key_seq_len will be the maximum sequence length of all the conditioning embeddings. All other
            # embeddings will be padded to match this length.
            key_seq_len = 0
            for c in [uncond_text, cond_text]:
                _, seq_len, _ = c.text_conditioning.embeds.shape
                if seq_len > key_seq_len:
                    key_seq_len = seq_len

            cross_attention_kwargs = {
                "regional_prompt_data": RegionalPromptData.from_masks_and_ranges(
                    masks=masks, embedding_ranges=embedding_ranges, key_seq_len=key_seq_len
                )
            }

        # TODO(ryand): Figure out interactions between regional prompting and IP-Adapter conditioning.
        if ip_adapter_conditioning is not None:
            # Note that we 'stack' to produce tensors of shape (batch_size, num_ip_images, seq_len, token_len).
            cross_attention_kwargs = {
                "ip_adapter_image_prompt_embeds": [
                    torch.stack(
                        [ipa_conditioning.uncond_image_prompt_embeds, ipa_conditioning.cond_image_prompt_embeds]
                    )
                    for ipa_conditioning in ip_adapter_conditioning
                ]
            }

        added_cond_kwargs = None
        if cond_text.is_sdxl():
            added_cond_kwargs = {
                "text_embeds": torch.cat(
                    [uncond_text.text_conditioning.pooled_embeds, cond_text.text_conditioning.pooled_embeds], dim=0
                ),
                "time_ids": torch.cat(
                    [uncond_text.text_conditioning.add_time_ids, cond_text.text_conditioning.add_time_ids], dim=0
                ),
            }

        both_conditionings, encoder_attention_mask = self._concat_conditionings_for_batch(
            uncond_text.text_conditioning.embeds, cond_text.text_conditioning.embeds
        )
        both_results = self.model_forward_callback(
            x_twice,
            sigma_twice,
            both_conditionings,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            added_cond_kwargs=added_cond_kwargs,
        )
        unconditioned_next_x, conditioned_next_x = both_results.chunk(2)
        return unconditioned_next_x, conditioned_next_x

    def _apply_standard_conditioning_sequentially(
        self,
        x: torch.Tensor,
        sigma,
        conditioning_data: ConditioningData,
        ip_adapter_conditioning: Optional[list[IPAdapterConditioningInfo]],
        cross_attention_control_types_to_do: list[CrossAttentionType],
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ):
        """Runs the conditioned and unconditioned UNet forward passes sequentially for lower memory usage at the cost of
        slower execution speed.
        """

        assert len(conditioning_data.cond_text_embeddings) == 1
        text_embeddings = conditioning_data.cond_text_embeddings[0]

        # Since we are running the conditioned and unconditioned passes sequentially, we need to split the ControlNet
        # and T2I-Adapter residuals into two chunks.
        uncond_down_block, cond_down_block = None, None
        if down_block_additional_residuals is not None:
            uncond_down_block, cond_down_block = [], []
            for down_block in down_block_additional_residuals:
                _uncond_down, _cond_down = down_block.chunk(2)
                uncond_down_block.append(_uncond_down)
                cond_down_block.append(_cond_down)

        uncond_down_intrablock, cond_down_intrablock = None, None
        if down_intrablock_additional_residuals is not None:
            uncond_down_intrablock, cond_down_intrablock = [], []
            for down_intrablock in down_intrablock_additional_residuals:
                _uncond_down, _cond_down = down_intrablock.chunk(2)
                uncond_down_intrablock.append(_uncond_down)
                cond_down_intrablock.append(_cond_down)

        uncond_mid_block, cond_mid_block = None, None
        if mid_block_additional_residual is not None:
            uncond_mid_block, cond_mid_block = mid_block_additional_residual.chunk(2)

        # If cross-attention control is enabled, prepare the SwapCrossAttnContext.
        cross_attn_processor_context = None
        if self.cross_attention_control_context is not None:
            # Note that the SwapCrossAttnContext is initialized with an empty list of cross_attention_types_to_do.
            # This list is empty because cross-attention control is not applied in the unconditioned pass. This field
            # will be populated before the conditioned pass.
            cross_attn_processor_context = SwapCrossAttnContext(
                modified_text_embeddings=self.cross_attention_control_context.arguments.edited_conditioning,
                index_map=self.cross_attention_control_context.cross_attention_index_map,
                mask=self.cross_attention_control_context.cross_attention_mask,
                cross_attention_types_to_do=[],
            )

        #####################
        # Unconditioned pass
        #####################

        cross_attention_kwargs = None

        # Prepare IP-Adapter cross-attention kwargs for the unconditioned pass.
        if ip_adapter_conditioning is not None:
            # Note that we 'unsqueeze' to produce tensors of shape (batch_size=1, num_ip_images, seq_len, token_len).
            cross_attention_kwargs = {
                "ip_adapter_image_prompt_embeds": [
                    torch.unsqueeze(ipa_conditioning.uncond_image_prompt_embeds, dim=0)
                    for ipa_conditioning in ip_adapter_conditioning
                ]
            }

        # Prepare cross-attention control kwargs for the unconditioned pass.
        if cross_attn_processor_context is not None:
            cross_attention_kwargs = {"swap_cross_attn_context": cross_attn_processor_context}

        # Prepare SDXL conditioning kwargs for the unconditioned pass.
        added_cond_kwargs = None
        is_sdxl = type(text_embeddings) is SDXLConditioningInfo
        if is_sdxl:
            added_cond_kwargs = {
                "text_embeds": conditioning_data.uncond_text_embeddings.pooled_embeds,
                "time_ids": conditioning_data.uncond_text_embeddings.add_time_ids,
            }

        # Run unconditioned UNet denoising (i.e. negative prompt).
        unconditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            conditioning_data.uncond_text_embeddings.embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=uncond_down_block,
            mid_block_additional_residual=uncond_mid_block,
            down_intrablock_additional_residuals=uncond_down_intrablock,
            added_cond_kwargs=added_cond_kwargs,
        )

        ###################
        # Conditioned pass
        ###################

        cross_attention_kwargs = None

        # Prepare IP-Adapter cross-attention kwargs for the conditioned pass.
        if ip_adapter_conditioning is not None:
            # Note that we 'unsqueeze' to produce tensors of shape (batch_size=1, num_ip_images, seq_len, token_len).
            cross_attention_kwargs = {
                "ip_adapter_image_prompt_embeds": [
                    torch.unsqueeze(ipa_conditioning.cond_image_prompt_embeds, dim=0)
                    for ipa_conditioning in ip_adapter_conditioning
                ]
            }

        # Prepare cross-attention control kwargs for the conditioned pass.
        if cross_attn_processor_context is not None:
            cross_attn_processor_context.cross_attention_types_to_do = cross_attention_control_types_to_do
            cross_attention_kwargs = {"swap_cross_attn_context": cross_attn_processor_context}

        # Prepare SDXL conditioning kwargs for the conditioned pass.
        added_cond_kwargs = None
        if is_sdxl:
            added_cond_kwargs = {
                "text_embeds": text_embeddings.pooled_embeds,
                "time_ids": text_embeddings.add_time_ids,
            }

        # Run conditioned UNet denoising (i.e. positive prompt).
        conditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            text_embeddings.embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=cond_down_block,
            mid_block_additional_residual=cond_mid_block,
            down_intrablock_additional_residuals=cond_down_intrablock,
            added_cond_kwargs=added_cond_kwargs,
        )
        return unconditioned_next_x, conditioned_next_x

    def _combine(self, unconditioned_next_x, conditioned_next_x, guidance_scale):
        # to scale how much effect conditioning has, calculate the changes it does and then scale that
        scaled_delta = (conditioned_next_x - unconditioned_next_x) * guidance_scale
        combined_next_x = unconditioned_next_x + scaled_delta
        return combined_next_x
