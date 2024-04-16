from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import torch
import threading
from typing_extensions import TypeAlias

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    IPAdapterData,
    Range,
    TextConditioningData,
    TextConditioningRegions,
)
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData

ModelForwardCallback: TypeAlias = Union[
    # x, t, conditioning, Optional[cross-attention kwargs]
    Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, Any]]],
        torch.Tensor,
    ],
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
]


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
        config = get_config()
        self.conditioning = None
        self.model = model
        self.model_forward_callback = model_forward_callback
        self.sequential_guidance = config.sequential_guidance

    def do_controlnet_step(
        self,
        control_data,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
        total_step_count: int,
        conditioning_data: TextConditioningData,
    ):
        down_block_res_samples, mid_block_res_sample = None, None

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
                    if conditioning_data.is_sdxl():
                        added_cond_kwargs = {
                            "text_embeds": conditioning_data.cond_text.pooled_embeds,
                            "time_ids": conditioning_data.cond_text.add_time_ids,
                        }
                    encoder_hidden_states = conditioning_data.cond_text.embeds
                    encoder_attention_mask = None
                else:
                    if conditioning_data.is_sdxl():
                        added_cond_kwargs = {
                            "text_embeds": torch.cat(
                                [
                                    # TODO: how to pad? just by zeros? or even truncate?
                                    conditioning_data.uncond_text.pooled_embeds,
                                    conditioning_data.cond_text.pooled_embeds,
                                ],
                                dim=0,
                            ),
                            "time_ids": torch.cat(
                                [
                                    conditioning_data.uncond_text.add_time_ids,
                                    conditioning_data.cond_text.add_time_ids,
                                ],
                                dim=0,
                            ),
                        }
                    (
                        encoder_hidden_states,
                        encoder_attention_mask,
                    ) = self._concat_conditionings_for_batch(
                        conditioning_data.uncond_text.embeds,
                        conditioning_data.cond_text.embeds,
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
        conditioning_data: TextConditioningData,
        ip_adapter_data: Optional[list[IPAdapterData]],
        step_index: int,
        total_step_count: int,
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ):
        if self.sequential_guidance:
            (
                unconditioned_next_x,
                conditioned_next_x,
            ) = self._apply_standard_conditioning_sequentially(
                x=sample,
                sigma=timestep,
                conditioning_data=conditioning_data,
                ip_adapter_data=ip_adapter_data,
                step_index=step_index,
                total_step_count=total_step_count,
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
                ip_adapter_data=ip_adapter_data,
                step_index=step_index,
                total_step_count=total_step_count,
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

        if unconditioning.device != conditioning.device:
            print(f'DEBUG: TID={threading.current_thread().ident}: Unconditioning device = {unconditioning.device}, conditioning device={conditioning.device}')
        return torch.cat([unconditioning, conditioning]), encoder_attention_mask

    # methods below are called from do_diffusion_step and should be considered private to this class.

    def _apply_standard_conditioning(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        conditioning_data: TextConditioningData,
        ip_adapter_data: Optional[list[IPAdapterData]],
        step_index: int,
        total_step_count: int,
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the conditioned and unconditioned UNet forward passes in a single batch for faster inference speed at
        the cost of higher memory usage.
        """
        x_twice = torch.cat([x] * 2)
        sigma_twice = torch.cat([sigma] * 2)

        cross_attention_kwargs = {}
        if ip_adapter_data is not None:
            ip_adapter_conditioning = [ipa.ip_adapter_conditioning for ipa in ip_adapter_data]
            # Note that we 'stack' to produce tensors of shape (batch_size, num_ip_images, seq_len, token_len).
            image_prompt_embeds = [
                torch.stack([ipa_conditioning.uncond_image_prompt_embeds, ipa_conditioning.cond_image_prompt_embeds])
                for ipa_conditioning in ip_adapter_conditioning
            ]
            scales = [ipa.scale_for_step(step_index, total_step_count) for ipa in ip_adapter_data]
            ip_masks = [ipa.mask for ipa in ip_adapter_data]
            regional_ip_data = RegionalIPData(
                image_prompt_embeds=image_prompt_embeds, scales=scales, masks=ip_masks, dtype=x.dtype, device=x.device
            )
            cross_attention_kwargs["regional_ip_data"] = regional_ip_data

        added_cond_kwargs = None
        if conditioning_data.is_sdxl():
            added_cond_kwargs = {
                "text_embeds": torch.cat(
                    [
                        # TODO: how to pad? just by zeros? or even truncate?
                        conditioning_data.uncond_text.pooled_embeds,
                        conditioning_data.cond_text.pooled_embeds,
                    ],
                    dim=0,
                ),
                "time_ids": torch.cat(
                    [
                        conditioning_data.uncond_text.add_time_ids,
                        conditioning_data.cond_text.add_time_ids,
                    ],
                    dim=0,
                ),
            }

        if conditioning_data.cond_regions is not None or conditioning_data.uncond_regions is not None:
            # TODO(ryand): We currently initialize RegionalPromptData for every denoising step. The text conditionings
            # and masks are not changing from step-to-step, so this really only needs to be done once. While this seems
            # painfully inefficient, the time spent is typically negligible compared to the forward inference pass of
            # the UNet. The main reason that this hasn't been moved up to eliminate redundancy is that it is slightly
            # awkward to handle both standard conditioning and sequential conditioning further up the stack.
            regions = []
            for c, r in [
                (conditioning_data.uncond_text, conditioning_data.uncond_regions),
                (conditioning_data.cond_text, conditioning_data.cond_regions),
            ]:
                if r is None:
                    # Create a dummy mask and range for text conditioning that doesn't have region masks.
                    _, _, h, w = x.shape
                    r = TextConditioningRegions(
                        masks=torch.ones((1, 1, h, w), dtype=x.dtype),
                        ranges=[Range(start=0, end=c.embeds.shape[1])],
                    )
                regions.append(r)

            cross_attention_kwargs["regional_prompt_data"] = RegionalPromptData(
                regions=regions, device=x.device, dtype=x.dtype
            )
            cross_attention_kwargs["percent_through"] = step_index / total_step_count

        both_conditionings, encoder_attention_mask = self._concat_conditionings_for_batch(
            conditioning_data.uncond_text.embeds, conditioning_data.cond_text.embeds
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
        conditioning_data: TextConditioningData,
        ip_adapter_data: Optional[list[IPAdapterData]],
        step_index: int,
        total_step_count: int,
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ):
        """Runs the conditioned and unconditioned UNet forward passes sequentially for lower memory usage at the cost of
        slower execution speed.
        """
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

        #####################
        # Unconditioned pass
        #####################

        cross_attention_kwargs = {}

        # Prepare IP-Adapter cross-attention kwargs for the unconditioned pass.
        if ip_adapter_data is not None:
            ip_adapter_conditioning = [ipa.ip_adapter_conditioning for ipa in ip_adapter_data]
            # Note that we 'unsqueeze' to produce tensors of shape (batch_size=1, num_ip_images, seq_len, token_len).
            image_prompt_embeds = [
                torch.unsqueeze(ipa_conditioning.uncond_image_prompt_embeds, dim=0)
                for ipa_conditioning in ip_adapter_conditioning
            ]

            scales = [ipa.scale_for_step(step_index, total_step_count) for ipa in ip_adapter_data]
            ip_masks = [ipa.mask for ipa in ip_adapter_data]
            regional_ip_data = RegionalIPData(
                image_prompt_embeds=image_prompt_embeds, scales=scales, masks=ip_masks, dtype=x.dtype, device=x.device
            )
            cross_attention_kwargs["regional_ip_data"] = regional_ip_data

        # Prepare SDXL conditioning kwargs for the unconditioned pass.
        added_cond_kwargs = None
        if conditioning_data.is_sdxl():
            added_cond_kwargs = {
                "text_embeds": conditioning_data.uncond_text.pooled_embeds,
                "time_ids": conditioning_data.uncond_text.add_time_ids,
            }

        # Prepare prompt regions for the unconditioned pass.
        if conditioning_data.uncond_regions is not None:
            cross_attention_kwargs["regional_prompt_data"] = RegionalPromptData(
                regions=[conditioning_data.uncond_regions], device=x.device, dtype=x.dtype
            )
            cross_attention_kwargs["percent_through"] = step_index / total_step_count

        # Run unconditioned UNet denoising (i.e. negative prompt).
        unconditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            conditioning_data.uncond_text.embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=uncond_down_block,
            mid_block_additional_residual=uncond_mid_block,
            down_intrablock_additional_residuals=uncond_down_intrablock,
            added_cond_kwargs=added_cond_kwargs,
        )

        ###################
        # Conditioned pass
        ###################

        cross_attention_kwargs = {}

        if ip_adapter_data is not None:
            ip_adapter_conditioning = [ipa.ip_adapter_conditioning for ipa in ip_adapter_data]
            # Note that we 'unsqueeze' to produce tensors of shape (batch_size=1, num_ip_images, seq_len, token_len).
            image_prompt_embeds = [
                torch.unsqueeze(ipa_conditioning.cond_image_prompt_embeds, dim=0)
                for ipa_conditioning in ip_adapter_conditioning
            ]

            scales = [ipa.scale_for_step(step_index, total_step_count) for ipa in ip_adapter_data]
            ip_masks = [ipa.mask for ipa in ip_adapter_data]
            regional_ip_data = RegionalIPData(
                image_prompt_embeds=image_prompt_embeds, scales=scales, masks=ip_masks, dtype=x.dtype, device=x.device
            )
            cross_attention_kwargs["regional_ip_data"] = regional_ip_data

        # Prepare SDXL conditioning kwargs for the conditioned pass.
        added_cond_kwargs = None
        if conditioning_data.is_sdxl():
            added_cond_kwargs = {
                "text_embeds": conditioning_data.cond_text.pooled_embeds,
                "time_ids": conditioning_data.cond_text.add_time_ids,
            }

        # Prepare prompt regions for the conditioned pass.
        if conditioning_data.cond_regions is not None:
            cross_attention_kwargs["regional_prompt_data"] = RegionalPromptData(
                regions=[conditioning_data.cond_regions], device=x.device, dtype=x.dtype
            )
            cross_attention_kwargs["percent_through"] = step_index / total_step_count

        # Run conditioned UNet denoising (i.e. positive prompt).
        conditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            conditioning_data.cond_text.embeds,
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
