from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union

import torch
from diffusers import UNet2DConditionModel
from typing_extensions import TypeAlias

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningData,
    ExtraConditioningInfo,
    PostprocessingSettings,
    SDXLConditioningInfo,
)

from .cross_attention_control import (
    Context,
    CrossAttentionType,
    SwapCrossAttnContext,
    get_cross_attention_modules,
    setup_cross_attention_control_attention_processors,
)
from .cross_attention_map_saving import AttentionMapSaver

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
        config = InvokeAIAppConfig.get_config()
        self.conditioning = None
        self.model = model
        self.model_forward_callback = model_forward_callback
        self.cross_attention_control_context = None
        self.sequential_guidance = config.sequential_guidance

    @contextmanager
    def custom_attention_context(
        self,
        unet: UNet2DConditionModel,
        extra_conditioning_info: Optional[ExtraConditioningInfo],
        step_count: int,
    ):
        old_attn_processors = unet.attn_processors

        try:
            self.cross_attention_control_context = Context(
                arguments=extra_conditioning_info.cross_attention_control_args,
                step_count=step_count,
            )
            setup_cross_attention_control_attention_processors(
                unet,
                self.cross_attention_control_context,
            )

            yield None
        finally:
            self.cross_attention_control_context = None
            unet.set_attn_processor(old_attn_processors)
            # TODO resuscitate attention map saving
            # self.remove_attention_map_saving()

    def setup_attention_map_saving(self, saver: AttentionMapSaver):
        def callback(slice, dim, offset, slice_size, key):
            if dim is not None:
                # sliced tokens attention map saving is not implemented
                return
            saver.add_attention_maps(slice, key)

        tokens_cross_attention_modules = get_cross_attention_modules(self.model, CrossAttentionType.TOKENS)
        for identifier, module in tokens_cross_attention_modules:
            key = "down" if identifier.startswith("down") else "up" if identifier.startswith("up") else "mid"
            module.set_attention_slice_calculated_callback(
                lambda slice, dim, offset, slice_size, key=key: callback(slice, dim, offset, slice_size, key)
            )

    def remove_attention_map_saving(self):
        tokens_cross_attention_modules = get_cross_attention_modules(self.model, CrossAttentionType.TOKENS)
        for _, module in tokens_cross_attention_modules:
            module.set_attention_slice_calculated_callback(None)

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
        text_embeddings = conditioning_data.text_embeddings[0]

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
                    if type(text_embeddings) is SDXLConditioningInfo:
                        added_cond_kwargs = {
                            "text_embeds": text_embeddings.pooled_embeds,
                            "time_ids": text_embeddings.add_time_ids,
                        }
                    encoder_hidden_states = text_embeddings.embeds
                    encoder_attention_mask = None
                else:
                    if type(text_embeddings) is SDXLConditioningInfo:
                        added_cond_kwargs = {
                            "text_embeds": torch.cat(
                                [
                                    # TODO: how to pad? just by zeros? or even truncate?
                                    conditioning_data.unconditioned_embeddings.pooled_embeds,
                                    text_embeddings.pooled_embeds,
                                ],
                                dim=0,
                            ),
                            "time_ids": torch.cat(
                                [
                                    conditioning_data.unconditioned_embeddings.add_time_ids,
                                    text_embeddings.add_time_ids,
                                ],
                                dim=0,
                            ),
                        }
                    (
                        encoder_hidden_states,
                        encoder_attention_mask,
                    ) = self._concat_conditionings_for_batch(
                        conditioning_data.unconditioned_embeddings.embeds,
                        text_embeddings.embeds,
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
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )

        return unconditioned_next_x, conditioned_next_x

    def do_latent_postprocessing(
        self,
        postprocessing_settings: PostprocessingSettings,
        latents: torch.Tensor,
        sigma,
        step_index,
        total_step_count,
    ) -> torch.Tensor:
        if postprocessing_settings is not None:
            percent_through = step_index / total_step_count
            latents = self.apply_symmetry(postprocessing_settings, latents, percent_through)
        return latents

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

    # methods below are called from do_diffusion_step and should be considered private to this class.

    def _apply_standard_conditioning(
        self,
        x,
        sigma,
        conditioning_data: ConditioningData,
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ):
        """Runs the conditioned and unconditioned UNet forward passes in a single batch for faster inference speed at
        the cost of higher memory usage.
        """
        x_twice = torch.cat([x] * 2)
        sigma_twice = torch.cat([sigma] * 2)

        assert len(conditioning_data.text_embeddings) == 1
        text_embeddings = conditioning_data.text_embeddings[0]

        cross_attention_kwargs = None
        if conditioning_data.ip_adapter_conditioning is not None:
            # Note that we 'stack' to produce tensors of shape (batch_size, num_ip_images, seq_len, token_len).
            cross_attention_kwargs = {
                "ip_adapter_image_prompt_embeds": [
                    torch.stack(
                        [ipa_conditioning.uncond_image_prompt_embeds, ipa_conditioning.cond_image_prompt_embeds]
                    )
                    for ipa_conditioning in conditioning_data.ip_adapter_conditioning
                ]
            }

        added_cond_kwargs = None
        if type(text_embeddings) is SDXLConditioningInfo:
            added_cond_kwargs = {
                "text_embeds": torch.cat(
                    [
                        # TODO: how to pad? just by zeros? or even truncate?
                        conditioning_data.unconditioned_embeddings.pooled_embeds,
                        text_embeddings.pooled_embeds,
                    ],
                    dim=0,
                ),
                "time_ids": torch.cat(
                    [
                        conditioning_data.unconditioned_embeddings.add_time_ids,
                        text_embeddings.add_time_ids,
                    ],
                    dim=0,
                ),
            }

        both_conditionings, encoder_attention_mask = self._concat_conditionings_for_batch(
            conditioning_data.unconditioned_embeddings.embeds, text_embeddings.embeds
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
        cross_attention_control_types_to_do: list[CrossAttentionType],
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ):
        """Runs the conditioned and unconditioned UNet forward passes sequentially for lower memory usage at the cost of
        slower execution speed.
        """

        assert len(conditioning_data.text_embeddings) == 1
        text_embeddings = conditioning_data.text_embeddings[0]

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
        if conditioning_data.ip_adapter_conditioning is not None:
            # Note that we 'unsqueeze' to produce tensors of shape (batch_size=1, num_ip_images, seq_len, token_len).
            cross_attention_kwargs = {
                "ip_adapter_image_prompt_embeds": [
                    torch.unsqueeze(ipa_conditioning.uncond_image_prompt_embeds, dim=0)
                    for ipa_conditioning in conditioning_data.ip_adapter_conditioning
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
                "text_embeds": conditioning_data.unconditioned_embeddings.pooled_embeds,
                "time_ids": conditioning_data.unconditioned_embeddings.add_time_ids,
            }

        # Run unconditioned UNet denoising (i.e. negative prompt).
        unconditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            conditioning_data.unconditioned_embeddings.embeds,
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
        if conditioning_data.ip_adapter_conditioning is not None:
            # Note that we 'unsqueeze' to produce tensors of shape (batch_size=1, num_ip_images, seq_len, token_len).
            cross_attention_kwargs = {
                "ip_adapter_image_prompt_embeds": [
                    torch.unsqueeze(ipa_conditioning.cond_image_prompt_embeds, dim=0)
                    for ipa_conditioning in conditioning_data.ip_adapter_conditioning
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

    def apply_symmetry(
        self,
        postprocessing_settings: PostprocessingSettings,
        latents: torch.Tensor,
        percent_through: float,
    ) -> torch.Tensor:
        # Reset our last percent through if this is our first step.
        if percent_through == 0.0:
            self.last_percent_through = 0.0

        if postprocessing_settings is None:
            return latents

        # Check for out of bounds
        h_symmetry_time_pct = postprocessing_settings.h_symmetry_time_pct
        if h_symmetry_time_pct is not None and (h_symmetry_time_pct <= 0.0 or h_symmetry_time_pct > 1.0):
            h_symmetry_time_pct = None

        v_symmetry_time_pct = postprocessing_settings.v_symmetry_time_pct
        if v_symmetry_time_pct is not None and (v_symmetry_time_pct <= 0.0 or v_symmetry_time_pct > 1.0):
            v_symmetry_time_pct = None

        dev = latents.device.type

        latents.to(device="cpu")

        if (
            h_symmetry_time_pct is not None
            and self.last_percent_through < h_symmetry_time_pct
            and percent_through >= h_symmetry_time_pct
        ):
            # Horizontal symmetry occurs on the 3rd dimension of the latent
            width = latents.shape[3]
            x_flipped = torch.flip(latents, dims=[3])
            latents = torch.cat(
                [
                    latents[:, :, :, 0 : int(width / 2)],
                    x_flipped[:, :, :, int(width / 2) : int(width)],
                ],
                dim=3,
            )

        if (
            v_symmetry_time_pct is not None
            and self.last_percent_through < v_symmetry_time_pct
            and percent_through >= v_symmetry_time_pct
        ):
            # Vertical symmetry occurs on the 2nd dimension of the latent
            height = latents.shape[2]
            y_flipped = torch.flip(latents, dims=[2])
            latents = torch.cat(
                [
                    latents[:, :, 0 : int(height / 2)],
                    y_flipped[:, :, int(height / 2) : int(height)],
                ],
                dim=2,
            )

        self.last_percent_through = percent_through
        return latents.to(device=dev)

    # todo: make this work
    @classmethod
    def apply_conjunction(cls, x, t, forward_func, uc, c_or_weighted_c_list, global_guidance_scale):
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)  # aka sigmas

        deltas = None
        uncond_latents = None
        weighted_cond_list = (
            c_or_weighted_c_list if isinstance(c_or_weighted_c_list, list) else [(c_or_weighted_c_list, 1)]
        )

        # below is fugly omg
        conditionings = [uc] + [c for c, weight in weighted_cond_list]
        weights = [1] + [weight for c, weight in weighted_cond_list]
        chunk_count = math.ceil(len(conditionings) / 2)
        deltas = None
        for chunk_index in range(chunk_count):
            offset = chunk_index * 2
            chunk_size = min(2, len(conditionings) - offset)

            if chunk_size == 1:
                c_in = conditionings[offset]
                latents_a = forward_func(x_in[:-1], t_in[:-1], c_in)
                latents_b = None
            else:
                c_in = torch.cat(conditionings[offset : offset + 2])
                latents_a, latents_b = forward_func(x_in, t_in, c_in).chunk(2)

            # first chunk is guaranteed to be 2 entries: uncond_latents + first conditioining
            if chunk_index == 0:
                uncond_latents = latents_a
                deltas = latents_b - uncond_latents
            else:
                deltas = torch.cat((deltas, latents_a - uncond_latents))
                if latents_b is not None:
                    deltas = torch.cat((deltas, latents_b - uncond_latents))

        # merge the weighted deltas together into a single merged delta
        per_delta_weights = torch.tensor(weights[1:], dtype=deltas.dtype, device=deltas.device)
        normalize = False
        if normalize:
            per_delta_weights /= torch.sum(per_delta_weights)
        reshaped_weights = per_delta_weights.reshape(per_delta_weights.shape + (1, 1, 1))
        deltas_merged = torch.sum(deltas * reshaped_weights, dim=0, keepdim=True)

        # old_return_value = super().forward(x, sigma, uncond, cond, cond_scale)
        # assert(0 == len(torch.nonzero(old_return_value - (uncond_latents + deltas_merged * cond_scale))))

        return uncond_latents + deltas_merged * global_guidance_scale
