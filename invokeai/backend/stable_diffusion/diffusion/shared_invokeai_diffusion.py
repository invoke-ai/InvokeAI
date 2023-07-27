from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, Dict, Optional, Union, List

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttentionProcessor
from typing_extensions import TypeAlias

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import InvokeAIAppConfig

from .cross_attention_control import (
    Arguments,
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


@dataclass(frozen=True)
class PostprocessingSettings:
    threshold: float
    warmup: float
    h_symmetry_time_pct: Optional[float]
    v_symmetry_time_pct: Optional[float]


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

    @dataclass
    class ExtraConditioningInfo:
        tokens_count_including_eos_bos: int
        cross_attention_control_args: Optional[Arguments] = None

        @property
        def wants_cross_attention_control(self):
            return self.cross_attention_control_args is not None

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

    @classmethod
    @contextmanager
    def custom_attention_context(
        cls,
        unet: UNet2DConditionModel,  # note: also may futz with the text encoder depending on requested LoRAs
        extra_conditioning_info: Optional[ExtraConditioningInfo],
        step_count: int,
    ):
        old_attn_processors = None
        if extra_conditioning_info and (extra_conditioning_info.wants_cross_attention_control):
            old_attn_processors = unet.attn_processors
            # Load lora conditions into the model
            if extra_conditioning_info.wants_cross_attention_control:
                cross_attention_control_context = Context(
                    arguments=extra_conditioning_info.cross_attention_control_args,
                    step_count=step_count,
                )
                setup_cross_attention_control_attention_processors(
                    unet,
                    cross_attention_control_context,
                )

        try:
            yield None
        finally:
            if old_attn_processors is not None:
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

    def do_diffusion_step(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        unconditioning: Union[torch.Tensor, dict],
        conditioning: Union[torch.Tensor, dict],
        # unconditional_guidance_scale: float,
        unconditional_guidance_scale: Union[float, List[float]],
        step_index: Optional[int] = None,
        total_step_count: Optional[int] = None,
        **kwargs,
    ):
        """
        :param x: current latents
        :param sigma: aka t, passed to the internal model to control how much denoising will occur
        :param unconditioning: embeddings for unconditioned output. for hybrid conditioning this is a dict of tensors [B x 77 x 768], otherwise a single tensor [B x 77 x 768]
        :param conditioning: embeddings for conditioned output. for hybrid conditioning this is a dict of tensors [B x 77 x 768], otherwise a single tensor [B x 77 x 768]
        :param unconditional_guidance_scale: aka CFG scale, controls how much effect the conditioning tensor has
        :param step_index: counts upwards from 0 to (step_count-1) (as passed to setup_cross_attention_control, if using). May be called multiple times for a single step, therefore do not assume that its value will monotically increase. If None, will be estimated by comparing sigma against self.model.sigmas .
        :return: the new latents after applying the model to x using unscaled unconditioning and CFG-scaled conditioning.
        """

        if isinstance(unconditional_guidance_scale, list):
            guidance_scale = unconditional_guidance_scale[step_index]
        else:
            guidance_scale = unconditional_guidance_scale

        cross_attention_control_types_to_do = []
        context: Context = self.cross_attention_control_context
        if self.cross_attention_control_context is not None:
            percent_through = step_index / total_step_count
            cross_attention_control_types_to_do = context.get_active_cross_attention_control_types_for_step(
                percent_through
            )

        wants_cross_attention_control = len(cross_attention_control_types_to_do) > 0
        wants_hybrid_conditioning = isinstance(conditioning, dict)

        if wants_hybrid_conditioning:
            unconditioned_next_x, conditioned_next_x = self._apply_hybrid_conditioning(
                x,
                sigma,
                unconditioning,
                conditioning,
                **kwargs,
            )
        elif wants_cross_attention_control:
            (
                unconditioned_next_x,
                conditioned_next_x,
            ) = self._apply_cross_attention_controlled_conditioning(
                x,
                sigma,
                unconditioning,
                conditioning,
                cross_attention_control_types_to_do,
                **kwargs,
            )
        elif self.sequential_guidance:
            (
                unconditioned_next_x,
                conditioned_next_x,
            ) = self._apply_standard_conditioning_sequentially(
                x,
                sigma,
                unconditioning,
                conditioning,
                **kwargs,
            )

        else:
            (
                unconditioned_next_x,
                conditioned_next_x,
            ) = self._apply_standard_conditioning(
                x,
                sigma,
                unconditioning,
                conditioning,
                **kwargs,
            )

        combined_next_x = self._combine(
            # unconditioned_next_x, conditioned_next_x, unconditional_guidance_scale
            unconditioned_next_x,
            conditioned_next_x,
            guidance_scale,
        )

        return combined_next_x

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
            latents = self.apply_threshold(postprocessing_settings, latents, percent_through)
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

    def _apply_standard_conditioning(self, x, sigma, unconditioning, conditioning, **kwargs):
        # fast batched path
        x_twice = torch.cat([x] * 2)
        sigma_twice = torch.cat([sigma] * 2)

        both_conditionings, encoder_attention_mask = self._concat_conditionings_for_batch(unconditioning, conditioning)
        both_results = self.model_forward_callback(
            x_twice,
            sigma_twice,
            both_conditionings,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )
        unconditioned_next_x, conditioned_next_x = both_results.chunk(2)
        return unconditioned_next_x, conditioned_next_x

    def _apply_standard_conditioning_sequentially(
        self,
        x: torch.Tensor,
        sigma,
        unconditioning: torch.Tensor,
        conditioning: torch.Tensor,
        **kwargs,
    ):
        # low-memory sequential path
        uncond_down_block, cond_down_block = None, None
        down_block_additional_residuals = kwargs.pop("down_block_additional_residuals", None)
        if down_block_additional_residuals is not None:
            uncond_down_block, cond_down_block = [], []
            for down_block in down_block_additional_residuals:
                _uncond_down, _cond_down = down_block.chunk(2)
                uncond_down_block.append(_uncond_down)
                cond_down_block.append(_cond_down)

        uncond_mid_block, cond_mid_block = None, None
        mid_block_additional_residual = kwargs.pop("mid_block_additional_residual", None)
        if mid_block_additional_residual is not None:
            uncond_mid_block, cond_mid_block = mid_block_additional_residual.chunk(2)

        unconditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            unconditioning,
            down_block_additional_residuals=uncond_down_block,
            mid_block_additional_residual=uncond_mid_block,
            **kwargs,
        )
        conditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            conditioning,
            down_block_additional_residuals=cond_down_block,
            mid_block_additional_residual=cond_mid_block,
            **kwargs,
        )
        return unconditioned_next_x, conditioned_next_x

    # TODO: looks unused
    def _apply_hybrid_conditioning(self, x, sigma, unconditioning, conditioning, **kwargs):
        assert isinstance(conditioning, dict)
        assert isinstance(unconditioning, dict)
        x_twice = torch.cat([x] * 2)
        sigma_twice = torch.cat([sigma] * 2)
        both_conditionings = dict()
        for k in conditioning:
            if isinstance(conditioning[k], list):
                both_conditionings[k] = [
                    torch.cat([unconditioning[k][i], conditioning[k][i]]) for i in range(len(conditioning[k]))
                ]
            else:
                both_conditionings[k] = torch.cat([unconditioning[k], conditioning[k]])
        unconditioned_next_x, conditioned_next_x = self.model_forward_callback(
            x_twice,
            sigma_twice,
            both_conditionings,
            **kwargs,
        ).chunk(2)
        return unconditioned_next_x, conditioned_next_x

    def _apply_cross_attention_controlled_conditioning(
        self,
        x: torch.Tensor,
        sigma,
        unconditioning,
        conditioning,
        cross_attention_control_types_to_do,
        **kwargs,
    ):
        context: Context = self.cross_attention_control_context

        uncond_down_block, cond_down_block = None, None
        down_block_additional_residuals = kwargs.pop("down_block_additional_residuals", None)
        if down_block_additional_residuals is not None:
            uncond_down_block, cond_down_block = [], []
            for down_block in down_block_additional_residuals:
                _uncond_down, _cond_down = down_block.chunk(2)
                uncond_down_block.append(_uncond_down)
                cond_down_block.append(_cond_down)

        uncond_mid_block, cond_mid_block = None, None
        mid_block_additional_residual = kwargs.pop("mid_block_additional_residual", None)
        if mid_block_additional_residual is not None:
            uncond_mid_block, cond_mid_block = mid_block_additional_residual.chunk(2)

        cross_attn_processor_context = SwapCrossAttnContext(
            modified_text_embeddings=context.arguments.edited_conditioning,
            index_map=context.cross_attention_index_map,
            mask=context.cross_attention_mask,
            cross_attention_types_to_do=[],
        )
        # no cross attention for unconditioning (negative prompt)
        unconditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            unconditioning,
            {"swap_cross_attn_context": cross_attn_processor_context},
            down_block_additional_residuals=uncond_down_block,
            mid_block_additional_residual=uncond_mid_block,
            **kwargs,
        )

        # do requested cross attention types for conditioning (positive prompt)
        cross_attn_processor_context.cross_attention_types_to_do = cross_attention_control_types_to_do
        conditioned_next_x = self.model_forward_callback(
            x,
            sigma,
            conditioning,
            {"swap_cross_attn_context": cross_attn_processor_context},
            down_block_additional_residuals=cond_down_block,
            mid_block_additional_residual=cond_mid_block,
            **kwargs,
        )
        return unconditioned_next_x, conditioned_next_x

    def _combine(self, unconditioned_next_x, conditioned_next_x, guidance_scale):
        # to scale how much effect conditioning has, calculate the changes it does and then scale that
        scaled_delta = (conditioned_next_x - unconditioned_next_x) * guidance_scale
        combined_next_x = unconditioned_next_x + scaled_delta
        return combined_next_x

    def apply_threshold(
        self,
        postprocessing_settings: PostprocessingSettings,
        latents: torch.Tensor,
        percent_through: float,
    ) -> torch.Tensor:
        if postprocessing_settings.threshold is None or postprocessing_settings.threshold == 0.0:
            return latents

        threshold = postprocessing_settings.threshold
        warmup = postprocessing_settings.warmup

        if percent_through < warmup:
            current_threshold = threshold + threshold * 5 * (1 - (percent_through / warmup))
        else:
            current_threshold = threshold

        if current_threshold <= 0:
            return latents

        maxval = latents.max().item()
        minval = latents.min().item()

        scale = 0.7  # default value from #395

        if self.debug_thresholding:
            std, mean = [i.item() for i in torch.std_mean(latents)]
            outside = torch.count_nonzero((latents < -current_threshold) | (latents > current_threshold))
            logger.info(f"Threshold: %={percent_through} threshold={current_threshold:.3f} (of {threshold:.3f})")
            logger.debug(f"min, mean, max = {minval:.3f}, {mean:.3f}, {maxval:.3f}\tstd={std}")
            logger.debug(f"{outside / latents.numel() * 100:.2f}% values outside threshold")

        if maxval < current_threshold and minval > -current_threshold:
            return latents

        num_altered = 0

        # MPS torch.rand_like is fine because torch.rand_like is wrapped in generate.py!

        if maxval > current_threshold:
            latents = torch.clone(latents)
            maxval = np.clip(maxval * scale, 1, current_threshold)
            num_altered += torch.count_nonzero(latents > maxval)
            latents[latents > maxval] = torch.rand_like(latents[latents > maxval]) * maxval

        if minval < -current_threshold:
            latents = torch.clone(latents)
            minval = np.clip(minval * scale, -current_threshold, -1)
            num_altered += torch.count_nonzero(latents < minval)
            latents[latents < minval] = torch.rand_like(latents[latents < minval]) * minval

        if self.debug_thresholding:
            logger.debug(f"min,     , max = {minval:.3f},        , {maxval:.3f}\t(scaled by {scale})")
            logger.debug(f"{num_altered / latents.numel() * 100:.2f}% values altered")

        return latents

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
            h_symmetry_time_pct != None
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
            v_symmetry_time_pct != None
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

    def estimate_percent_through(self, step_index, sigma):
        if step_index is not None and self.cross_attention_control_context is not None:
            # percent_through will never reach 1.0 (but this is intended)
            return float(step_index) / float(self.cross_attention_control_context.step_count)
        # find the best possible index of the current sigma in the sigma sequence
        smaller_sigmas = torch.nonzero(self.model.sigmas <= sigma)
        sigma_index = smaller_sigmas[-1].item() if smaller_sigmas.shape[0] > 0 else 0
        # flip because sigmas[0] is for the fully denoised image
        # percent_through must be <1
        return 1.0 - float(sigma_index + 1) / float(self.model.sigmas.shape[0])
        # print('estimated percent_through', percent_through, 'from sigma', sigma.item())

    # todo: make this work
    @classmethod
    def apply_conjunction(cls, x, t, forward_func, uc, c_or_weighted_c_list, global_guidance_scale):
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)  # aka sigmas

        deltas = None
        uncond_latents = None
        weighted_cond_list = c_or_weighted_c_list if type(c_or_weighted_c_list) is list else [(c_or_weighted_c_list, 1)]

        # below is fugly omg
        conditionings = [uc] + [c for c, weight in weighted_cond_list]
        weights = [1] + [weight for c, weight in weighted_cond_list]
        chunk_count = ceil(len(conditionings) / 2)
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
