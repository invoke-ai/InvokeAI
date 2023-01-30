import math
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Callable, Optional, Union, Any, Dict

import numpy as np
import torch

from diffusers.models.cross_attention import AttnProcessor
from ldm.models.diffusion.cross_attention_control import Arguments, \
    restore_default_cross_attention, override_cross_attention, Context, get_cross_attention_modules, \
    CrossAttentionType, SwapCrossAttnContext
from ldm.models.diffusion.cross_attention_map_saving import AttentionMapSaver


@dataclass(frozen=True)
class ThresholdSettings:
    threshold: float
    warmup: float


class InvokeAIDiffuserComponent:
    '''
    The aim of this component is to provide a single place for code that can be applied identically to
    all InvokeAI diffusion procedures.

    At the moment it includes the following features:
    * Cross attention control ("prompt2prompt")
    * Hybrid conditioning (used for inpainting)
    '''
    debug_thresholding = False


    @dataclass
    class ExtraConditioningInfo:

        tokens_count_including_eos_bos: int
        cross_attention_control_args: Optional[Arguments] = None

        @property
        def wants_cross_attention_control(self):
            return self.cross_attention_control_args is not None


    def __init__(self, model, model_forward_callback:
                    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str,Any]]], torch.Tensor],
                 is_running_diffusers: bool=False,
                 ):
        """
        :param model: the unet model to pass through to cross attention control
        :param model_forward_callback: a lambda with arguments (x, sigma, conditioning_to_apply). will be called repeatedly. most likely, this should simply call model.forward(x, sigma, conditioning)
        """
        self.conditioning = None
        self.model = model
        self.is_running_diffusers = is_running_diffusers
        self.model_forward_callback = model_forward_callback
        self.cross_attention_control_context = None

    @contextmanager
    def custom_attention_context(self,
                                 extra_conditioning_info: Optional[ExtraConditioningInfo],
                                 step_count: int):
        do_swap = extra_conditioning_info is not None and extra_conditioning_info.wants_cross_attention_control
        old_attn_processor = None
        if do_swap:
            old_attn_processor = self.override_cross_attention(extra_conditioning_info,
                                                               step_count=step_count)
        try:
            yield None
        finally:
            if old_attn_processor is not None:
                self.restore_default_cross_attention(old_attn_processor)
            # TODO resuscitate attention map saving
            #self.remove_attention_map_saving()

    def override_cross_attention(self, conditioning: ExtraConditioningInfo, step_count: int) -> Dict[str, AttnProcessor]:
        """
        setup cross attention .swap control. for diffusers this replaces the attention processor, so
        the previous attention processor is returned so that the caller can restore it later.
        """
        self.conditioning = conditioning
        self.cross_attention_control_context = Context(
            arguments=self.conditioning.cross_attention_control_args,
            step_count=step_count
        )
        return override_cross_attention(self.model,
                                        self.cross_attention_control_context,
                                        is_running_diffusers=self.is_running_diffusers)

    def restore_default_cross_attention(self, restore_attention_processor: Optional['AttnProcessor']=None):
        self.conditioning = None
        self.cross_attention_control_context = None
        restore_default_cross_attention(self.model,
                                        is_running_diffusers=self.is_running_diffusers,
                                        restore_attention_processor=restore_attention_processor)

    def setup_attention_map_saving(self, saver: AttentionMapSaver):
        def callback(slice, dim, offset, slice_size, key):
            if dim is not None:
                # sliced tokens attention map saving is not implemented
                return
            saver.add_attention_maps(slice, key)

        tokens_cross_attention_modules = get_cross_attention_modules(self.model, CrossAttentionType.TOKENS)
        for identifier, module in tokens_cross_attention_modules:
            key = ('down' if identifier.startswith('down') else
                   'up' if identifier.startswith('up') else
                   'mid')
            module.set_attention_slice_calculated_callback(
                lambda slice, dim, offset, slice_size, key=key: callback(slice, dim, offset, slice_size, key))

    def remove_attention_map_saving(self):
        tokens_cross_attention_modules = get_cross_attention_modules(self.model, CrossAttentionType.TOKENS)
        for _, module in tokens_cross_attention_modules:
            module.set_attention_slice_calculated_callback(None)

    def do_diffusion_step(self, x: torch.Tensor, sigma: torch.Tensor,
                                unconditioning: Union[torch.Tensor,dict],
                                conditioning: Union[torch.Tensor,dict],
                                unconditional_guidance_scale: float,
                                step_index: Optional[int]=None,
                                total_step_count: Optional[int]=None,
                                threshold: Optional[ThresholdSettings]=None,
                          ):
        """
        :param x: current latents
        :param sigma: aka t, passed to the internal model to control how much denoising will occur
        :param unconditioning: embeddings for unconditioned output. for hybrid conditioning this is a dict of tensors [B x 77 x 768], otherwise a single tensor [B x 77 x 768]
        :param conditioning: embeddings for conditioned output. for hybrid conditioning this is a dict of tensors [B x 77 x 768], otherwise a single tensor [B x 77 x 768]
        :param unconditional_guidance_scale: aka CFG scale, controls how much effect the conditioning tensor has
        :param step_index: counts upwards from 0 to (step_count-1) (as passed to setup_cross_attention_control, if using). May be called multiple times for a single step, therefore do not assume that its value will monotically increase. If None, will be estimated by comparing sigma against self.model.sigmas .
        :param threshold: threshold to apply after each step
        :return: the new latents after applying the model to x using unscaled unconditioning and CFG-scaled conditioning.
        """


        cross_attention_control_types_to_do = []
        context: Context = self.cross_attention_control_context
        if self.cross_attention_control_context is not None:
            if step_index is not None and total_step_count is not None:
                # 🧨diffusers codepath
                percent_through = step_index / total_step_count  # will never reach 1.0 - this is deliberate
            else:
                # legacy compvis codepath
                # TODO remove when compvis codepath support is dropped
                if step_index is None and sigma is None:
                    raise ValueError(f"Either step_index or sigma is required when doing cross attention control, but both are None.")
                percent_through = self.estimate_percent_through(step_index, sigma)
            cross_attention_control_types_to_do = context.get_active_cross_attention_control_types_for_step(percent_through)

        wants_cross_attention_control = (len(cross_attention_control_types_to_do) > 0)
        wants_hybrid_conditioning = isinstance(conditioning, dict)

        if wants_hybrid_conditioning:
            unconditioned_next_x, conditioned_next_x = self.apply_hybrid_conditioning(x, sigma, unconditioning, conditioning)
        elif wants_cross_attention_control:
            unconditioned_next_x, conditioned_next_x = self.apply_cross_attention_controlled_conditioning(x, sigma, unconditioning, conditioning, cross_attention_control_types_to_do)
        else:
            unconditioned_next_x, conditioned_next_x = self.apply_standard_conditioning(x, sigma, unconditioning, conditioning)

        combined_next_x = self._combine(unconditioned_next_x, conditioned_next_x, unconditional_guidance_scale)

        if threshold:
            combined_next_x = self._threshold(threshold.threshold, threshold.warmup, combined_next_x, sigma)

        return combined_next_x

    # methods below are called from do_diffusion_step and should be considered private to this class.

    def apply_standard_conditioning(self, x, sigma, unconditioning, conditioning):
        # fast batched path
        x_twice = torch.cat([x] * 2)
        sigma_twice = torch.cat([sigma] * 2)
        both_conditionings = torch.cat([unconditioning, conditioning])
        both_results = self.model_forward_callback(x_twice, sigma_twice, both_conditionings)
        unconditioned_next_x, conditioned_next_x = both_results.chunk(2)
        if conditioned_next_x.device.type == 'mps':
            # prevent a result filled with zeros. seems to be a torch bug.
            conditioned_next_x = conditioned_next_x.clone()
        return unconditioned_next_x, conditioned_next_x


    def apply_hybrid_conditioning(self, x, sigma, unconditioning, conditioning):
        assert isinstance(conditioning, dict)
        assert isinstance(unconditioning, dict)
        x_twice = torch.cat([x] * 2)
        sigma_twice = torch.cat([sigma] * 2)
        both_conditionings = dict()
        for k in conditioning:
            if isinstance(conditioning[k], list):
                both_conditionings[k] = [
                    torch.cat([unconditioning[k][i], conditioning[k][i]])
                    for i in range(len(conditioning[k]))
                ]
            else:
                both_conditionings[k] = torch.cat([unconditioning[k], conditioning[k]])
        unconditioned_next_x, conditioned_next_x = self.model_forward_callback(x_twice, sigma_twice, both_conditionings).chunk(2)
        return unconditioned_next_x, conditioned_next_x


    def apply_cross_attention_controlled_conditioning(self,
                                                     x: torch.Tensor,
                                                     sigma,
                                                     unconditioning,
                                                     conditioning,
                                                     cross_attention_control_types_to_do):
        if self.is_running_diffusers:
            return self.apply_cross_attention_controlled_conditioning__diffusers(x, sigma, unconditioning, conditioning, cross_attention_control_types_to_do)
        else:
            return self.apply_cross_attention_controlled_conditioning__compvis(x, sigma, unconditioning, conditioning, cross_attention_control_types_to_do)

    def apply_cross_attention_controlled_conditioning__diffusers(self,
                                                                 x: torch.Tensor,
                                                                 sigma,
                                                                 unconditioning,
                                                                 conditioning,
                                                                 cross_attention_control_types_to_do):
        context: Context = self.cross_attention_control_context

        cross_attn_processor_context = SwapCrossAttnContext(modified_text_embeddings=context.arguments.edited_conditioning,
                                                            index_map=context.cross_attention_index_map,
                                                            mask=context.cross_attention_mask,
                                                            cross_attention_types_to_do=[])
        # no cross attention for unconditioning (negative prompt)
        unconditioned_next_x = self.model_forward_callback(x, sigma, unconditioning,
                                                           {"swap_cross_attn_context": cross_attn_processor_context})

        # do requested cross attention types for conditioning (positive prompt)
        cross_attn_processor_context.cross_attention_types_to_do = cross_attention_control_types_to_do
        conditioned_next_x = self.model_forward_callback(x, sigma, conditioning,
                                                         {"swap_cross_attn_context": cross_attn_processor_context})
        return unconditioned_next_x, conditioned_next_x


    def apply_cross_attention_controlled_conditioning__compvis(self, x:torch.Tensor, sigma, unconditioning, conditioning, cross_attention_control_types_to_do):
        # print('pct', percent_through, ': doing cross attention control on', cross_attention_control_types_to_do)
        # slower non-batched path (20% slower on mac MPS)
        # We are only interested in using attention maps for conditioned_next_x, but batching them with generation of
        # unconditioned_next_x causes attention maps to *also* be saved for the unconditioned_next_x.
        # This messes app their application later, due to mismatched shape of dim 0 (seems to be 16 for batched vs. 8)
        # (For the batched invocation the `wrangler` function gets attention tensor with shape[0]=16,
        # representing batched uncond + cond, but then when it comes to applying the saved attention, the
        # wrangler gets an attention tensor which only has shape[0]=8, representing just self.edited_conditionings.)
        # todo: give CrossAttentionControl's `wrangler` function more info so it can work with a batched call as well.
        context:Context = self.cross_attention_control_context

        try:
            unconditioned_next_x = self.model_forward_callback(x, sigma, unconditioning)

            # process x using the original prompt, saving the attention maps
            #print("saving attention maps for", cross_attention_control_types_to_do)
            for ca_type in cross_attention_control_types_to_do:
                context.request_save_attention_maps(ca_type)
            _ = self.model_forward_callback(x, sigma, conditioning)
            context.clear_requests(cleanup=False)

            # process x again, using the saved attention maps to control where self.edited_conditioning will be applied
            #print("applying saved attention maps for", cross_attention_control_types_to_do)
            for ca_type in cross_attention_control_types_to_do:
                context.request_apply_saved_attention_maps(ca_type)
            edited_conditioning = self.conditioning.cross_attention_control_args.edited_conditioning
            conditioned_next_x = self.model_forward_callback(x, sigma, edited_conditioning)
            context.clear_requests(cleanup=True)

        except:
            context.clear_requests(cleanup=True)
            raise

        return unconditioned_next_x, conditioned_next_x

    def _combine(self, unconditioned_next_x, conditioned_next_x, guidance_scale):
        # to scale how much effect conditioning has, calculate the changes it does and then scale that
        scaled_delta = (conditioned_next_x - unconditioned_next_x) * guidance_scale
        combined_next_x = unconditioned_next_x + scaled_delta
        return combined_next_x

    def _threshold(self, threshold, warmup, latents: torch.Tensor, sigma) -> torch.Tensor:
        warmup_scale = (1 - sigma.item() / 1000) / warmup if warmup else math.inf
        if warmup_scale < 1:
            # This arithmetic based on https://github.com/invoke-ai/InvokeAI/pull/395
            warming_threshold = 1 + (threshold - 1) * warmup_scale
            current_threshold = np.clip(warming_threshold, 1, threshold)
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
            print(f"\nThreshold: 𝜎={sigma.item()} threshold={current_threshold:.3f} (of {threshold:.3f})\n"
                  f"  | min, mean, max = {minval:.3f}, {mean:.3f}, {maxval:.3f}\tstd={std}\n"
                  f"  | {outside / latents.numel() * 100:.2f}% values outside threshold")

        if maxval < current_threshold and minval > -current_threshold:
            return latents

        if maxval > current_threshold:
            maxval = np.clip(maxval * scale, 1, current_threshold)

        if minval < -current_threshold:
            minval = np.clip(minval * scale, -current_threshold, -1)

        if self.debug_thresholding:
            outside = torch.count_nonzero((latents < minval) | (latents > maxval))
            print(f"  | min,     , max = {minval:.3f},        , {maxval:.3f}\t(scaled by {scale})\n"
                  f"  | {outside / latents.numel() * 100:.2f}% values will be clamped")

        return latents.clamp(minval, maxval)

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
        t_in = torch.cat([t] * 2) # aka sigmas

        deltas = None
        uncond_latents = None
        weighted_cond_list = c_or_weighted_c_list if type(c_or_weighted_c_list) is list else [(c_or_weighted_c_list, 1)]

        # below is fugly omg
        num_actual_conditionings = len(c_or_weighted_c_list)
        conditionings = [uc] + [c for c,weight in weighted_cond_list]
        weights = [1] + [weight for c,weight in weighted_cond_list]
        chunk_count = ceil(len(conditionings)/2)
        deltas = None
        for chunk_index in range(chunk_count):
            offset = chunk_index*2
            chunk_size = min(2, len(conditionings)-offset)

            if chunk_size == 1:
                c_in = conditionings[offset]
                latents_a = forward_func(x_in[:-1], t_in[:-1], c_in)
                latents_b = None
            else:
                c_in = torch.cat(conditionings[offset:offset+2])
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

