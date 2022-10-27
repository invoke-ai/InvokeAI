from math import ceil
from typing import Callable, Optional

import torch

from ldm.models.diffusion.cross_attention_control import CrossAttentionControl


class InvokeAIDiffuserComponent:
    '''
    The aim of this component is to provide a single place for code that can be applied identically to
    all InvokeAI diffusion procedures.

    At the moment it includes the following features:
    * Cross Attention Control ("prompt2prompt")
    '''


    class ExtraConditioningInfo:
        def __init__(self, cross_attention_control_args: Optional[CrossAttentionControl.Arguments]):
            self.cross_attention_control_args = cross_attention_control_args

        @property
        def wants_cross_attention_control(self):
            return self.cross_attention_control_args is not None

    def __init__(self, model, model_forward_callback:
                    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
                ):
        """
        :param model: the unet model to pass through to cross attention control
        :param model_forward_callback: a lambda with arguments (x, sigma, conditioning_to_apply). will be called repeatedly. most likely, this should simply call model.forward(x, sigma, conditioning)
        """
        self.model = model
        self.model_forward_callback = model_forward_callback


    def setup_cross_attention_control(self, conditioning: ExtraConditioningInfo, step_count: int):
        self.conditioning = conditioning
        self.cross_attention_control_context = CrossAttentionControl.Context(
            arguments=self.conditioning.cross_attention_control_args,
            step_count=step_count
        )
        CrossAttentionControl.setup_cross_attention_control(self.model,
                                                            cross_attention_control_args=self.conditioning.cross_attention_control_args
                                                            )
        #todo: refactor  edited_conditioning, edit_opcodes, edit_options into a struct
        #todo: apply edit_options using step_count


    def remove_cross_attention_control(self):
        self.conditioning = None
        self.cross_attention_control_context = None
        CrossAttentionControl.remove_cross_attention_control(self.model)

    def do_diffusion_step(self, x: torch.Tensor, sigma: torch.Tensor,
                                   unconditioning: torch.Tensor, conditioning: torch.Tensor,
                                   unconditional_guidance_scale: float,
                                   step_index: int=None
                          ):
        """
        :param x: Current latents
        :param sigma: aka t, passed to the internal model to control how much denoising will occur
        :param unconditioning: [B x 77 x 768] embeddings for unconditioned output
        :param conditioning: [B x 77 x 768] embeddings for conditioned output
        :param unconditional_guidance_scale: aka CFG scale, controls how much effect the conditioning tensor has
        :param step_index: Counts upwards from 0 to (step_count-1) (as passed to setup_cross_attention_control, if using). May be called multiple times for a single step, therefore do not assume that its value will monotically increase.
        :return: the new latents after applying the model to x using unscaled unconditioning and CFG-scaled conditioning.
        """

        CrossAttentionControl.clear_requests(self.model)
        cross_attention_control_types_to_do = []

        if self.cross_attention_control_context is not None:
            if step_index is not None:
                # percent_through will never reach 1.0 (but this is intended)
                percent_through = float(step_index) / float(self.cross_attention_control_context.step_count)
            else:
                # find the current sigma in the sigma sequence
                # todo: this doesn't work with k_dpm_2 because the sigma used jumps around in the sequence
                sigma_index = torch.nonzero(self.model.sigmas <= sigma)[-1]
                # flip because sigmas[0] is for the fully denoised image
                # percent_through must be <1
                percent_through = 1.0 - float(sigma_index.item() + 1) / float(self.model.sigmas.shape[0])
                #print('estimated percent_through', percent_through, 'from sigma', sigma.item())
            cross_attention_control_types_to_do = CrossAttentionControl.get_active_cross_attention_control_types_for_step(self.cross_attention_control_context, percent_through)

        if len(cross_attention_control_types_to_do)==0:
            #print('not doing cross attention control')
            # faster batched path
            x_twice = torch.cat([x]*2)
            sigma_twice = torch.cat([sigma]*2)
            if isinstance(conditioning, dict):
                assert isinstance(unconditioning, dict)
                both_conditionings = dict()
                for k in conditioning:
                    if isinstance(conditioning[k], list):
                        both_conditionings[k] = [
                            torch.cat([unconditioning[k][i], conditioning[k][i]])
                            for i in range(len(conditioning[k]))
                        ]
                    else:
                        both_conditionings[k] = torch.cat([unconditioning[k], conditioning[k]])
            else:
                both_conditionings = torch.cat([unconditioning, conditioning])
            unconditioned_next_x, conditioned_next_x = self.model_forward_callback(x_twice, sigma_twice, both_conditionings).chunk(2)
        else:
            #print('pct', percent_through, ': doing cross attention control on', cross_attention_control_types_to_do)
            # slower non-batched path (20% slower on mac MPS)
            # We are only interested in using attention maps for conditioned_next_x, but batching them with generation of
            # unconditioned_next_x causes attention maps to *also* be saved for the unconditioned_next_x.
            # This messes app their application later, due to mismatched shape of dim 0 (seems to be 16 for batched vs. 8)
            # (For the batched invocation the `wrangler` function gets attention tensor with shape[0]=16,
            # representing batched uncond + cond, but then when it comes to applying the saved attention, the
            # wrangler gets an attention tensor which only has shape[0]=8, representing just self.edited_conditionings.)
            # todo: give CrossAttentionControl's `wrangler` function more info so it can work with a batched call as well.
            unconditioned_next_x = self.model_forward_callback(x, sigma, unconditioning)

            # process x using the original prompt, saving the attention maps
            for type in cross_attention_control_types_to_do:
                CrossAttentionControl.request_save_attention_maps(self.model, type)
            _ = self.model_forward_callback(x, sigma, conditioning)
            CrossAttentionControl.clear_requests(self.model)

            # process x again, using the saved attention maps to control where self.edited_conditioning will be applied
            for type in cross_attention_control_types_to_do:
                CrossAttentionControl.request_apply_saved_attention_maps(self.model, type)
            edited_conditioning = self.conditioning.cross_attention_control_args.edited_conditioning
            conditioned_next_x = self.model_forward_callback(x, sigma, edited_conditioning)
            CrossAttentionControl.clear_requests(self.model)


        # to scale how much effect conditioning has, calculate the changes it does and then scale that
        scaled_delta = (conditioned_next_x - unconditioned_next_x) * unconditional_guidance_scale
        combined_next_x = unconditioned_next_x + scaled_delta

        return combined_next_x

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

