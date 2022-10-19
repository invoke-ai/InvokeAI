from enum import Enum
from typing import Callable

import torch


class InvokeAIDiffuserComponent:

    class StructuredConditioning:
        def __init__(self, edited_conditioning: torch.Tensor = None, edit_opcodes: list[tuple] = None):
            """
            :param edited_conditioning: if doing cross-attention control, the edited conditioning (1 x 77 x 768)
            :param edit_opcodes: if doing cross-attention control, opcodes from a SequenceMatcher describing how to map original conditioning tokens to edited conditioning tokens
            """
            # TODO migrate conditioning and unconditioning here, too
            #self.conditioning = conditioning
            #self.unconditioning = unconditioning
            self.edited_conditioning = edited_conditioning
            self.edit_opcodes = edit_opcodes

        @property
        def wants_cross_attention_control(self):
            return self.edited_conditioning is not None

    '''
    The aim of this component is to provide a single place for code that can be applied identically to
    all InvokeAI diffusion procedures.

    At the moment it includes the following features:
    * Cross Attention Control ("prompt2prompt")
    '''

    def __init__(self, model, model_forward_callback: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        :param model: the unet model to pass through to cross attention control
        :param model_forward_callback: a lambda with arguments (x, sigma, conditioning_to_apply). will be called repeatedly. most likely, this should simply call model.forward(x, sigma, conditioning)
        """
        self.model = model
        self.model_forward_callback = model_forward_callback


    def setup_cross_attention_control(self, conditioning: StructuredConditioning):
        self.conditioning = conditioning
        CrossAttentionControl.setup_cross_attention_control(self.model, conditioning.edited_conditioning, conditioning.edit_opcodes)

    def remove_cross_attention_control(self):
        self.conditioning = None
        CrossAttentionControl.remove_cross_attention_control(self.model)

    @property
    def edited_conditioning(self):
        if self.conditioning is None:
            return None
        else:
            return self.conditioning.edited_conditioning

    def do_diffusion_step(self, x: torch.Tensor, sigma: torch.Tensor,
                                   unconditioning: torch.Tensor, conditioning: torch.Tensor,
                                   unconditional_guidance_scale: float):
        """
        :param x: Current latents
        :param sigma: aka t, passed to the internal model to control how much denoising will occur
        :param unconditioning: [B x 77 x 768] embeddings for unconditioned output
        :param conditioning: [B x 77 x 768] embeddings for conditioned output
        :param unconditional_guidance_scale: aka CFG scale, controls how much effect the conditioning tensor has
        :param model: the unet model to pass through to cross attention control
        :param model_forward_callback: a lambda with arguments (x, sigma, conditioning_to_apply). will be called repeatedly. most likely, this should simply call model.forward(x, sigma, conditioning)
        :return: the new latents after applying the model to x using unconditioning and CFG-scaled conditioning.
        """

        CrossAttentionControl.clear_requests(self.model)

        if self.edited_conditioning is None:
            # faster batched path
            x_twice = torch.cat([x]*2)
            sigma_twice = torch.cat([sigma]*2)
            both_conditionings = torch.cat([unconditioning, conditioning])
            unconditioned_next_x, conditioned_next_x = self.model_forward_callback(x_twice, sigma_twice, both_conditionings).chunk(2)
        else:
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
            CrossAttentionControl.request_save_attention_maps(self.model)
            _ = self.model_forward_callback(x, sigma, conditioning)
            CrossAttentionControl.clear_requests(self.model)

            # process x again, using the saved attention maps to control where self.edited_conditioning will be applied
            CrossAttentionControl.request_apply_saved_attention_maps(self.model)
            conditioned_next_x = self.model_forward_callback(x, sigma, self.edited_conditioning)
            CrossAttentionControl.clear_requests(self.model)


        # to scale how much effect conditioning has, calculate the changes it does and then scale that
        scaled_delta = (conditioned_next_x - unconditioned_next_x) * unconditional_guidance_scale
        combined_next_x = unconditioned_next_x + scaled_delta

        return combined_next_x

# adapted from bloc97's CrossAttentionControl colab
# https://github.com/bloc97/CrossAttentionControl

class CrossAttentionControl:


    @classmethod
    def remove_cross_attention_control(cls, model):
        cls.remove_attention_function(model)

    @classmethod
    def setup_cross_attention_control(cls, model,
                                      substitute_conditioning: torch.Tensor,
                                      edit_opcodes: list):
        """
        Inject attention parameters and functions into the passed in model to enable cross attention editing.

        :param model: The unet model to inject into.
        :param substitute_conditioning: The "edited" conditioning vector, [Bx77x768]
        :param edit_opcodes: Opcodes from difflib.SequenceMatcher describing how the base
                             conditionings map to the "edited" conditionings.
        :return:
        """

        # adapted from init_attention_edit
        device = substitute_conditioning.device

        # urgh. should this be hardcoded?
        max_length = 77
        # mask=1 means use base prompt attention, mask=0 means use edited prompt attention
        mask = torch.zeros(max_length)
        indices_target = torch.arange(max_length, dtype=torch.long)
        indices = torch.zeros(max_length, dtype=torch.long)
        for name, a0, a1, b0, b1 in edit_opcodes:
            if b0 < max_length:
                if name == "equal":# or (name == "replace" and a1 - a0 == b1 - b0):
                    # these tokens have not been edited
                    indices[b0:b1] = indices_target[a0:a1]
                    mask[b0:b1] = 1

        for m in cls.get_attention_modules(model, cls.AttentionType.SELF):
            m.last_attn_slice_mask = None
            m.last_attn_slice_indices = None

        for m in cls.get_attention_modules(model, cls.AttentionType.TOKENS):
            m.last_attn_slice_mask = mask.to(device)
            m.last_attn_slice_indices = indices.to(device)

        cls.inject_attention_function(model)


    class AttentionType(Enum):
        SELF = 1
        TOKENS = 2


    @classmethod
    def get_attention_modules(cls, model, which: AttentionType):
        which_attn = "attn1" if which is cls.AttentionType.SELF else "attn2"
        return [module for name, module in model.named_modules() if
                      type(module).__name__ == "CrossAttention" and which_attn in name]

    @classmethod
    def clear_requests(cls, model):
        self_attention_modules = cls.get_attention_modules(model, cls.AttentionType.SELF)
        tokens_attention_modules = cls.get_attention_modules(model, cls.AttentionType.TOKENS)
        for m in self_attention_modules+tokens_attention_modules:
            m.save_last_attn_slice = False
            m.use_last_attn_slice = False

    @classmethod
    def request_save_attention_maps(cls, model):
        self_attention_modules = cls.get_attention_modules(model, cls.AttentionType.SELF)
        tokens_attention_modules = cls.get_attention_modules(model, cls.AttentionType.TOKENS)
        for m in self_attention_modules+tokens_attention_modules:
            # clear out the saved slice in case the outermost dim changes
            m.last_attn_slice = None
            m.save_last_attn_slice = True

    @classmethod
    def request_apply_saved_attention_maps(cls, model):
        self_attention_modules = cls.get_attention_modules(model, cls.AttentionType.SELF)
        tokens_attention_modules = cls.get_attention_modules(model, cls.AttentionType.TOKENS)
        for m in self_attention_modules+tokens_attention_modules:
            m.use_last_attn_slice = True



    @classmethod
    def inject_attention_function(cls, unet):
        # ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276

        def attention_slice_wrangler(self, attention_scores, suggested_attention_slice, dim, offset, slice_size):

            #print("in wrangler with suggested_attention_slice shape", suggested_attention_slice.shape, "dim", dim)

            attn_slice = suggested_attention_slice
            if dim is not None:
                start = offset
                end = start+slice_size
                #print(f"in wrangler, sliced dim {dim} {start}-{end}, use_last_attn_slice is {self.use_last_attn_slice}, save_last_attn_slice is {self.save_last_attn_slice}")
            #else:
            #    print(f"in wrangler, whole, use_last_attn_slice is {self.use_last_attn_slice}, save_last_attn_slice is {self.save_last_attn_slice}")


            if self.use_last_attn_slice:
                this_attn_slice = attn_slice
                if self.last_attn_slice_mask is not None:
                    # indices and mask operate on dim=2, no need to slice
                    base_attn_slice_full = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    base_attn_slice_mask = self.last_attn_slice_mask
                    if dim is None:
                        base_attn_slice = base_attn_slice_full
                        #print("using whole base slice of shape", base_attn_slice.shape, "from complete shape", base_attn_slice_full.shape)
                    elif dim == 0:
                        base_attn_slice = base_attn_slice_full[start:end]
                        #print("using base dim 0 slice of shape", base_attn_slice.shape, "from complete shape", base_attn_slice_full.shape)
                    elif dim == 1:
                        base_attn_slice = base_attn_slice_full[:, start:end]
                        #print("using base dim 1 slice of shape", base_attn_slice.shape, "from complete shape", base_attn_slice_full.shape)

                    attn_slice = this_attn_slice * (1 - base_attn_slice_mask) + \
                                 base_attn_slice * base_attn_slice_mask
                else:
                    if dim is None:
                        attn_slice = self.last_attn_slice
                        #print("took whole slice of shape", attn_slice.shape, "from complete shape", self.last_attn_slice.shape)
                    elif dim == 0:
                        attn_slice = self.last_attn_slice[start:end]
                        #print("took dim 0 slice of shape", attn_slice.shape, "from complete shape", self.last_attn_slice.shape)
                    elif dim == 1:
                        attn_slice = self.last_attn_slice[:, start:end]
                        #print("took dim 1 slice of shape", attn_slice.shape, "from complete shape", self.last_attn_slice.shape)

            if self.save_last_attn_slice:
                if dim is None:
                    self.last_attn_slice = attn_slice
                elif dim == 0:
                    # dynamically grow last_attn_slice if needed
                    if self.last_attn_slice is None:
                        self.last_attn_slice = attn_slice
                        #print("no last_attn_slice: shape now", self.last_attn_slice.shape)
                    elif self.last_attn_slice.shape[0] == start:
                        self.last_attn_slice = torch.cat([self.last_attn_slice, attn_slice], dim=0)
                        assert(self.last_attn_slice.shape[0] == end)
                        #print("last_attn_slice too small, appended dim 0 shape", attn_slice.shape, ", shape now", self.last_attn_slice.shape)
                    else:
                        # no need to grow
                        self.last_attn_slice[start:end] = attn_slice
                        #print("last_attn_slice shape is fine, setting dim 0 shape", attn_slice.shape, ", shape now", self.last_attn_slice.shape)

                elif dim == 1:
                    # dynamically grow last_attn_slice if needed
                    if self.last_attn_slice is None:
                        self.last_attn_slice = attn_slice
                    elif self.last_attn_slice.shape[1] == start:
                        self.last_attn_slice = torch.cat([self.last_attn_slice, attn_slice], dim=1)
                        assert(self.last_attn_slice.shape[1] == end)
                    else:
                        # no need to grow
                        self.last_attn_slice[:, start:end] = attn_slice

            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                if dim is None:
                    weights = self.last_attn_slice_weights
                elif dim == 0:
                    weights = self.last_attn_slice_weights[start:end]
                elif dim == 1:
                    weights = self.last_attn_slice_weights[:, start:end]
                attn_slice = attn_slice * weights

            return attn_slice

        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention":
                module.last_attn_slice = None
                module.last_attn_slice_indices = None
                module.last_attn_slice_mask = None
                module.use_last_attn_weights = False
                module.use_last_attn_slice = False
                module.save_last_attn_slice = False
                module.set_attention_slice_wrangler(attention_slice_wrangler)

    @classmethod
    def remove_attention_function(cls, unet):
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention":
                module.set_attention_slice_wrangler(None)


# original code below

# Functions supporting Cross-Attention Control
# Copied from https://github.com/bloc97/CrossAttentionControl

from difflib import SequenceMatcher

import torch


def prompt_token(prompt, index, clip_tokenizer):
    tokens = clip_tokenizer(prompt,
                            padding='max_length',
                            max_length=clip_tokenizer.model_max_length,
                            truncation=True,
                            return_tensors='pt',
                            return_overflowing_tokens=True
                            ).input_ids[0]
    return clip_tokenizer.decode(tokens[index:index + 1])


def use_last_tokens_attention(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == 'CrossAttention' and 'attn2' in name:
            module.use_last_attn_slice = use


def use_last_tokens_attention_weights(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == 'CrossAttention' and 'attn2' in name:
            module.use_last_attn_weights = use


def use_last_self_attention(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == 'CrossAttention' and 'attn1' in name:
            module.use_last_attn_slice = use


def save_last_tokens_attention(unet, save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == 'CrossAttention' and 'attn2' in name:
            module.save_last_attn_slice = save


def save_last_self_attention(unet, save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == 'CrossAttention' and 'attn1' in name:
            module.save_last_attn_slice = save
