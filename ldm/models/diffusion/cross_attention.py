from enum import Enum
import torch


# adapted from bloc97's CrossAttentionControl colab
# https://github.com/bloc97/CrossAttentionControl

class CrossAttentionControl:
    class AttentionType(Enum):
        SELF = 1
        TOKENS = 2

    @classmethod
    def setup_attention_editing(cls, model,
                                substitute_conditioning: torch.Tensor = None,
                                edit_opcodes: list = None):
        """
        :param model: The unet model to inject into.
        :param substitute_conditioning: The "edited" conditioning vector, [Bx77x768]
        :param edit_opcodes: Opcodes from difflib.SequenceMatcher describing how the base
                             conditionings map to the "edited" conditionings.
        :return:
        """

        # adapted from init_attention_edit
        if substitute_conditioning is not None:

            device = substitute_conditioning.device

            max_length = model.inner_model.cond_stage_model.max_length
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

        cls.inject_attention_functions(model)


    @classmethod
    def get_attention_modules(cls, model, which: AttentionType):
        which_attn = "attn1" if which is cls.AttentionType.SELF else "attn2"
        return [module for name, module in model.named_modules() if
                      type(module).__name__ == "CrossAttention" and which_attn in name]


    @classmethod
    def request_save_attention_maps(cls, model):
        self_attention_modules = cls.get_attention_modules(model, cls.AttentionType.SELF)
        tokens_attention_modules = cls.get_attention_modules(model, cls.AttentionType.TOKENS)
        for m in self_attention_modules+tokens_attention_modules:
            m.save_last_attn_slice = True

    @classmethod
    def request_apply_saved_attention_maps(cls, model):
        self_attention_modules = cls.get_attention_modules(model, cls.AttentionType.SELF)
        tokens_attention_modules = cls.get_attention_modules(model, cls.AttentionType.TOKENS)
        for m in self_attention_modules+tokens_attention_modules:
            m.use_last_attn_slice = True


    @classmethod
    def inject_attention_functions(cls, unet):
        # ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276

        def new_attention(self, query, key, value):

            attention_scores = torch.functional.einsum('b i d, b j d -> b i j', query, key)
            # calculate attention slice by taking the best scores for each latent pixel
            attn_slice = attention_scores.softmax(dim=-1, dtype=attention_scores.dtype)

            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    base_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    base_attn_slice_mask = self.last_attn_slice_mask
                    this_attn_slice_mask = 1 - self.last_attn_slice_mask
                    attn_slice = attn_slice * this_attn_slice_mask + base_attn_slice * base_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice

                self.use_last_attn_slice = False

            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False

            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False

            return torch.functional.einsum('b i j, b j d -> b i d', attn_slice, value)

        def new_sliced_attention(self, query, key, value, sequence_length, dim):

            raise NotImplementedError("not tested yet")

            batch_size_attention = query.shape[0]
            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
            )
            slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
            for i in range(hidden_states.shape[0] // slice_size):
                start_idx = i * slice_size
                end_idx = (i + 1) * slice_size
                attn_slice = (
                        torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
                )  # TODO: use baddbmm for better performance
                attn_slice = attn_slice.softmax(dim=-1)

                if self.use_last_attn_slice:
                    if self.last_attn_slice_mask is not None:
                        new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                        attn_slice = attn_slice * (
                                1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                    else:
                        attn_slice = self.last_attn_slice

                    self.use_last_attn_slice = False

                if self.save_last_attn_slice:
                    self.last_attn_slice = attn_slice
                    self.save_last_attn_slice = False

                if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                    attn_slice = attn_slice * self.last_attn_slice_weights
                    self.use_last_attn_weights = False

                attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

                hidden_states[start_idx:end_idx] = attn_slice

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        def select_attention_func(module, q, k, v, dim, offset, slice_size):
            if dim == 0 or dim == 1:
                return new_sliced_attention(module, q, k, v, sequence_length=slice_size, dim=dim)
            else:
                return new_attention(module, q, k, v)

        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention":
                module.last_attn_slice = None
                module.use_last_attn_slice = False
                module.use_last_attn_weights = False
                module.save_last_attn_slice = False
                module.set_custom_attention_calculator(select_attention_func)


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
