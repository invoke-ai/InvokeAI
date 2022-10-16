from enum import Enum
import torch


class CrossAttentionControl:
    class AttentionType(Enum):
        SELF = 1
        TOKENS = 2

    @classmethod
    def get_attention_module(cls, model, which: AttentionType):
        which_attn = "attn1" if which is cls.AttentionType.SELF else "attn2"
        module = next(module for name, module in model.named_modules() if
                      type(module).__name__ == "CrossAttention" and which_attn in name)
        return module

    @classmethod
    def setup_attention_editing(cls, model, original_tokens_length: int,
                                substitute_conditioning: torch.Tensor = None,
                                token_indices_to_edit: list = None):

        # adapted from init_attention_edit
        self_attention_module = cls.get_attention_module(model, cls.AttentionType.SELF)
        tokens_attention_module = cls.get_attention_module(model, cls.AttentionType.TOKENS)

        if substitute_conditioning is not None:

            device = substitute_conditioning.device

            # this is not very torch-y
            mask = torch.zeros(original_tokens_length)
            for i in token_indices_to_edit:
                mask[i] = 1

            self_attention_module.last_attn_slice_mask = None
            self_attention_module.last_attn_slice_indices = None
            tokens_attention_module.last_attn_slice_mask = mask.to(device)
            tokens_attention_module.last_attn_slice_indices = torch.tensor(token_indices_to_edit, device=device)

        cls.inject_attention_functions(model)

    @classmethod
    def request_save_attention_maps(cls, model):
        self_attention_module = cls.get_attention_module(model, cls.AttentionType.SELF)
        tokens_attention_module = cls.get_attention_module(model, cls.AttentionType.TOKENS)
        self_attention_module.save_last_attn_slice = True
        tokens_attention_module.save_last_attn_slice = True

    @classmethod
    def request_apply_saved_attention_maps(cls, model):
        self_attention_module = cls.get_attention_module(model, cls.AttentionType.SELF)
        tokens_attention_module = cls.get_attention_module(model, cls.AttentionType.TOKENS)
        self_attention_module.use_last_attn_slice = True
        tokens_attention_module.use_last_attn_slice = True

    @classmethod
    def inject_attention_functions(cls, unet):
        # ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276
        def new_attention(self, query, key, value):
            # TODO: use baddbmm for better performance
            print(f"entered new_attention")
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            attn_slice = attention_scores.softmax(dim=-1)
            # compute attention output

            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    print('using masked last_attn_slice')

                    new_attn_slice = (torch.index_select(self.last_attn_slice, -1,
                                                         self.last_attn_slice_indices))
                    attn_slice = (attn_slice * (1 - self.last_attn_slice_mask)
                                  + new_attn_slice * self.last_attn_slice_mask)
                else:
                    print('using unmasked last_attn_slice')
                    attn_slice = self.last_attn_slice

                self.use_last_attn_slice = False
            else:
                print('not using last_attn_slice')

            if self.save_last_attn_slice:
                print('saving last_attn_slice')
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False
            else:
                print('not saving last_attn_slice')

            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False

            hidden_states = torch.matmul(attn_slice, value)
            # reshape hidden_states
            return hidden_states

        for _, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == 'CrossAttention':
                module.last_attn_slice = None
                module.use_last_attn_slice = False
                module.use_last_attn_weights = False
                module.save_last_attn_slice = False
                module.cross_attention_callback = new_attention.__get__(module, type(module))


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
