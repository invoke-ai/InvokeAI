from enum import Enum
import torch


# adapted from bloc97's CrossAttentionControl colab
# https://github.com/bloc97/CrossAttentionControl

class CrossAttentionControl:


    @classmethod
    def clear_attention_editing(cls, model):
        cls.remove_attention_function(model)

    @classmethod
    def setup_attention_editing(cls, model,
                                substitute_conditioning: torch.Tensor,
                                edit_opcodes: list):
        """
        :param model: The unet model to inject into.
        :param substitute_conditioning: The "edited" conditioning vector, [Bx77x768]
        :param edit_opcodes: Opcodes from difflib.SequenceMatcher describing how the base
                             conditionings map to the "edited" conditionings.
        :return:
        """

        # adapted from init_attention_edit
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
