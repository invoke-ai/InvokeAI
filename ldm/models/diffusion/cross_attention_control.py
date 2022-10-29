from enum import Enum

import torch

# adapted from bloc97's CrossAttentionControl colab
# https://github.com/bloc97/CrossAttentionControl

class CrossAttentionControl:

    class Arguments:
        def __init__(self, edited_conditioning: torch.Tensor, edit_opcodes: list[tuple], edit_options: dict):
            """
            :param edited_conditioning: if doing cross-attention control, the edited conditioning [1 x 77 x 768]
            :param edit_opcodes: if doing cross-attention control, a list of difflib.SequenceMatcher-like opcodes describing how to map original conditioning tokens to edited conditioning tokens (only the 'equal' opcode is required)
            :param edit_options: if doing cross-attention control, per-edit options. there should be 1 item in edit_options for each item in edit_opcodes.
            """
            # todo: rewrite this to take embedding fragments rather than a single edited_conditioning vector
            self.edited_conditioning = edited_conditioning
            self.edit_opcodes = edit_opcodes

            if edited_conditioning is not None:
                assert len(edit_opcodes) == len(edit_options), \
                        "there must be 1 edit_options dict for each edit_opcodes tuple"
                non_none_edit_options = [x for x in edit_options if x is not None]
                assert len(non_none_edit_options)>0, "missing edit_options"
                if len(non_none_edit_options)>1:
                    print('warning: cross-attention control options are not working properly for >1 edit')
                self.edit_options = non_none_edit_options[0]

    class Context:
        def __init__(self, arguments: 'CrossAttentionControl.Arguments', step_count: int):
            """
            :param arguments: Arguments for the cross-attention control process
            :param step_count: The absolute total number of steps of diffusion (for img2img this is likely larger than the number of steps that will actually run)
            """
            self.arguments = arguments
            self.step_count = step_count

    @classmethod
    def remove_cross_attention_control(cls, model):
        cls.remove_attention_function(model)

    @classmethod
    def setup_cross_attention_control(cls, model,
                                      cross_attention_control_args: Arguments
                                      ):
        """
        Inject attention parameters and functions into the passed in model to enable cross attention editing.

        :param model: The unet model to inject into.
        :param cross_attention_control_args: Arugments passeed to the CrossAttentionControl implementations
        :return: None
        """

        # adapted from init_attention_edit
        device = cross_attention_control_args.edited_conditioning.device

        # urgh. should this be hardcoded?
        max_length = 77
        # mask=1 means use base prompt attention, mask=0 means use edited prompt attention
        mask = torch.zeros(max_length)
        indices_target = torch.arange(max_length, dtype=torch.long)
        indices = torch.zeros(max_length, dtype=torch.long)
        for name, a0, a1, b0, b1 in cross_attention_control_args.edit_opcodes:
            if b0 < max_length:
                if name == "equal":# or (name == "replace" and a1 - a0 == b1 - b0):
                    # these tokens have not been edited
                    indices[b0:b1] = indices_target[a0:a1]
                    mask[b0:b1] = 1

        for m in cls.get_attention_modules(model, cls.CrossAttentionType.SELF):
            m.last_attn_slice_mask = None
            m.last_attn_slice_indices = None

        for m in cls.get_attention_modules(model, cls.CrossAttentionType.TOKENS):
            m.last_attn_slice_mask = mask.to(device)
            m.last_attn_slice_indices = indices.to(device)

        cls.inject_attention_function(model)


    class CrossAttentionType(Enum):
        SELF = 1
        TOKENS = 2

    @classmethod
    def get_active_cross_attention_control_types_for_step(cls, context: 'CrossAttentionControl.Context', percent_through:float=None)\
            -> list['CrossAttentionControl.CrossAttentionType']:
        """
        Should cross-attention control be applied on the given step?
        :param percent_through: How far through the step sequence are we (0.0=pure noise, 1.0=completely denoised image). Expected range 0.0..<1.0.
        :return: A list of attention types that cross-attention control should be performed for on the given step. May be [].
        """
        if percent_through is None:
            return [cls.CrossAttentionType.SELF, cls.CrossAttentionType.TOKENS]

        opts = context.arguments.edit_options
        to_control = []
        if opts['s_start'] <= percent_through and percent_through < opts['s_end']:
            to_control.append(cls.CrossAttentionType.SELF)
        if opts['t_start'] <= percent_through and percent_through < opts['t_end']:
            to_control.append(cls.CrossAttentionType.TOKENS)
        return to_control


    @classmethod
    def get_attention_modules(cls, model, which: CrossAttentionType):
        which_attn = "attn1" if which is cls.CrossAttentionType.SELF else "attn2"
        return [module for name, module in model.named_modules() if
                      type(module).__name__ == "CrossAttention" and which_attn in name]

    @classmethod
    def clear_requests(cls, model):
        self_attention_modules = cls.get_attention_modules(model, cls.CrossAttentionType.SELF)
        tokens_attention_modules = cls.get_attention_modules(model, cls.CrossAttentionType.TOKENS)
        for m in self_attention_modules+tokens_attention_modules:
            m.save_last_attn_slice = False
            m.use_last_attn_slice = False

    @classmethod
    def request_save_attention_maps(cls, model, cross_attention_type: CrossAttentionType):
        modules = cls.get_attention_modules(model, cross_attention_type)
        for m in modules:
            # clear out the saved slice in case the outermost dim changes
            m.last_attn_slice = None
            m.save_last_attn_slice = True

    @classmethod
    def request_apply_saved_attention_maps(cls, model, cross_attention_type: CrossAttentionType):
        modules = cls.get_attention_modules(model, cross_attention_type)
        for m in modules:
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

