import enum
from typing import Optional

import torch

# adapted from bloc97's CrossAttentionControl colab
# https://github.com/bloc97/CrossAttentionControl


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


class CrossAttentionType(enum.Enum):
    SELF = 1
    TOKENS = 2


class Context:

    cross_attention_mask: Optional[torch.Tensor]
    cross_attention_index_map: Optional[torch.Tensor]

    class Action(enum.Enum):
        NONE = 0
        SAVE = 1,
        APPLY = 2

    def __init__(self, arguments: Arguments, step_count: int):
        """
        :param arguments: Arguments for the cross-attention control process
        :param step_count: The absolute total number of steps of diffusion (for img2img this is likely larger than the number of steps that will actually run)
        """
        self.cross_attention_mask = None
        self.cross_attention_index_map = None
        self.self_cross_attention_action = Context.Action.NONE
        self.tokens_cross_attention_action = Context.Action.NONE
        self.arguments = arguments
        self.step_count = step_count

        self.self_cross_attention_module_identifiers = []
        self.tokens_cross_attention_module_identifiers = []

        self.saved_cross_attention_maps = {}

        self.clear_requests(cleanup=True)

    def register_cross_attention_modules(self, model):
        for name,module in get_attention_modules(model, CrossAttentionType.SELF):
            self.self_cross_attention_module_identifiers.append(name)
        for name,module in get_attention_modules(model, CrossAttentionType.TOKENS):
            self.tokens_cross_attention_module_identifiers.append(name)

    def request_save_attention_maps(self, cross_attention_type: CrossAttentionType):
        if cross_attention_type == CrossAttentionType.SELF:
            self.self_cross_attention_action = Context.Action.SAVE
        else:
            self.tokens_cross_attention_action = Context.Action.SAVE

    def request_apply_saved_attention_maps(self, cross_attention_type: CrossAttentionType):
        if cross_attention_type == CrossAttentionType.SELF:
            self.self_cross_attention_action = Context.Action.APPLY
        else:
            self.tokens_cross_attention_action = Context.Action.APPLY

    def is_tokens_cross_attention(self, module_identifier) -> bool:
        return module_identifier in self.tokens_cross_attention_module_identifiers

    def get_should_save_maps(self, module_identifier: str) -> bool:
        if module_identifier in self.self_cross_attention_module_identifiers:
            return self.self_cross_attention_action == Context.Action.SAVE
        elif module_identifier in self.tokens_cross_attention_module_identifiers:
            return self.tokens_cross_attention_action == Context.Action.SAVE
        return False

    def get_should_apply_saved_maps(self, module_identifier: str) -> bool:
        if module_identifier in self.self_cross_attention_module_identifiers:
            return self.self_cross_attention_action == Context.Action.APPLY
        elif module_identifier in self.tokens_cross_attention_module_identifiers:
            return self.tokens_cross_attention_action == Context.Action.APPLY
        return False

    def get_active_cross_attention_control_types_for_step(self, percent_through:float=None)\
            -> list[CrossAttentionType]:
        """
        Should cross-attention control be applied on the given step?
        :param percent_through: How far through the step sequence are we (0.0=pure noise, 1.0=completely denoised image). Expected range 0.0..<1.0.
        :return: A list of attention types that cross-attention control should be performed for on the given step. May be [].
        """
        if percent_through is None:
            return [CrossAttentionType.SELF, CrossAttentionType.TOKENS]

        opts = self.arguments.edit_options
        to_control = []
        if opts['s_start'] <= percent_through < opts['s_end']:
            to_control.append(CrossAttentionType.SELF)
        if opts['t_start'] <= percent_through < opts['t_end']:
            to_control.append(CrossAttentionType.TOKENS)
        return to_control

    def save_slice(self, identifier: str, slice: torch.Tensor, dim: Optional[int], offset: int,
                   slice_size: Optional[int]):
        if identifier not in self.saved_cross_attention_maps:
            self.saved_cross_attention_maps[identifier] = {
                'dim': dim,
                'slice_size': slice_size,
                'slices': {offset or 0: slice}
            }
        else:
            self.saved_cross_attention_maps[identifier]['slices'][offset or 0] = slice

    def get_slice(self, identifier: str, requested_dim: Optional[int], requested_offset: int, slice_size: int):
        saved_attention_dict = self.saved_cross_attention_maps[identifier]
        if requested_dim is None:
            if saved_attention_dict['dim'] is not None:
                raise RuntimeError(f"dim mismatch: expected dim=None, have {saved_attention_dict['dim']}")
            return saved_attention_dict['slices'][0]

        if saved_attention_dict['dim'] == requested_dim:
            if slice_size != saved_attention_dict['slice_size']:
                raise RuntimeError(
                    f"slice_size mismatch: expected slice_size={slice_size}, have {saved_attention_dict['slice_size']}")
            return saved_attention_dict['slices'][requested_offset]

        if saved_attention_dict['dim'] is None:
            whole_saved_attention = saved_attention_dict['slices'][0]
            if requested_dim == 0:
                return whole_saved_attention[requested_offset:requested_offset + slice_size]
            elif requested_dim == 1:
                return whole_saved_attention[:, requested_offset:requested_offset + slice_size]

        raise RuntimeError(f"Cannot convert dim {saved_attention_dict['dim']} to requested dim {requested_dim}")

    def get_slicing_strategy(self, identifier: str) -> tuple[Optional[int], Optional[int]]:
        saved_attention = self.saved_cross_attention_maps.get(identifier, None)
        if saved_attention is None:
            return None, None
        return saved_attention['dim'], saved_attention['slice_size']

    def clear_requests(self, cleanup=True):
        self.tokens_cross_attention_action = Context.Action.NONE
        self.self_cross_attention_action = Context.Action.NONE
        if cleanup:
            self.saved_cross_attention_maps = {}

    def offload_saved_attention_slices_to_cpu(self):
        for key, map_dict in self.saved_cross_attention_maps.items():
            for offset, slice in map_dict['slices'].items():
                map_dict[offset] = slice.to('cpu')


def remove_cross_attention_control(model):
    remove_attention_function(model)


def setup_cross_attention_control(model, context: Context):
    """
    Inject attention parameters and functions into the passed in model to enable cross attention editing.

    :param model: The unet model to inject into.
    :param cross_attention_control_args: Arugments passeed to the CrossAttentionControl implementations
    :return: None
    """

    # adapted from init_attention_edit
    device = context.arguments.edited_conditioning.device

    # urgh. should this be hardcoded?
    max_length = 77
    # mask=1 means use base prompt attention, mask=0 means use edited prompt attention
    mask = torch.zeros(max_length)
    indices_target = torch.arange(max_length, dtype=torch.long)
    indices = torch.zeros(max_length, dtype=torch.long)
    for name, a0, a1, b0, b1 in context.arguments.edit_opcodes:
        if b0 < max_length:
            if name == "equal":# or (name == "replace" and a1 - a0 == b1 - b0):
                # these tokens have not been edited
                indices[b0:b1] = indices_target[a0:a1]
                mask[b0:b1] = 1

    context.register_cross_attention_modules(model)
    context.cross_attention_mask = mask.to(device)
    context.cross_attention_index_map = indices.to(device)
    inject_attention_function(model, context)


def get_attention_modules(model, which: CrossAttentionType):
    which_attn = "attn1" if which is CrossAttentionType.SELF else "attn2"
    return [(name,module) for name, module in model.named_modules() if
                  type(module).__name__ == "CrossAttention" and which_attn in name]


def inject_attention_function(unet, context: Context):
    # ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276

    def attention_slice_wrangler(module, suggested_attention_slice:torch.Tensor, dim, offset, slice_size):

        #memory_usage = suggested_attention_slice.element_size() * suggested_attention_slice.nelement()

        attention_slice = suggested_attention_slice

        if context.get_should_save_maps(module.identifier):
            #print(module.identifier, "saving suggested_attention_slice of shape",
            #      suggested_attention_slice.shape, "dim", dim, "offset", offset)
            slice_to_save = attention_slice.to('cpu') if dim is not None else attention_slice
            context.save_slice(module.identifier, slice_to_save, dim=dim, offset=offset, slice_size=slice_size)
        elif context.get_should_apply_saved_maps(module.identifier):
            #print(module.identifier, "applying saved attention slice for dim", dim, "offset", offset)
            saved_attention_slice = context.get_slice(module.identifier, dim, offset, slice_size)

            # slice may have been offloaded to CPU
            saved_attention_slice = saved_attention_slice.to(suggested_attention_slice.device)

            if context.is_tokens_cross_attention(module.identifier):
                index_map = context.cross_attention_index_map
                remapped_saved_attention_slice = torch.index_select(saved_attention_slice, -1, index_map)
                this_attention_slice = suggested_attention_slice

                mask = context.cross_attention_mask
                saved_mask = mask
                this_mask = 1 - mask
                attention_slice = remapped_saved_attention_slice * saved_mask + \
                                  this_attention_slice * this_mask
            else:
                # just use everything
                attention_slice = saved_attention_slice

        return attention_slice

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.identifier = name
            module.set_attention_slice_wrangler(attention_slice_wrangler)
            module.set_slicing_strategy_getter(lambda module, module_identifier=name: \
                                                   context.get_slicing_strategy(module_identifier))


def remove_attention_function(unet):
    # clear wrangler callback
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.set_attention_slice_wrangler(None)
            module.set_slicing_strategy_getter(None)
