import enum
import math
from typing import Optional, Callable

import psutil
import torch
import diffusers
from torch import nn
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from ldm.invoke.devices import torch_dtype

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
        for name,module in get_cross_attention_modules(model, CrossAttentionType.SELF):
            if name in self.self_cross_attention_module_identifiers:
                assert False, f"name {name} cannot appear more than once"
            self.self_cross_attention_module_identifiers.append(name)
        for name,module in get_cross_attention_modules(model, CrossAttentionType.TOKENS):
            if name in self.tokens_cross_attention_module_identifiers:
                assert False, f"name {name} cannot appear more than once"
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



class InvokeAICrossAttentionMixin:
    """
    Enable InvokeAI-flavoured CrossAttention calculation, which does aggressive low-memory slicing and calls
    through both to an attention_slice_wrangler and a slicing_strategy_getter for custom attention map wrangling
    and dymamic slicing strategy selection.
    """
    def __init__(self):
        self.mem_total_gb = psutil.virtual_memory().total // (1 << 30)
        self.attention_slice_wrangler = None
        self.slicing_strategy_getter = None
        self.attention_slice_calculated_callback = None

    def set_attention_slice_wrangler(self, wrangler: Optional[Callable[[nn.Module, torch.Tensor, int, int, int], torch.Tensor]]):
        '''
        Set custom attention calculator to be called when attention is calculated
        :param wrangler: Callback, with args (module, suggested_attention_slice, dim, offset, slice_size),
        which returns either the suggested_attention_slice or an adjusted equivalent.
            `module` is the current CrossAttention module for which the callback is being invoked.
            `suggested_attention_slice` is the default-calculated attention slice
            `dim` is -1 if the attenion map has not been sliced, or 0 or 1 for dimension-0 or dimension-1 slicing.
                If `dim` is >= 0, `offset` and `slice_size` specify the slice start and length.

        Pass None to use the default attention calculation.
        :return:
        '''
        self.attention_slice_wrangler = wrangler

    def set_slicing_strategy_getter(self, getter: Optional[Callable[[nn.Module], tuple[int,int]]]):
        self.slicing_strategy_getter = getter

    def set_attention_slice_calculated_callback(self, callback: Optional[Callable[[torch.Tensor], None]]):
        self.attention_slice_calculated_callback = callback

    def einsum_lowest_level(self, query, key, value, dim, offset, slice_size):
        # calculate attention scores
        #attention_scores = torch.einsum('b i d, b j d -> b i j', q, k)
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        # calculate attention slice by taking the best scores for each latent pixel
        default_attention_slice = attention_scores.softmax(dim=-1, dtype=attention_scores.dtype)
        attention_slice_wrangler = self.attention_slice_wrangler
        if attention_slice_wrangler is not None:
            attention_slice = attention_slice_wrangler(self, default_attention_slice, dim, offset, slice_size)
        else:
            attention_slice = default_attention_slice

        if self.attention_slice_calculated_callback is not None:
            self.attention_slice_calculated_callback(attention_slice, dim, offset, slice_size)

        hidden_states = torch.bmm(attention_slice, value)
        return hidden_states

    def einsum_op_slice_dim0(self, q, k, v, slice_size):
        r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[0], slice_size):
            end = i + slice_size
            r[i:end] = self.einsum_lowest_level(q[i:end], k[i:end], v[i:end], dim=0, offset=i, slice_size=slice_size)
        return r

    def einsum_op_slice_dim1(self, q, k, v, slice_size):
        r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            r[:, i:end] = self.einsum_lowest_level(q[:, i:end], k, v, dim=1, offset=i, slice_size=slice_size)
        return r

    def einsum_op_mps_v1(self, q, k, v):
        if q.shape[1] <= 4096: # (512x512) max q.shape[1]: 4096
            return self.einsum_lowest_level(q, k, v, None, None, None)
        else:
            slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
            return self.einsum_op_slice_dim1(q, k, v, slice_size)

    def einsum_op_mps_v2(self, q, k, v):
        if self.mem_total_gb > 8 and q.shape[1] <= 4096:
            return self.einsum_lowest_level(q, k, v, None, None, None)
        else:
            return self.einsum_op_slice_dim0(q, k, v, 1)

    def einsum_op_tensor_mem(self, q, k, v, max_tensor_mb):
        size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
        if size_mb <= max_tensor_mb:
            return self.einsum_lowest_level(q, k, v, None, None, None)
        div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
        if div <= q.shape[0]:
            return self.einsum_op_slice_dim0(q, k, v, q.shape[0] // div)
        return self.einsum_op_slice_dim1(q, k, v, max(q.shape[1] // div, 1))

    def einsum_op_cuda(self, q, k, v):
        # check if we already have a slicing strategy (this should only happen during cross-attention controlled generation)
        slicing_strategy_getter = self.slicing_strategy_getter
        if slicing_strategy_getter is not None:
            (dim, slice_size) = slicing_strategy_getter(self)
            if dim is not None:
                # print("using saved slicing strategy with dim", dim, "slice size", slice_size)
                if dim == 0:
                    return self.einsum_op_slice_dim0(q, k, v, slice_size)
                elif dim == 1:
                    return self.einsum_op_slice_dim1(q, k, v, slice_size)

        # fallback for when there is no saved strategy, or saved strategy does not slice
        mem_free_total = get_mem_free_total(q.device)
        # Divide factor of safety as there's copying and fragmentation
        return self.einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20))


    def get_invokeai_attention_mem_efficient(self, q, k, v):
        if q.device.type == 'cuda':
            #print("in get_attention_mem_efficient with q shape", q.shape, ", k shape", k.shape, ", free memory is", get_mem_free_total(q.device))
            return self.einsum_op_cuda(q, k, v)

        if q.device.type == 'mps' or q.device.type == 'cpu':
            if self.mem_total_gb >= 32:
                return self.einsum_op_mps_v1(q, k, v)
            return self.einsum_op_mps_v2(q, k, v)

        # Smaller slices are faster due to L2/L3/SLC caches.
        # Tested on i7 with 8MB L3 cache.
        return self.einsum_op_tensor_mem(q, k, v, 32)



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
    indices = torch.arange(max_length, dtype=torch.long)
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


def get_cross_attention_modules(model, which: CrossAttentionType) -> list[tuple[str, InvokeAICrossAttentionMixin]]:
    from ldm.modules.attention import CrossAttention # avoid circular import
    cross_attention_class: type = InvokeAIDiffusersCrossAttention if isinstance(model,UNet2DConditionModel) else CrossAttention
    which_attn = "attn1" if which is CrossAttentionType.SELF else "attn2"
    attention_module_tuples = [(name,module) for name, module in model.named_modules() if
                isinstance(module, cross_attention_class) and which_attn in name]
    cross_attention_modules_in_model_count = len(attention_module_tuples)
    expected_count = 16
    if cross_attention_modules_in_model_count != expected_count:
        # non-fatal error but .swap() won't work.
        print(f"Error! CrossAttentionControl found an unexpected number of {cross_attention_class} modules in the model " +
              f"(expected {expected_count}, found {cross_attention_modules_in_model_count}). Either monkey-patching failed " +
              f"or some assumption has changed about the structure of the model itself. Please fix the monkey-patching, " +
              f"and/or update the {expected_count} above to an appropriate number, and/or find and inform someone who knows " +
              f"what it means. This error is non-fatal, but it is likely that .swap() and attention map display will not " +
              f"work properly until it is fixed.")
    return attention_module_tuples


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

                mask = context.cross_attention_mask.to(torch_dtype(suggested_attention_slice.device))
                saved_mask = mask
                this_mask = 1 - mask
                attention_slice = remapped_saved_attention_slice * saved_mask + \
                                  this_attention_slice * this_mask
            else:
                # just use everything
                attention_slice = saved_attention_slice

        return attention_slice

    cross_attention_modules = get_cross_attention_modules(unet, CrossAttentionType.TOKENS) + get_cross_attention_modules(unet, CrossAttentionType.SELF)
    for identifier, module in cross_attention_modules:
        module.identifier = identifier
        try:
            module.set_attention_slice_wrangler(attention_slice_wrangler)
            module.set_slicing_strategy_getter(
                lambda module: context.get_slicing_strategy(identifier)
            )
        except AttributeError as e:
            if is_attribute_error_about(e, 'set_attention_slice_wrangler'):
                print(f"TODO: implement set_attention_slice_wrangler for {type(module)}")  # TODO
            else:
                raise


def remove_attention_function(unet):
    cross_attention_modules = get_cross_attention_modules(unet, CrossAttentionType.TOKENS) + get_cross_attention_modules(unet, CrossAttentionType.SELF)
    for identifier, module in cross_attention_modules:
        try:
            # clear wrangler callback
            module.set_attention_slice_wrangler(None)
            module.set_slicing_strategy_getter(None)
        except AttributeError as e:
            if is_attribute_error_about(e, 'set_attention_slice_wrangler'):
                print(f"TODO: implement set_attention_slice_wrangler for {type(module)}")
            else:
                raise


def is_attribute_error_about(error: AttributeError, attribute: str):
    if hasattr(error, 'name'):  # Python 3.10
        return error.name == attribute
    else:  # Python 3.9
        return attribute in str(error)



def get_mem_free_total(device):
    #only on cuda
    if not torch.cuda.is_available():
        return None
    stats = torch.cuda.memory_stats(device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(device)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    return mem_free_total


class InvokeAIDiffusersCrossAttention(diffusers.models.attention.CrossAttention, InvokeAICrossAttentionMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        InvokeAICrossAttentionMixin.__init__(self)

    def _attention(self, query, key, value, attention_mask=None):
        #default_result = super()._attention(query,  key, value)
        if attention_mask is not None:
            print(f"{type(self).__name__} ignoring passed-in attention_mask")
        attention_result = self.get_invokeai_attention_mem_efficient(query, key, value)

        hidden_states = self.reshape_batch_dim_to_heads(attention_result)
        return hidden_states

