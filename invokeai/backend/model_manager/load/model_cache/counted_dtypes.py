"""Returning weights a holder cast in place to the dtypes the model cache counts them in."""

import torch

_PLAIN_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _shares_storage(tensor: torch.Tensor, other: torch.Tensor) -> bool:
    """Whether `tensor` already holds the memory of `other`.

    A storage keeps its Python object while it lives, so identity is exact; were it not kept, the answer would cost a
    needless re-point, never a missed one. Packed tensor subclasses (GGUF, SDNQ) answer True: their storage says nothing
    about the packed data, and a cast never changes them.
    """
    if type(tensor) not in _PLAIN_TENSOR_TYPES or type(other) not in _PLAIN_TENSOR_TYPES:
        return True
    return tensor.untyped_storage() is other.untyped_storage()


@torch.no_grad()
def return_to_counted_dtypes(
    model: torch.nn.Module,
    counted_dtypes: dict[str, torch.dtype],
    alias_groups: dict[str, tuple[str, ...]],
    cpu_state_dict: dict[str, torch.Tensor] | None,
) -> None:
    """Return every weight of `model` to the memory the cache counts for it: its counted dtype, one tensor for all names
    of a tied weight, and on the CPU the RAM copy in `cpu_state_dict` when there is one.

    `counted_dtypes` maps each state dict key to the dtype the cache counted it in, and `alias_groups` names the keys of
    each tied weight (see `tensor_aliases`). A holder may cast the model it holds (the SD VAE nodes upcast the VAE to
    fp32), and `Module.to(dtype)` gives each `Parameter` its own cast copy. The cache's moves and the multi-GPU adoption
    of another device's RAM copy re-point weights with `load_state_dict(assign=True)`, which gives each name of a tied
    weight its own `Parameter` over the one tensor, so a cast unties it. A cast back to the counted dtype leaves it
    untied, and leaves a CPU weight beside its RAM copy: the dtype alone does not show a weight is back, so every name of
    a tied weight and every CPU weight with a RAM copy is checked against the tensor it should hold.

    The tensors are re-pointed through `Parameter.data` and `Module._buffers`, never `load_state_dict(assign=True)`:
    assignment goes through `register_parameter`, which a construction on another thread
    (`accelerate.init_empty_weights` patches it process-wide) would route to the meta device, and this runs without
    MODEL_LOAD_LOCK. Parameter objects stay as they are, and one weight at a time holds both dtypes. Every weight keeps
    its device, except that the names of a tied weight found on different devices all get the tensor of the first name
    visited, on its device; the cache moves a group as a unit, so that takes a group someone else split. Packed tensors
    (GGUF, SDNQ) refuse a dtype change, so a cast never changed theirs and nothing here re-points them.
    """
    # The tensor all names of a tied weight hold after the restore, keyed by the group's first key.
    group_tensors: dict[str, torch.Tensor] = {}
    # A module registered under two names is visited under both, as `state_dict()` lists it under both.
    for module_name, module in model.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for slots in (module._parameters, module._buffers):
            for name, tensor in slots.items():
                key = prefix + name
                counted_dtype = counted_dtypes.get(key)
                # Keys outside the state dict (non-persistent buffers) are not counted.
                if tensor is None or counted_dtype is None:
                    continue
                group = alias_groups.get(key)
                new = group_tensors.get(group[0]) if group is not None else None
                if new is None:
                    if tensor.device.type == "cpu" and cpu_state_dict is not None:
                        new = cpu_state_dict[key]
                    elif tensor.dtype != counted_dtype:
                        new = tensor.to(dtype=counted_dtype)
                    elif group is not None:
                        new = tensor.detach()
                    else:
                        continue
                    if group is not None:
                        group_tensors[group[0]] = new
                if tensor.dtype == new.dtype and _shares_storage(tensor, new):
                    continue
                if isinstance(tensor, torch.nn.Parameter):
                    tensor.data = new
                else:
                    slots[name] = new
