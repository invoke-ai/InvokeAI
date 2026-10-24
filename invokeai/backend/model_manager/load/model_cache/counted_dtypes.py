"""Returning weights a holder cast in place to the dtypes the model cache counts them in."""

import torch


@torch.no_grad()
def return_to_counted_dtypes(
    model: torch.nn.Module,
    counted_dtypes: dict[str, torch.dtype],
    alias_groups: dict[str, tuple[str, ...]],
    cpu_state_dict: dict[str, torch.Tensor] | None,
) -> None:
    """Return every weight of `model` whose dtype differs from `counted_dtypes` to that dtype, on the device it is on.

    `counted_dtypes` maps each state dict key to the dtype the cache counted it in, and `alias_groups` names the keys of
    each tied weight (see `tensor_aliases`). A holder may cast the model it holds (the SD VAE nodes upcast the VAE to
    fp32), and `Module.to(dtype)` casts each name of a tied weight on its own, so the cast unties it. Each group gets one
    tensor again under all of its names; on the CPU that is the RAM copy in `cpu_state_dict` when there is one.

    The tensors are re-pointed through `Parameter.data` and `Module._buffers`, never `load_state_dict(assign=True)`:
    assignment goes through `register_parameter`, which a construction on another thread
    (`accelerate.init_empty_weights` patches it process-wide) would route to the meta device, and this runs without
    MODEL_LOAD_LOCK. Parameter objects and devices stay as they are, and one weight at a time holds both dtypes. Packed
    tensors (GGUF, SDNQ) refuse a dtype change, so a cast never changed theirs and nothing here selects them.
    """
    restored: dict[str, torch.Tensor] = {}
    # A module registered under two names is visited under both, as `state_dict()` lists it under both.
    for module_name, module in model.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for slots in (module._parameters, module._buffers):
            for name, tensor in slots.items():
                key = prefix + name
                counted_dtype = counted_dtypes.get(key)
                # Keys outside the state dict (non-persistent buffers) are not counted.
                if tensor is None or counted_dtype is None or tensor.dtype == counted_dtype:
                    continue
                group = alias_groups.get(key, (key,))[0]
                new = restored.get(group)
                if new is None:
                    if tensor.device.type == "cpu" and cpu_state_dict is not None:
                        new = cpu_state_dict[key]
                    else:
                        new = tensor.to(dtype=counted_dtype)
                    restored[group] = new
                if isinstance(tensor, torch.nn.Parameter):
                    tensor.data = new
                else:
                    slots[name] = new
