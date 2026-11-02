import torch


def calc_tensor_size(t: torch.Tensor) -> int:
    """Calculate the size of a tensor in bytes."""
    # SDNQ quantized tensors advertise the dequantized shape with a uint8 dtype, which both
    # over-counts the packed uint4/int5 storage and omits the scale/zero_point/svd payloads. Ask the
    # wrapper for its real storage. Detected by duck-typing to avoid importing the subclass (and a
    # dependency on the quantization package) into this low-level util.
    sdnq_storage_nbytes = getattr(t, "sdnq_storage_nbytes", None)
    if callable(sdnq_storage_nbytes):
        return sdnq_storage_nbytes()
    return t.nelement() * t.element_size()


def calc_tensors_size(tensors: list[torch.Tensor | None]) -> int:
    """Calculate the size of a list of tensors in bytes."""
    return sum(calc_tensor_size(t) for t in tensors if t is not None)


def module_bytes(module: torch.nn.Module) -> int:
    """A module's own parameters and buffers as stored: what partial loading copies to the device for its forward."""
    return sum(calc_tensor_size(t) for t in (*module.parameters(recurse=False), *module.buffers(recurse=False)))


def largest_module_bytes(model: torch.nn.Module) -> int:
    """The largest `module_bytes` of `model`'s modules: what partial loading copies to the device for one layer's
    forward. 0 for a model without tensors."""
    return max((module_bytes(module) for module in model.modules()), default=0)
