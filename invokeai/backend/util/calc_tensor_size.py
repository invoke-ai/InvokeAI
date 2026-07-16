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
