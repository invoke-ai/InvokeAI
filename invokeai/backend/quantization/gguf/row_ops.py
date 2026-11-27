"""Row operations on GGUF state-dict tensors that keep GGML weights packed."""

import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


def _packed_rows(value: GGMLTensor) -> torch.Tensor:
    """The packed data of ``value``, refused unless it is stored one row per output.

    GGML blocks run along a row, never across rows, so such data can be cut by rows without
    dequantizing.
    """
    data = value.quantized_data
    if data.shape[0] != value.tensor_shape[0]:
        raise ValueError(f"GGML tensor of shape {tuple(value.tensor_shape)} is not stored one row per output")
    return data


def _with_rows(value: GGMLTensor, data: torch.Tensor) -> GGMLTensor:
    shape = torch.Size((data.shape[0], *value.tensor_shape[1:]))
    return GGMLTensor(data, value._ggml_quantization_type, shape, value.compute_dtype)


def take_rows(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather the rows -- the leading, output dimension -- of a weight or bias at ``index``.

    A ``GGMLTensor`` is gathered on its packed bytes and keeps its quantization type and compute
    dtype. Plain tensors index along dim 0.
    """
    if not isinstance(value, GGMLTensor):
        return value[index]
    return _with_rows(value, _packed_rows(value)[index])


def split_rows(value: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a tensor's leading (output) dimension in two at ``rows``.

    Plain tensors split as views. GGML halves are sliced and cloned, so each owns its storage; a
    contiguous slice copies about twice as fast as an index gather of the same rows.
    """
    if not isinstance(value, GGMLTensor):
        return value[:rows], value[rows:]
    data = _packed_rows(value)
    return _with_rows(value, data[:rows].clone()), _with_rows(value, data[rows:].clone())
