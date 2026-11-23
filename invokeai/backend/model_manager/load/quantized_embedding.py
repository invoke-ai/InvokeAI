"""Materialize indexed embedding weights within an absolute host RAM reservation.

``nn.Embedding`` indexes its weight directly, which neither GGML nor SDNQ serves in packed form.
Gemma-2's encoder and Flux's T5/CLIP still materialize embeddings without a reservation; their
load paths do not pass a cache here.
"""

from collections.abc import Iterable, Mapping
from itertools import chain
from typing import Any

import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.quantization.gguf.ggml_tensor import HOST_DEQUANT_PEAK_COPIES, GGMLTensor
from invokeai.backend.quantization.gguf.loaders import ggml_unpack_temp_bytes, unpack_ggml_tensor
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor


def materialize_quantized_embedding(
    embedding: torch.nn.Module, *, model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor], ram_cache: ModelCache
) -> bool:
    """Replace ``embedding.weight`` with a dequantized ``Parameter``, reserving the room first.

    Does nothing and returns False when the weight is already a plain tensor, so a caller can run it
    unconditionally over a model whose quantization it does not know.

    The model parameters, buffers, checkpoint values and embedding are all counted. Shared backing
    storage is counted once, including checkpoint aliases retained by assign=True and weights the
    model did not consume. Meta tensors do not hold storage.
    """
    weight: Any = embedding.weight
    dequantize = getattr(weight, "get_dequantized_tensor", None)
    if dequantize is None:
        return False

    held_bytes = _held_tensor_bytes(chain((weight,), state_dict.values(), model.parameters(), model.buffers()))
    # `.shape` is logical on both wrappers; `Tensor.numel()` instead reports packed elements.
    output_bytes = weight.shape.numel() * weight.compute_dtype.itemsize
    if isinstance(weight, GGMLTensor):
        # Some unquantized GGML decodes return a view; reserving an output is conservative there.
        ram_cache.make_room(held_bytes + output_bytes + ggml_unpack_temp_bytes(weight))
        result = unpack_ggml_tensor(weight)
    else:
        # SDNQ retains whole-tensor decoding. Its peak allowance already includes the result.
        ram_cache.make_room(held_bytes + HOST_DEQUANT_PEAK_COPIES * output_bytes)
        result = dequantize()
    # Keep the original Parameter intact if reserving or decoding raises.
    embedding.weight = torch.nn.Parameter(result, requires_grad=False)
    return True


def _held_tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    """Live backing storage, including quantized payloads and storage retained by tensor views."""
    seen: set[torch.UntypedStorage] = set()
    total = 0
    for tensor in tensors:
        if isinstance(tensor, SDNQTensor):
            payloads = tensor.storage_tensors()
        elif isinstance(tensor, GGMLTensor):
            payloads = (tensor.quantized_data,)
        else:
            payloads = (tensor,)
        for payload in payloads:
            if payload is None or payload.is_meta:
                continue
            storage = payload.untyped_storage()
            if storage not in seen:
                seen.add(storage)
                total += storage.nbytes()
    return total
