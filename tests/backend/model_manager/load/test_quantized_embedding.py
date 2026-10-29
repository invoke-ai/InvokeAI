"""Materializing a quantized embedding weight, and reserving the RAM it needs first.

Five encoder loaders share this one step. The reservation is the part worth pinning: the framework
reserves a model's file size before the load, which by construction does not cover unpacking a
tensor on top of it, and getting the figure from the *packed* element count would under-reserve by
whatever the quantization saved — roughly half for a 4-bit quant.
"""

from unittest.mock import call, create_autospec

import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.quantized_embedding import materialize_quantized_embedding
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType

_VOCAB, _HIDDEN = 64, 32


def _q8_0_embedding() -> torch.nn.Embedding:
    """A real Q8_0 `GGMLTensor` on an `nn.Embedding`, as `gguf_sd_loader` hands one over."""
    import gguf
    import numpy as np
    from gguf.quants import quantize

    logical = torch.Size((_VOCAB, _HIDDEN))
    packed = torch.from_numpy(quantize(np.full(tuple(logical), 0.25, dtype=np.float32), gguf.GGMLQuantizationType.Q8_0))
    embedding = torch.nn.Embedding(_VOCAB, _HIDDEN)
    embedding.weight = torch.nn.Parameter(
        GGMLTensor(
            packed,
            ggml_quantization_type=gguf.GGMLQuantizationType.Q8_0,
            tensor_shape=logical,
            compute_dtype=torch.float32,
        ),
        requires_grad=False,
    )
    return embedding


def test_a_packed_weight_is_materialized_and_reserved_by_its_dequantized_size() -> None:
    ram_cache = create_autospec(ModelCache, instance=True)
    embedding = _q8_0_embedding()
    # The packed element count is a different number from the logical one in either direction — Q8_0
    # spends 34 bytes per 32 values, a 4-bit quant roughly half — so a reservation taken from it is
    # simply wrong, not conservative.
    assert embedding.weight.quantized_data.numel() != _VOCAB * _HIDDEN

    assert materialize_quantized_embedding(embedding, ram_cache=ram_cache) is True

    assert ram_cache.make_room.call_args_list == [call(6 * _VOCAB * _HIDDEN * torch.float32.itemsize)]
    assert not isinstance(embedding.weight, GGMLTensor)
    assert embedding.weight.shape == (_VOCAB, _HIDDEN)
    # An indexed read is the whole reason this step exists.
    assert torch.allclose(embedding(torch.tensor([[1, 2]])), torch.full((1, 2, _HIDDEN), 0.25), atol=1e-3)


def test_a_plain_weight_is_left_alone_and_reserves_nothing() -> None:
    """Callers run this unconditionally over a model whose quantization they do not know — a
    non-quantized build must not pay a reservation, which evicts other models from the cache.
    """
    ram_cache = create_autospec(ModelCache, instance=True)
    embedding = torch.nn.Embedding(_VOCAB, _HIDDEN)
    weight_before = embedding.weight

    assert materialize_quantized_embedding(embedding, ram_cache=ram_cache) is False

    assert ram_cache.make_room.call_args_list == []
    assert embedding.weight is weight_before


def test_an_sdnq_weight_is_handled_by_the_same_duck_typed_path() -> None:
    """Three of the five call sites are SDNQ, not GGUF, and `SDNQTensor` shares no base class with
    `GGMLTensor` — only `get_dequantized_tensor`, `.shape` as the dequantized shape, and
    `.compute_dtype`. That contract is what the helper dispatches on, so it is asserted against the
    real type: `compute_dtype` is a plain public attribute elsewhere reached via `hasattr`, so
    renaming it would look free and would break every SDNQ encoder load.
    """
    ram_cache = create_autospec(ModelCache, instance=True)
    embedding = torch.nn.Embedding(_VOCAB, _HIDDEN)
    embedding.weight = torch.nn.Parameter(
        SDNQTensor(
            data=torch.full((_VOCAB, _HIDDEN), 10, dtype=torch.int8),
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size((_VOCAB, _HIDDEN)),
            compute_dtype=torch.bfloat16,
            scale=torch.full((_VOCAB, 1), 0.1),
        ),
        requires_grad=False,
    )

    assert materialize_quantized_embedding(embedding, ram_cache=ram_cache) is True

    assert ram_cache.make_room.call_args_list == [call(6 * _VOCAB * _HIDDEN * torch.bfloat16.itemsize)]
    assert not isinstance(embedding.weight, SDNQTensor)
    assert embedding.weight.dtype is torch.bfloat16
    assert embedding(torch.tensor([[0, 1]])).shape == (1, 2, _HIDDEN)
