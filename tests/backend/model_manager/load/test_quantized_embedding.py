"""Embedding materialization must budget live storage and bound GGML decode temporaries."""

from itertools import chain
from unittest.mock import call, create_autospec

import gguf
import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.quantized_embedding import materialize_quantized_embedding
from invokeai.backend.quantization.gguf import loaders
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType
from tests.backend.quantization.gguf.test_loaders import _quantized_embedding

_VOCAB, _HIDDEN = 64, 32


@pytest.mark.parametrize(
    "qtype",
    [gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q6_K, gguf.GGMLQuantizationType.IQ4_NL],
    ids=lambda q: q.name,
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows_per_chunk", [1, 3])
def test_ggml_embedding_decodes_bounded_rows_bit_exactly(monkeypatch, qtype, dtype, rows_per_chunk):
    # A row wider than the element budget must still decode one complete row.
    budget = 511 if rows_per_chunk == 1 else 3 * 512
    monkeypatch.setattr(loaders, "_UNPACK_CHUNK_ELEMENTS", budget)
    embedding = torch.nn.Embedding(37, 512, device="meta")
    weight = _quantized_embedding(qtype)
    weight.compute_dtype = dtype
    embedding.weight = torch.nn.Parameter(weight, requires_grad=False)
    # Independent whole-tensor decode, before observing the production calls.
    expected = weight.get_dequantized_tensor()
    decoded = []
    real_decode = GGMLTensor.get_dequantized_tensor
    cache = create_autospec(ModelCache, instance=True)

    def recording(tensor):
        assert cache.make_room.called
        result = real_decode(tensor)
        decoded.append(result.numel())
        return result

    monkeypatch.setattr(GGMLTensor, "get_dequantized_tensor", recording)

    assert materialize_quantized_embedding(embedding, ram_cache=cache) is True

    assert decoded == ([512] * 37 if rows_per_chunk == 1 else [3 * 512] * 12 + [512])
    assert torch.equal(embedding.weight.view(torch.uint8), expected.view(torch.uint8))
    assert embedding.weight.dtype == dtype
    assert embedding.weight.device == weight.quantized_data.device
    assert not embedding.weight.requires_grad
    assert cache.make_room.call_args_list == [
        call(weight.quantized_data.nbytes + expected.nbytes + 6 * rows_per_chunk * 512 * dtype.itemsize)
    ]


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
    weight_bytes = embedding.weight.quantized_data.nbytes

    assert materialize_quantized_embedding(embedding, ram_cache=ram_cache) is True

    assert ram_cache.make_room.call_args_list == [call(weight_bytes + 7 * _VOCAB * _HIDDEN * torch.float32.itemsize)]
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


def test_an_sdnq_weight_keeps_whole_tensor_decoding_and_reserves_its_payloads(monkeypatch) -> None:
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
    weight = embedding.weight
    expected = weight.get_dequantized_tensor()
    decoded = []
    real_decode = SDNQTensor.get_dequantized_tensor

    def recording(tensor):
        decoded.append(tensor)
        return real_decode(tensor)

    monkeypatch.setattr(SDNQTensor, "get_dequantized_tensor", recording)

    assert materialize_quantized_embedding(embedding, ram_cache=ram_cache) is True

    assert ram_cache.make_room.call_args_list == [call(weight.sdnq_storage_nbytes() + 6 * expected.nbytes)]
    assert len(decoded) == 1 and decoded[0] is weight
    assert torch.equal(embedding.weight, expected)
    assert not isinstance(embedding.weight, SDNQTensor)
    assert embedding.weight.dtype is torch.bfloat16
    assert embedding(torch.tensor([[0, 1]])).shape == (1, 2, _HIDDEN)


@pytest.mark.parametrize("shared", [False, True])
def test_checkpoint_and_model_aliases_count_their_backing_storage_once(shared):
    embedding = _q8_0_embedding()
    model = torch.nn.Module()
    model.add_module("embedding", embedding)
    model.add_module("projection", torch.nn.Linear(32, 64, bias=False, device="meta"))
    projection_weight = embedding.weight.detach() if shared else embedding.weight.clone()
    model.projection.weight = torch.nn.Parameter(projection_weight, requires_grad=False)
    storage = torch.arange(100, dtype=torch.float32)
    model.register_buffer("frequencies", storage[:8])
    unused = torch.ones(79)
    checkpoint = {
        "embed.weight": embedding.weight.detach(),
        "tied.weight": embedding.weight.detach(),
        "unused": unused,
        "buffer_view": storage[8:16],
    }
    cache = create_autospec(ModelCache, instance=True)
    packed_bytes = embedding.weight.quantized_data.nbytes
    output_bytes = _VOCAB * _HIDDEN * torch.float32.itemsize

    assert materialize_quantized_embedding(
        embedding, ram_cache=cache, held_tensors=chain(checkpoint.values(), model.parameters(), model.buffers())
    )

    assert cache.make_room.call_args_list == [
        call((packed_bytes if shared else 2 * packed_bytes) + storage.nbytes + unused.nbytes + 7 * output_bytes)
    ]
    assert isinstance(model.projection.weight, GGMLTensor)
    assert isinstance(checkpoint["embed.weight"], GGMLTensor)


def test_other_live_sdnq_weights_include_all_auxiliary_storage_once():
    embedding = _q8_0_embedding()
    auxiliary = [torch.ones(64, 1), torch.zeros(64, 1), torch.ones(64, 2), torch.ones(2, 32)]
    other = SDNQTensor(
        data=torch.ones(64, 32, dtype=torch.int8),
        quantization_type=SDNQQuantizationType.INT8_ASYM,
        tensor_shape=torch.Size((64, 32)),
        compute_dtype=torch.float32,
        scale=auxiliary[0],
        zero_point=auxiliary[1],
        svd_up=auxiliary[2],
        svd_down=auxiliary[3],
    )
    cache = create_autospec(ModelCache, instance=True)
    packed_bytes = embedding.weight.quantized_data.nbytes
    held_bytes = other.quantized_data.nbytes + sum(t.nbytes for t in auxiliary)

    assert materialize_quantized_embedding(
        embedding, ram_cache=cache, held_tensors=chain((other, other.detach()), auxiliary)
    )

    assert cache.make_room.call_args_list == [
        call(packed_bytes + held_bytes + 7 * _VOCAB * _HIDDEN * torch.float32.itemsize)
    ]


@pytest.mark.parametrize(
    "qtype", [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16]
)
def test_unquantized_ggml_storage_still_materializes_in_the_compute_dtype(qtype):
    dtype = {
        gguf.GGMLQuantizationType.F32: torch.float32,
        gguf.GGMLQuantizationType.F16: torch.float16,
        gguf.GGMLQuantizationType.BF16: torch.bfloat16,
    }[qtype]
    source = torch.linspace(-1, 1, 64 * 32).reshape(64, 32).to(dtype)
    packed = source.view(torch.uint8) if qtype == gguf.GGMLQuantizationType.BF16 else source
    embedding = torch.nn.Embedding(64, 32, device="meta")
    embedding.weight = torch.nn.Parameter(
        GGMLTensor(packed, qtype, torch.Size((64, 32)), torch.float32), requires_grad=False
    )
    cache = create_autospec(ModelCache, instance=True)

    assert materialize_quantized_embedding(embedding, ram_cache=cache)

    assert torch.equal(embedding.weight, source.float())
    assert embedding.weight.dtype == torch.float32
    assert cache.make_room.call_args_list == [call(packed.nbytes + source.numel() * torch.float32.itemsize)]


@pytest.mark.parametrize("failure", ["reservation", "decode"])
def test_failed_materialization_preserves_the_original_weight(monkeypatch, failure):
    embedding = _q8_0_embedding()
    weight = embedding.weight
    cache = create_autospec(ModelCache, instance=True)
    monkeypatch.setattr(loaders, "_UNPACK_CHUNK_ELEMENTS", 3 * _HIDDEN)
    calls = []
    real_decode = GGMLTensor.get_dequantized_tensor

    def failing_decode(tensor):
        calls.append(tensor.shape.numel())
        if len(calls) == 2:
            raise RuntimeError("decode failed")
        return real_decode(tensor)

    monkeypatch.setattr(GGMLTensor, "get_dequantized_tensor", failing_decode)
    if failure == "reservation":
        cache.make_room.side_effect = RuntimeError("reservation failed")

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        materialize_quantized_embedding(embedding, ram_cache=cache)

    assert embedding.weight is weight
    assert calls == ([] if failure == "reservation" else [3 * _HIDDEN] * 2)
