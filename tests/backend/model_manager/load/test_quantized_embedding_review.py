"""The reservation owns the complete, deduplicated in-flight model and checkpoint."""

from unittest.mock import create_autospec

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.quantized_embedding import materialize_quantized_embedding
from tests.backend.model_manager.load.test_quantized_embedding import _q8_0_embedding


@pytest.mark.parametrize("missing", ["model", "state_dict"])
def test_model_and_checkpoint_are_required_even_for_a_plain_embedding(missing):
    embedding = torch.nn.Embedding(2, 2)
    args = {"model": embedding, "state_dict": {}}
    del args[missing]
    with pytest.raises(TypeError, match=missing):
        materialize_quantized_embedding(embedding, ram_cache=create_autospec(ModelCache, instance=True), **args)


def test_counts_nonaliased_model_parameters_but_not_meta_storage():
    embedding = _q8_0_embedding()
    model = torch.nn.Module()
    model.embedding = embedding
    model.extra = torch.nn.Parameter(torch.ones(31), requires_grad=False)
    model.meta = torch.nn.Parameter(torch.empty(10000, device="meta"), requires_grad=False)
    model.register_buffer("buffer", torch.ones(11))
    checkpoint = {"embedding": embedding.weight.detach(), "unused": torch.ones(7)}
    cache = create_autospec(ModelCache, instance=True)
    packed = embedding.weight.quantized_data.nbytes
    output = embedding.weight.shape.numel() * torch.float32.itemsize

    assert materialize_quantized_embedding(embedding, model=model, state_dict=checkpoint, ram_cache=cache)

    cache.make_room.assert_called_once_with(
        packed + (31 + 11 + 7) * 4 + 7 * output,
        host_bytes_needed=7 * output,
    )
