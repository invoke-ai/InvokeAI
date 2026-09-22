"""Tests for the Ideogram 4 text-encoder load-completeness guard.

The encoder is built under accelerate.init_empty_weights() and filled from the checkpoint. A missing
non-tied weight would leave a tensor on the meta device — passing the load but failing later during
device movement / encoding. _verify_encoder_fully_materialized must reject that, while tolerating tied
weights that transformers resolves via tie_weights().
"""

import accelerate
import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.ideogram4 import _verify_encoder_fully_materialized


class _UntiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(4, 4, bias=False)
        self.extra = torch.nn.Linear(4, 4, bias=False)


class _TiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(4, 4, bias=False)
        self.head = torch.nn.Linear(4, 4, bias=False)

    def tie_weights(self) -> None:
        # Mirror transformers: the output weight is tied to (shares storage with) the input embedding.
        self.head.weight = self.embed.weight


def test_passes_when_fully_materialized() -> None:
    """A normally-initialized model (no meta tensors) is accepted."""
    _verify_encoder_fully_materialized(_UntiedModel(), context="test")


def test_raises_on_leftover_meta_from_missing_non_tied_weight() -> None:
    """A missing non-tied weight leaves a meta tensor and must be rejected."""
    with accelerate.init_empty_weights():
        model = _UntiedModel()
    # Simulate loading only `embed`; `extra` stays on the meta device (missing non-tied weight).
    model.embed.weight = torch.nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(RuntimeError, match="meta device"):
        _verify_encoder_fully_materialized(model, context="test")


def test_tolerates_tied_weight_resolved_by_tie_weights() -> None:
    """A tied output weight left on meta is materialized by tie_weights() and must not raise."""
    with accelerate.init_empty_weights():
        model = _TiedModel()
    # Only the source embedding is "loaded"; head.weight is still meta but is tied to embed.
    model.embed.weight = torch.nn.Parameter(torch.zeros(4, 4))
    _verify_encoder_fully_materialized(model, context="test")
    assert model.head.weight is model.embed.weight
    assert not model.head.weight.is_meta
