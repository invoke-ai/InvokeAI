"""`lora_sidecar_bytes` for a layer wrapped to swap its halves, measured against what its sidecar forward allocates.

FLUX's output modulation in BFL's row order is the wrapped case (`SwappedHalvesLayerPatch`). The unwrapped kinds are
measured through the Qwen-Image-2.1 denoise node, which prices with the same helper
(`test_qwen_image_2_1_nodes.py::test_a_sidecar_reserves_what_its_forward_allocates`).
"""

import itertools
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.patches.sidecar_working_memory import lora_sidecar_bytes

PREFIX = "lora_transformer-"
_W = 64  # A Linear wide enough that only weight-sized tensors count against half a copy of its weight.


class _PeakAllocation(TorchDispatchMode):
    """The most bytes that tensors made while `counting` is on -- and views of them -- hold at once."""

    def __init__(self) -> None:
        super().__init__()
        self.counting = True
        self.peak = 0
        self._made: set[int] = set()
        self._tensors: weakref.WeakValueDictionary[int, torch.Tensor] = weakref.WeakValueDictionary()
        self._ids = itertools.count()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        inputs = {t.untyped_storage().data_ptr() for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)}
        out = func(*args, **(kwargs or {}))
        if self.counting:
            for tensor in (t for t in tree_leaves(out) if isinstance(t, torch.Tensor)):
                storage = tensor.untyped_storage().data_ptr()
                if storage not in inputs:
                    self._made.add(storage)
                # A view keeps its storage alive after the tensor it was taken from is gone.
                if storage in self._made:
                    self._tensors[next(self._ids)] = tensor
            live = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in self._tensors.values()}
            self.peak = max(self.peak, sum(live.values()))
        return out


def _swapped_layer(kind: str) -> ModelPatchRaw:
    """One layer of `kind` on the modulation, wrapped to swap its halves."""
    r = 2
    values = {
        "low-rank": {"lora_down.weight": torch.randn(r, _W), "lora_up.weight": torch.randn(_W, r)},
        "full": {"diff": torch.randn(_W, _W)},
        "lokr": {
            "lokr_w1": torch.randn(4, 4),
            "lokr_w2_a": torch.randn(_W // 4, r),
            "lokr_w2_b": torch.randn(r, _W // 4),
        },
        "loha": {
            f"hada_w{i}_{s}": torch.randn(_W, r) if s == "a" else torch.randn(r, _W) for i in (1, 2) for s in "ab"
        },
        "dora": {
            "lora_down.weight": torch.randn(r, _W),
            "lora_up.weight": torch.randn(_W, r),
            "dora_magnitude": torch.rand(_W) + 1,
        },
    }[kind]
    return ModelPatchRaw(
        layers={PREFIX + "norm_out.linear": SwappedHalvesLayerPatch(any_lora_layer_from_state_dict(values))}
    )


@pytest.mark.parametrize(
    "kinds",
    [["low-rank"], ["full"], ["lokr"], ["loha"], ["dora"], ["low-rank", "low-rank"], ["low-rank", "low-rank", "lokr"]],
    ids=lambda kinds: "+".join(kinds),
)
def test_a_swapped_halves_sidecar_reserves_what_its_forward_allocates(
    kinds: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured: what a sidecar forward on an fp8 weight allocates for its LoRAs -- the layer's own forward aside, which
    an activation estimate covers -- is what is reserved for them, within half a weight copy. A wrapped layer forms its
    update whatever its kind and swaps the weight it reads and the update it returns; DoRA is priced a copy above the
    CPU's measure for the column norm CUDA adds."""
    torch.manual_seed(0)
    transformer = torch.nn.Module()
    transformer.norm_out = torch.nn.Module()
    transformer.norm_out.linear = torch.nn.Linear(_W, _W, bias=False)
    weight = torch.randn(_W, _W).to(torch.float8_e4m3fn)
    transformer.norm_out.linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    apply_custom_layers_to_model(transformer)
    linear = transformer.norm_out.linear
    patches = [_swapped_layer(kind) for kind in kinds]
    seq_len = 2

    mode = _PeakAllocation()
    own_forward = linear._autocast_forward

    def uncounted_forward(x: torch.Tensor) -> torch.Tensor:
        mode.counting = False
        try:
            return own_forward(x)
        finally:
            mode.counting = True

    monkeypatch.setattr(linear, "_autocast_forward", uncounted_forward)
    with (
        torch.no_grad(),
        LayerPatcher.apply_smart_model_patches(
            model=transformer, patches=[(patch, 1.0) for patch in patches], prefix=PREFIX, dtype=torch.float32
        ),
    ):
        with mode:
            linear(torch.randn(1, seq_len, _W))

    # fp8 weights take every layer as a sidecar, whatever the format says. The layers' own tensors exist before the
    # measured forward.
    reserved = lora_sidecar_bytes(
        patches,
        transformer,
        ModelFormat.Checkpoint,
        prefix=PREFIX,
        rows=seq_len,
        per_sample_rows={},
        dtype=torch.float32,
    ) - sum(patch.calc_size() for patch in patches)
    weight_copy = _W * _W * torch.float32.itemsize
    beyond_cpu = weight_copy if "dora" in kinds else 0
    assert abs(reserved - beyond_cpu - mode.peak) < weight_copy / 2, (reserved / weight_copy, mode.peak / weight_copy)
