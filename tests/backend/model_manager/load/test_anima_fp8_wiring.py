"""Guards for Anima's FP8-storage wiring.

Two things have to hold for the `fp8_storage` toggle to do the right thing on Anima, and neither
is observed by any other test:

1. The loader has to actually call `_apply_fp8_layerwise_casting`. Without it the toggle renders
   in the UI and silently does nothing.
2. `AnimaTransformer._skip_layerwise_casting_patterns` has to match the modules it is meant to
   match. The patterns are plain substrings matched against dotted module paths, so a rename in
   the transformer disables the skip without breaking anything loudly.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import accelerate
import torch

from invokeai.backend.anima.anima_transformer import AnimaTransformer
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Anima_Config
from invokeai.backend.model_manager.load.load_default import (
    _FP8_DEFAULT_SKIP_PATTERNS,
    _FP8_SUPPORTED_PYTORCH_LAYERS,
)
from invokeai.backend.model_manager.load.model_loaders.anima import (
    ANIMA_TRANSFORMER_CONFIG,
    AnimaCheckpointModel,
)
from invokeai.backend.model_manager.taxonomy import SubModelType

# Every cast-eligible module in the real Anima graph that the declared patterns protect. Pinned
# exhaustively rather than as a substring check, so both a rename (entries disappear) and an
# over-broad pattern (extra entries appear) fail here.
EXPECTED_SKIPPED_MODULES = {
    "t_embedder.1.linear_1",
    "t_embedder.1.linear_2",
    "x_embedder.proj.1",
    "final_layer.linear",
    "final_layer.adaln_modulation.1",
    "final_layer.adaln_modulation.2",
}


def _cast_eligible_modules(model: torch.nn.Module) -> list[str]:
    """The dotted paths `_apply_fp8_to_nn_module` would consider for casting, before skipping."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, _FP8_SUPPORTED_PYTORCH_LAYERS) and list(module.parameters(recurse=False))
    ]


def _build_meta_transformer() -> AnimaTransformer:
    with accelerate.init_empty_weights():
        return AnimaTransformer(**ANIMA_TRANSFORMER_CONFIG)


def test_declared_skip_patterns_pin_to_real_module_paths() -> None:
    """The declared patterns must resolve against the real module graph, not just be strings.

    `t_embedder` is the load-bearing one: casting it to FP8 renders a heavily dithered image,
    because it feeds `adaln_lora` into every block. Renaming it in `AnimaTransformer` while the
    pattern list still says `t_embedder` would silently reintroduce that, so match against an
    actually-instantiated model.
    """
    model = _build_meta_transformer()
    patterns = AnimaTransformer._skip_layerwise_casting_patterns

    skipped = {name for name in _cast_eligible_modules(model) if any(re.search(p, name) for p in patterns)}

    assert skipped == EXPECTED_SKIPPED_MODULES

    # Each declared pattern earns its place — none is dead.
    for pattern in patterns:
        assert any(re.search(pattern, name) for name in skipped), f"pattern {pattern!r} matches nothing"


def test_generic_skip_patterns_do_not_cover_anima() -> None:
    """The declared list is not redundant with `_FP8_DEFAULT_SKIP_PATTERNS`.

    Those defaults are written for diffusers' module naming (`norm`, `pos_embed`, `patch_embed`,
    `proj_in/out`); this architecture names the equivalent modules differently, so they protect
    nothing here. If a future default did start covering Anima, this test failing is the signal to
    re-check whether the declared list is still needed.
    """
    model = _build_meta_transformer()

    covered_by_defaults = [
        name for name in _cast_eligible_modules(model) if any(re.search(p, name) for p in _FP8_DEFAULT_SKIP_PATTERNS)
    ]

    assert covered_by_defaults == []


class _TinyAnimaTransformer(torch.nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2))


def test_single_file_loader_applies_fp8_layerwise_casting(monkeypatch, tmp_path) -> None:
    """Regression guard for the `fp8_storage` toggle being wired up at all.

    Deleting the `_apply_fp8_layerwise_casting` call from `_load_from_singlefile` leaves the whole
    `tests/backend/model_manager` and `tests/backend/anima` suites green — nothing else observes
    it — so the dead toggle can come straight back with CI passing.
    """
    import safetensors.torch

    import invokeai.backend.anima.anima_transformer as anima_transformer_module

    checkpoint_path = tmp_path / "anima.safetensors"
    checkpoint_path.touch()
    config = Main_Checkpoint_Anima_Config.model_construct(path=str(checkpoint_path), fp8_storage=True)

    monkeypatch.setattr(anima_transformer_module, "AnimaTransformer", _TinyAnimaTransformer)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda _path: {"weight": torch.ones(2, 2)})
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.TorchDevice.choose_anima_inference_dtype",
        lambda _device: torch.float32,
    )

    cast_calls: list[tuple[object, object]] = []

    def _record_cast(model, cfg, submodel):
        cast_calls.append((cfg, submodel))
        return model

    loader = object.__new__(AnimaCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._apply_fp8_layerwise_casting = _record_cast

    model = loader._load_from_singlefile(config)

    assert isinstance(model, _TinyAnimaTransformer)
    assert torch.equal(model.weight, torch.ones(2, 2))
    assert cast_calls == [(config, SubModelType.Transformer)]
