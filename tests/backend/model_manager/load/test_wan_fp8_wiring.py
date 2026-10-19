"""Guards for Wan's FP8-storage wiring, in both loaders.

Nothing else observes it. The registry walk in `test_fp8_capability.py` reads source and would still
see the call after the branch around it was inverted, and the whole `tests/backend/model_manager` suite
stays green with the cast deleted -- which is how `WanDiffusersModel` came to override
`GenericDiffusersLoader._load_model`, drop the cast with it, and go on offering the control for years.

Three behaviours are pinned here, and each of them is a decision this loader makes rather than
machinery it inherits:

- a plain checkpoint reaches the cast;
- a Comfy `fp8_scaled` checkpoint does *not*, and says why. Its per-tensor scales were folded into the
  weights and dropped, so re-encoding the result as unscaled fp8 would flush the low end of every
  tensor to zero -- 3.4% of the weights on `flux-2-klein-4b-fp8`. Identification switches FP8 Storage
  on by itself for a float8 denoiser, so this is the default path for such a file, not a road less
  travelled;
- the state dict is released first, or the bf16 originals stay reachable beside their fp8 copies while
  the reservation covers only one of the two.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.load.model_loaders.wan import WanCheckpointModel, WanDiffusersModel
from invokeai.backend.model_manager.taxonomy import SubModelType, WanVariantType

# Same tiny-but-faithful transformer the sibling loader tests use; `attention_head_dim` must stay 128,
# because the loader derives `num_attention_heads` from it.
TINY_MODEL_KWARGS = {
    "patch_size": (1, 2, 2),
    "in_channels": 16,
    "out_channels": 16,
    "num_layers": 2,
    "attention_head_dim": 128,
    "num_attention_heads": 1,
    "ffn_dim": 64,
    "text_dim": 32,
}


def _tiny_state_dict() -> dict[str, torch.Tensor]:
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel(**TINY_MODEL_KWARGS).state_dict()


def _checkpoint(path: Path, scaled: bool) -> Path:
    sd = _tiny_state_dict()
    if scaled:
        # What Comfy's `fp8_scaled` repack looks like: a per-tensor scale beside the weight, and the
        # marker tensor that names the convention.
        sd["blocks.0.attn1.to_q.scale_weight"] = torch.tensor([4.0])
        sd["scaled_fp8"] = torch.zeros(1, dtype=torch.float8_e4m3fn)
    save_file(sd, path)
    return path


def _loader(cast_calls: list, *, asks_for_fp8: bool = True) -> WanCheckpointModel:
    loader = object.__new__(WanCheckpointModel)
    loader._ram_cache = MagicMock()
    # The branch under test consults the gate to decide whether the decline is worth reporting; the
    # gate itself has its own cells in `test_load_default_fp8.py`.
    loader._should_use_fp8 = lambda *_args, **_kwargs: asks_for_fp8
    loader._apply_fp8_layerwise_casting = lambda model, config, submodel: (
        cast_calls.append((config, submodel)),
        model,
    )[1]
    return loader


def _load(loader: WanCheckpointModel, path: Path):
    config = MagicMock()
    config.path = str(path)
    config.variant = WanVariantType.T2V_A14B

    with (
        patch("invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_torch_device"),
        patch(
            "invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.bfloat16,
        ),
    ):
        return loader._load_from_singlefile(config)


class TestTheCheckpointLoader:
    def test_a_plain_checkpoint_reaches_the_cast(self, tmp_path: Path) -> None:
        """Nothing in this file's weights stops FP8 Storage doing exactly what it promises here."""
        cast_calls: list = []
        loader = _loader(cast_calls)

        _load(loader, _checkpoint(tmp_path / "wan2.2-t2v-a14b-high_noise.safetensors", scaled=False))

        assert len(cast_calls) == 1
        assert cast_calls[0][1] is SubModelType.Transformer

    def test_a_scaled_checkpoint_is_declined_and_the_reason_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The case identification turns on by itself, and the one where the cast would cost quality.

        Folding happened before this point and the scales are gone, so the cast could only re-encode
        the result unscaled. Declining leaves the model where it already is -- correct, at bf16 size.
        """
        cast_calls: list = []
        loader = _loader(cast_calls)

        with caplog.at_level("INFO"):
            model = _load(loader, _checkpoint(tmp_path / "Wan2.2-A14B-HighNoise-fp8_scaled.safetensors", scaled=True))

        assert cast_calls == []
        assert model is not None
        assert "FP8 Storage not applied" in caplog.text
        assert "unscaled fp8" in caplog.text

    def test_the_decline_is_silent_when_nobody_asked(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Most loads of a scaled file have the setting off, and they have nothing to be told."""
        cast_calls: list = []
        loader = _loader(cast_calls, asks_for_fp8=False)

        with caplog.at_level("INFO"):
            _load(loader, _checkpoint(tmp_path / "Wan2.2-A14B-HighNoise-fp8_scaled.safetensors", scaled=True))

        assert cast_calls == []
        assert "FP8 Storage" not in caplog.text

    def test_the_state_dict_is_released_before_the_cast(self, tmp_path: Path) -> None:
        """Peak RAM must not overshoot the `make_room()` reservation.

        `load_state_dict(..., assign=True)` aliases every parameter to its state-dict tensor, so a dict
        still holding those references means each bf16 original stays reachable while its fp8 copy is
        allocated -- against a reservation that counted the bf16 size once. `ltx2.py` carries the same
        `sd.clear()` for the same reason.
        """
        observed: list[int] = []
        loader = object.__new__(WanCheckpointModel)
        loader._ram_cache = MagicMock()
        loader._should_use_fp8 = lambda *_args, **_kwargs: True

        captured: dict[str, dict] = {}

        def record_and_check(model, _config, _submodel):
            observed.append(len(captured["sd"]))
            return model

        loader._apply_fp8_layerwise_casting = record_and_check

        real_load = torch.nn.Module.load_state_dict

        def capture(self, state_dict, *args, **kwargs):
            captured["sd"] = state_dict
            return real_load(self, state_dict, *args, **kwargs)

        with patch.object(torch.nn.Module, "load_state_dict", capture):
            _load(loader, _checkpoint(tmp_path / "wan2.2-t2v-a14b-high_noise.safetensors", scaled=False))

        assert observed == [0], "the loader still held every bf16 tensor while the cast allocated fp8 copies"


class TestTheDiffusersLoader:
    def test_a_diffusers_folder_reaches_the_cast(self, tmp_path: Path) -> None:
        """`WanDiffusersModel` overrides `_load_model` to force bfloat16, and in doing so replaced the
        parent's whole method -- including `GenericDiffusersLoader`'s cast. The control was offered for
        these folders and did nothing at all, which no test noticed because the inherited call was still
        there to be found in the parent's source."""
        cast_calls: list = []
        loader = object.__new__(WanDiffusersModel)
        loader._apply_fp8_layerwise_casting = lambda model, config, submodel: (
            cast_calls.append((config, submodel)),
            model,
        )[1]

        transformer = object()
        load_class = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: transformer)
        loader.get_hf_load_class = lambda *_args, **_kwargs: load_class

        config = SimpleNamespace(path=str(tmp_path), repo_variant=None)
        result = loader._load_model(config, SubModelType.Transformer)

        assert result is transformer
        assert cast_calls == [(config, SubModelType.Transformer)]
