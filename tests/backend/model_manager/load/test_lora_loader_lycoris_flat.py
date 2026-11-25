"""The loader's dispatch for SimpleTuner's LyCORIS flat-key LoRAs (issue #10174).

The flat path is entered only when `lycoris_flat_family` claims the state dict; everything else falls through to
the existing per-base dispatch, and SD 3's own flat spelling (`sd3_lokr_simpletuner_format`) has to keep going to
`lora_model_from_sd3_state_dict`. A base registered against a disagreeing file raises rather than mapping layers
onto the wrong model.
"""

from logging import Logger
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import invokeai.backend.model_manager.load.model_loaders.lora as lora_loader_module
from invokeai.backend.model_manager.load.model_loaders.lora import LoRALoader
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    flux1_lycoris_flat_lokr_format,
    sd3_lokr_simpletuner_format,
    sd15_lycoris_flat_lokr_format,
)


class _MockPatch:
    """Just enough of a `ModelPatchRaw` for the loader's tail (`to(dtype=...)`)."""

    def to(self, *_args, **_kwargs):
        return self


def _shape(value) -> list[int]:
    return list(value[0]) if isinstance(value, tuple) else list(value)


def _state_dict(keys: dict) -> dict[str, torch.Tensor]:
    return {key: torch.zeros((), dtype=torch.float32).expand(_shape(value)) for key, value in keys.items()}


def _loader(base: BaseModelType, state_dict: dict[str, torch.Tensor], monkeypatch) -> LoRALoader:
    monkeypatch.setattr(lora_loader_module, "load_file", lambda *_args, **_kwargs: state_dict)
    loader = LoRALoader.__new__(LoRALoader)
    loader._logger = MagicMock(spec=Logger)
    loader._model_base = base
    loader._torch_dtype = torch.float32
    return loader


def _config(tmp_path, base: BaseModelType) -> SimpleNamespace:
    return SimpleNamespace(
        path=tmp_path / "lycoris-flat.safetensors",
        name="lycoris-flat",
        base=base,
        format=ModelFormat.LyCORIS,
    )


def test_a_flat_file_registered_under_another_base_raises(tmp_path, monkeypatch) -> None:
    """A FLUX.1 flat file registered as SDXL by an earlier version must not patch SDXL modules."""
    state_dict = _state_dict(flux1_lycoris_flat_lokr_format.state_dict_keys)
    loader = _loader(BaseModelType.StableDiffusionXL, state_dict, monkeypatch)

    with pytest.raises(ValueError, match="keys are those of a flux LoRA"):
        loader._load_model(_config(tmp_path, BaseModelType.StableDiffusionXL))  # type: ignore[arg-type]


def test_the_sd3_flat_spelling_still_goes_through_the_sd3_converter(tmp_path, monkeypatch) -> None:
    """SD 3's `lycoris_` layout is claimed by no family here: it reaches its own converter, untouched."""
    routed: list[dict] = []

    def _spy(*, state_dict):
        routed.append(state_dict)
        return _MockPatch()

    monkeypatch.setattr(lora_loader_module, "lora_model_from_sd3_state_dict", _spy)
    state_dict = _state_dict(sd3_lokr_simpletuner_format.state_dict_keys)
    loader = _loader(BaseModelType.StableDiffusion3, state_dict, monkeypatch)

    loader._load_model(_config(tmp_path, BaseModelType.StableDiffusion3))  # type: ignore[arg-type]

    assert len(routed) == 1


def test_an_sd1_or_sd2_flat_file_falls_through_to_the_sd_dispatcher(tmp_path, monkeypatch) -> None:
    """Nothing in these names settles the family, so the loader must not enter the flat path with them.

    They reach the SD dispatcher instead: SD 1/2 has no conversion of its own, and the file stays a plain SD LoRA
    rather than being mapped onto SDXL modules that do not exist.
    """
    routed: list[dict] = []

    def _spy(*, state_dict):
        routed.append(state_dict)
        return _MockPatch()

    monkeypatch.setattr(lora_loader_module, "convert_sdxl_keys_to_diffusers_format", lambda state_dict: state_dict)
    monkeypatch.setattr(lora_loader_module, "lora_model_from_sd_state_dict", _spy)
    state_dict = _state_dict(sd15_lycoris_flat_lokr_format.state_dict_keys)
    loader = _loader(BaseModelType.StableDiffusion1, state_dict, monkeypatch)

    loader._load_model(_config(tmp_path, BaseModelType.StableDiffusion1))  # type: ignore[arg-type]

    assert len(routed) == 1
