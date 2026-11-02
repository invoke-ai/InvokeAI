"""Identifying Tiny AutoEncoder files, through the whole config factory as an install does.

Files are written at the published shapes, so every other config class -- the VAEs, Spandrel, the
LoRA and main-model heuristics -- also gets its say, and "exactly one match" proves they stay out.
The weights decide the latent space; within the two spaces two bases share (4 channels: SD 1 / SDXL,
16: SD3 / FLUX.1) the base override, then the TAE's published name, decides, and nothing else may.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import ModelClassificationResult, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.tiny_autoencoder import TinyAutoencoderLoader
from invokeai.backend.model_manager.starter_models import STARTER_MODELS, StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.tiny_autoencoder.checkpoint import TaesdLayout
from tests.fixtures.tiny_autoencoders import (
    TAEF2,
    TAEQI2_1,
    TAESD,
    TAESD3,
    diffusers_weights,
    native_weights,
    tinier_weights,
)

_COMMON: dict[str, Any] = {"hash": "blake3:fake", "file_size": 1000, "source_type": "path"}

_TAEF1_URL = (
    "https://huggingface.co/madebyollin/taef1/resolve/b1b2d00e9e440cfbf3dedb34266864da86016ceb/"
    "diffusion_pytorch_model.safetensors"
)
_TAESD3_URL = (
    "https://huggingface.co/madebyollin/taesd3/resolve/d58dcaccd2b36fcb7a6b9e93c1cc507acab5a778/"
    "diffusion_pytorch_model.safetensors"
)


def _zeros(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Real shapes, no values: identification reads only the header."""
    return {key: torch.zeros(value.shape, dtype=torch.float16) for key, value in sd.items()}


def _write(path: Path, layout: TaesdLayout, combined: bool) -> Path:
    native = native_weights(layout)
    save_file(_zeros(diffusers_weights(native) if combined else native), path)
    return path


def _identify(path: Path, **overrides: Any) -> ModelClassificationResult:
    return ModelConfigFactory.from_model_on_disk(path, {**_COMMON, "source": str(path), **overrides})


def _single_tae(result: ModelClassificationResult) -> Any:
    assert [type(c).__name__ for c in result.all_matches] == [type(result.config).__name__]
    config = result.config
    assert config.type is ModelType.TinyAutoencoder
    assert config.format is ModelFormat.Checkpoint
    return config


@pytest.mark.parametrize(
    ("filename", "layout", "combined", "base"),
    [
        ("taesd_decoder.safetensors", TAESD, False, BaseModelType.StableDiffusion1),
        ("taesdxl_decoder.safetensors", TAESD, False, BaseModelType.StableDiffusionXL),
        ("taesd3_decoder.safetensors", TAESD3, False, BaseModelType.StableDiffusion3),
        ("taef1.safetensors", TAESD3, True, BaseModelType.Flux),
        # The two families no other base shares need no name at all.
        ("preview.safetensors", TAEF2, True, BaseModelType.Flux2),
        ("preview.safetensors", TAEQI2_1, True, BaseModelType.QwenImage21),
    ],
)
def test_each_published_tae_is_filed_under_its_base_and_nothing_else_matches(
    tmp_path: Path, filename: str, layout: TaesdLayout, combined: bool, base: BaseModelType
) -> None:
    config = _single_tae(_identify(_write(tmp_path / filename, layout, combined)))

    assert config.base is base
    assert config.latent_channels == layout.latent_channels


@pytest.mark.parametrize(
    ("source", "base"), [(_TAEF1_URL, BaseModelType.Flux), (_TAESD3_URL, BaseModelType.StableDiffusion3)]
)
def test_the_install_url_names_a_file_its_own_name_does_not(tmp_path: Path, source: str, base: BaseModelType) -> None:
    path = _write(tmp_path / "diffusion_pytorch_model.safetensors", TAESD3, combined=True)

    assert _single_tae(_identify(path, source=source, source_type="url")).base is base


def _moved(tmp_path: Path) -> Path:
    """Where an install moves `diffusion_pytorch_model.safetensors`: a folder named for the record's key."""
    (tmp_path / "0123abcd").mkdir()
    return _write(tmp_path / "0123abcd" / "diffusion_pytorch_model.safetensors", TAESD3, combined=True)


@pytest.mark.parametrize(
    ("source", "base"),
    [
        (r"C:\Users\me\Downloads\taef1\diffusion_pytorch_model.safetensors", BaseModelType.Flux),
        ("/home/me/downloads/taesd3/diffusion_pytorch_model.safetensors", BaseModelType.StableDiffusion3),
    ],
)
def test_a_moved_local_install_is_named_by_the_folder_of_its_source_path_from_either_os(
    tmp_path: Path, source: str, base: BaseModelType
) -> None:
    """Re-identification reads the file and folder name of the recorded source, whichever OS recorded it."""
    assert _single_tae(_identify(_moved(tmp_path), source=source, source_type="path")).base is base


def test_an_ancestor_folder_of_a_local_source_path_does_not_name_the_tae(tmp_path: Path) -> None:
    """Only the source's file and folder name are evidence, not the folders the user keeps those in."""
    result = _identify(_moved(tmp_path), source=r"C:\taef1\Downloads\diffusion_pytorch_model.safetensors")

    assert result.config is None
    assert result.invalid_matches


@pytest.mark.parametrize("base", [BaseModelType.StableDiffusion3, BaseModelType.Flux])
def test_a_base_override_decides_within_the_shared_family(tmp_path: Path, base: BaseModelType) -> None:
    """The starter installer always sends the base; it outranks a name that says otherwise."""
    path = _write(tmp_path / "taesd3_decoder.safetensors", TAESD3, combined=False)

    assert _single_tae(_identify(path, base=base.value)).base is base


@pytest.mark.parametrize(
    ("filename", "layout"),
    [
        ("diffusion_pytorch_model.safetensors", TAESD3),
        ("decoder.safetensors", TAESD),
        # A name from the other family contradicts the weights and decides nothing.
        ("taef1.safetensors", TAESD),
    ],
)
def test_a_shared_family_file_nothing_names_is_refused_not_guessed(
    tmp_path: Path, filename: str, layout: TaesdLayout
) -> None:
    result = _identify(_write(tmp_path / filename, layout, combined=True))

    assert result.config is None
    assert result.invalid_matches
    assert all("starter models" in str(e) for e in result.invalid_matches)


def test_the_tinier_taeqi2_1_is_refused_with_the_file_to_install(tmp_path: Path) -> None:
    path = tmp_path / "taeqi2_1_tinier.safetensors"
    save_file(tinier_weights(), path)

    result = _identify(path)

    assert result.config is None
    assert result.invalid_matches
    assert all("Install taeqi2_1.safetensors" in str(e) for e in result.invalid_matches)


def test_a_base_override_the_weights_cannot_serve_is_refused_not_filed_as_unknown(tmp_path: Path) -> None:
    path = _write(tmp_path / "taesd_decoder.safetensors", TAESD, combined=False)

    result = _identify(path, base=BaseModelType.Flux.value)

    assert result.config is None
    assert any(isinstance(e, InvalidMatchError) for e in result.invalid_matches)


def test_comfyui_pth_decoders_are_identified(tmp_path: Path) -> None:
    path = tmp_path / "taesdxl_decoder.pth"
    torch.save(native_weights(TAESD), path)

    assert _single_tae(_identify(path)).base is BaseModelType.StableDiffusionXL


def test_a_tae_in_a_file_type_the_loader_does_not_read_is_not_installed_as_one(tmp_path: Path) -> None:
    """Identification reads `.ckpt` shapes fine; accepting it would install a model that cannot load."""
    path = tmp_path / "taef2.ckpt"
    torch.save(diffusers_weights(native_weights(TAEF2)), path)

    result = _identify(path)

    assert not any(config.type is ModelType.TinyAutoencoder for config in result.all_matches)
    assert result.config is None or result.config.type is not ModelType.TinyAutoencoder


# What each madebyollin repo publishes, independent of the starter entries: the base its latents belong to,
# and the layout (native decoder-only for taesd/taesdxl, combined otherwise).
_PUBLISHED_TAES: dict[str, tuple[BaseModelType, TaesdLayout, bool]] = {
    "taesd": (BaseModelType.StableDiffusion1, TAESD, False),
    "taesdxl": (BaseModelType.StableDiffusionXL, TAESD, False),
    "taesd3": (BaseModelType.StableDiffusion3, TAESD3, True),
    "taef1": (BaseModelType.Flux, TAESD3, True),
    "taef2": (BaseModelType.Flux2, TAEF2, True),
    "taeqi2_1": (BaseModelType.QwenImage21, TAEQI2_1, True),
}
_TAE_STARTERS = [m for m in STARTER_MODELS if m.type is ModelType.TinyAutoencoder]


def _repo(starter: StarterModel) -> str:
    return starter.source.removeprefix("https://huggingface.co/madebyollin/").split("/", 1)[0]


def test_every_tae_config_has_exactly_one_starter() -> None:
    config_bases = {
        c.model_fields["base"].default
        for c in Config_Base.CONFIG_CLASSES
        if c.model_fields["type"].default is ModelType.TinyAutoencoder
    }

    assert sorted(_repo(m) for m in _TAE_STARTERS) == sorted(_PUBLISHED_TAES)
    assert {base for base, _, _ in _PUBLISHED_TAES.values()} == config_bases


@pytest.mark.parametrize("starter", _TAE_STARTERS, ids=_repo)
def test_a_starter_install_identifies_round_trips_and_has_a_loader(tmp_path: Path, starter: StarterModel) -> None:
    """The payload is the one the starter installer sends: base, type, format, name, description."""
    base, layout, combined = _PUBLISHED_TAES[_repo(starter)]
    path = _write(tmp_path / starter.source.rsplit("/", 1)[-1], layout, combined)
    payload = {
        "base": starter.base.value,
        "type": starter.type.value,
        "format": starter.format.value,
        "name": starter.name,
        "description": starter.description,
    }

    config = _single_tae(_identify(path, **payload, source=starter.source, source_type="url"))

    assert config.base is base
    assert config.name == starter.name
    assert config.latent_channels == layout.latent_channels
    assert ModelConfigFactory.from_json(config.model_dump_json()) == config
    implementation, _, _ = ModelLoaderRegistry.get_implementation(config, None)
    assert implementation is TinyAutoencoderLoader
