"""Identifying Tiny AutoEncoder files, through the whole config factory as an install does.

Files are written at the published shapes, so every other config class -- the VAEs, Spandrel, the
LoRA and main-model heuristics -- also gets its say, and "exactly one match" proves they stay out.
The weights decide the latent space; where a network is not specific to one base (4 channels: SD 1 /
SDXL, 16: SD3 / FLUX.1, taew2_1's network: also HunyuanVideo and CogVideoX) the base override, then the
TAE's published name, decides, and nothing else may.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.architectures import LatentSpaceFacet, require
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import ModelClassificationResult, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.tiny_autoencoder import TinyAutoencoderLoader
from invokeai.backend.model_manager.starter_models import STARTER_MODELS, StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.tiny_autoencoder.checkpoint import (
    TAELTX2_3,
    TAEW2_1,
    TAEW2_2,
    TaehvLayout,
    TinyAutoencoderLayout,
)
from tests.fixtures.tiny_autoencoders import (
    TAEF2,
    TAEQI2_1,
    TAESD,
    TAESD3,
    diffusers_weights,
    native_weights,
    taehv_weights,
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


def _write(path: Path, layout: TinyAutoencoderLayout, combined: bool = True) -> Path:
    """A TAESD file in the combined or native layout, or a TAEHV file (which has only one)."""
    if isinstance(layout, TaehvLayout):
        save_file(_zeros(taehv_weights(layout)), path)
        return path
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
        ("taew2_1.safetensors", TAEW2_1, True, BaseModelType.Wan),
        ("preview.safetensors", TAEW2_2, True, BaseModelType.Wan),
        ("preview.safetensors", TAELTX2_3, True, BaseModelType.LTX2),
    ],
)
def test_each_published_tae_is_filed_under_its_base_and_nothing_else_matches(
    tmp_path: Path, filename: str, layout: TinyAutoencoderLayout, combined: bool, base: BaseModelType
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
    tmp_path: Path, filename: str, layout: TinyAutoencoderLayout
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


def test_a_taehv_network_shared_with_unsupported_models_is_refused_unless_named(tmp_path: Path) -> None:
    """taew2_1's network is also published for HunyuanVideo and CogVideoX, whose latents it cannot preview."""
    for filename in ("preview.safetensors", "taehv.safetensors", "taecvx.safetensors"):
        result = _identify(_write(tmp_path / filename, TAEW2_1))

        assert result.config is None, filename
        assert result.invalid_matches
        assert all("HunyuanVideo" in str(e) and "taew2_1" in str(e) for e in result.invalid_matches)


def test_a_wan_override_files_an_unnamed_taew2_1(tmp_path: Path) -> None:
    path = _write(tmp_path / "preview.safetensors", TAEW2_1)

    config = _single_tae(_identify(path, base=BaseModelType.Wan.value))

    assert (config.base, config.latent_channels) == (BaseModelType.Wan, 16)


def test_taeh3_is_left_to_the_minimax_h3_preview_path(tmp_path: Path) -> None:
    taeh3 = TaehvLayout(latent_channels=24, patch_size=2, time_upscale=TAEW2_1.time_upscale)

    result = _identify(_write(tmp_path / "taeh3.safetensors", taeh3))

    assert not any(config.type is ModelType.TinyAutoencoder for config in result.all_matches)
    assert not result.invalid_matches


# What each published TAE behind a starter is, independent of the starter entries: the base its latents
# belong to, and the layout (TAESD: native decoder-only for taesd/taesdxl, combined otherwise).
_PUBLISHED_TAES: dict[str, tuple[BaseModelType, TinyAutoencoderLayout, bool]] = {
    "taesd": (BaseModelType.StableDiffusion1, TAESD, False),
    "taesdxl": (BaseModelType.StableDiffusionXL, TAESD, False),
    "taesd3": (BaseModelType.StableDiffusion3, TAESD3, True),
    "taef1": (BaseModelType.Flux, TAESD3, True),
    "taef2": (BaseModelType.Flux2, TAEF2, True),
    "taeqi2_1": (BaseModelType.QwenImage21, TAEQI2_1, True),
    "taew2_1": (BaseModelType.Wan, TAEW2_1, True),
    "taew2_2": (BaseModelType.Wan, TAEW2_2, True),
    "taeltx2_3": (BaseModelType.LTX2, TAELTX2_3, True),
}
_TAE_STARTERS = [m for m in STARTER_MODELS if m.type is ModelType.TinyAutoencoder]


def _tae_name(starter: StarterModel) -> str:
    """The TAE's name: its Hugging Face repo, or its file in the TAEHV GitHub repo."""
    match = re.search(r"madebyollin/(?:taehv/raw/[0-9a-f]+/safetensors/)?(?P<name>\w+)", starter.source)
    assert match is not None, starter.source
    return match["name"]


def test_every_space_a_tae_can_be_filed_under_has_exactly_one_starter() -> None:
    """One starter per `(base, latent channels)` a TAE config accepts: Wan has two, one per latent space."""
    fileable = set()
    for config in Config_Base.CONFIG_CLASSES:
        if config.model_fields["type"].default is ModelType.TinyAutoencoder:
            base = config.model_fields["base"].default
            facet = require(base, LatentSpaceFacet)
            fileable |= {(base, space.channels) for space in (facet.primary, *facet.alternates)}
    starter_spaces = [(base, layout.latent_channels) for base, layout, _ in _PUBLISHED_TAES.values()]

    assert sorted(_tae_name(m) for m in _TAE_STARTERS) == sorted(_PUBLISHED_TAES)
    assert len(starter_spaces) == len(set(starter_spaces))
    assert set(starter_spaces) == fileable


@pytest.mark.parametrize("starter", _TAE_STARTERS, ids=_tae_name)
def test_a_starter_install_identifies_round_trips_and_has_a_loader(tmp_path: Path, starter: StarterModel) -> None:
    """The payload is the one the starter installer sends: base, type, format, name, description."""
    base, layout, combined = _PUBLISHED_TAES[_tae_name(starter)]
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


_TAELTX_2_URL = (
    "https://github.com/madebyollin/taehv/raw/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeltx_2.safetensors"
)


@pytest.mark.parametrize(
    ("filename", "overrides"),
    [
        ("taeltx_2.safetensors", {}),
        ("taeltx2.pth", {}),
        ("diffusion_pytorch_model.safetensors", {"source": _TAELTX_2_URL, "source_type": "url"}),
        # Not even an explicit LTX-2 base installs it: it previews LTX-2.5 worse than the linear projection.
        ("taeltx_2.safetensors", {"base": BaseModelType.LTX2.value}),
    ],
)
def test_taeltx_2_is_refused_by_name_with_the_decoder_to_install(
    tmp_path: Path, filename: str, overrides: dict[str, Any]
) -> None:
    path = tmp_path / filename
    if path.suffix == ".pth":
        torch.save(taehv_weights(TAELTX2_3), path)
    else:
        _write(path, TAELTX2_3)

    result = _identify(path, **overrides)

    assert result.config is None
    assert result.invalid_matches
    assert all("LTX-2.5 needs taeltx2_3" in str(e) for e in result.invalid_matches)


def test_taeltx2_3_in_a_folder_named_for_taeltx_2_is_accepted(tmp_path: Path) -> None:
    """The file's own name is the more specific evidence."""
    folder = tmp_path / "taeltx_2"
    folder.mkdir()

    config = _single_tae(_identify(_write(folder / "taeltx2_3.safetensors", TAELTX2_3)))

    assert config.base is BaseModelType.LTX2
