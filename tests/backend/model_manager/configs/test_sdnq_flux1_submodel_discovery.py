"""Tests for FLUX.1 SDNQ pipeline submodel discovery.

`model_index.json` advertises which components a pipeline *should* have and what class each one is.
Neither is a fact about the folder: a partial download keeps a complete index over missing or empty
component directories, and the advertised class is a claim the folder can contradict. Discovery must
key on what actually ships, and must record the path it really found — the index names components
with arbitrary keys, so a pipeline that calls its CLIP encoder something other than `text_encoder`
must still load from the folder that exists.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.main import Main_SDNQ_Diffusers_FLUX_Config
from invokeai.backend.model_manager.taxonomy import SubModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "file_size": 1000,
    "name": "sdnq-flux1",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

# index key -> (advertised class, the class the folder's own config declares)
_WEIGHT_BEARING = {
    "text_encoder": ("CLIPTextModel", {"architectures": ["CLIPTextModel"]}),
    "text_encoder_2": ("T5EncoderModel", {"architectures": ["T5EncoderModel"]}),
    "vae": ("AutoencoderKL", {"_class_name": "AutoencoderKL"}),
}
_TOKENIZERS = {
    "tokenizer": "CLIPTokenizer",
    "tokenizer_2": "T5TokenizerFast",
}


def _write_component(root: Path, name: str, config: dict, *, weights: bool, populated: bool = True) -> None:
    component = root / name
    component.mkdir(parents=True, exist_ok=True)
    if not populated:
        return
    if weights:
        (component / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file({"weight": torch.zeros(4, 4)}, str(component / "model.safetensors"))
    else:
        (component / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
        (component / "tokenizer.json").write_text(json.dumps({}), encoding="utf-8")


def _make_flux1_pipeline(
    root: Path,
    *,
    component_keys: dict[str, str] | None = None,
    omit: str | None = None,
    empty: str | None = None,
) -> Path:
    """A FLUX.1 SDNQ pipeline folder.

    `component_keys` renames index keys (index key -> folder name) to exercise a nonstandard layout.
    `omit` drops a component's folder entirely; `empty` creates it without files.
    """
    keys = component_keys or {}
    root.mkdir(parents=True, exist_ok=True)

    index: dict[str, object] = {"_class_name": "FluxPipeline", "transformer": ["diffusers", "FluxTransformer2DModel"]}
    for name, (advertised, _) in _WEIGHT_BEARING.items():
        index[keys.get(name, name)] = ["transformers", advertised]
    for name, advertised in _TOKENIZERS.items():
        index[keys.get(name, name)] = ["transformers", advertised]
    (root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")

    transformer = root / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text(
        json.dumps({"_class_name": "FluxTransformer2DModel", "guidance_embeds": True, "in_channels": 64}),
        encoding="utf-8",
    )
    (transformer / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")
    save_file({"w": torch.zeros(4, 4)}, str(transformer / "diffusion_pytorch_model.safetensors"))

    for name, (_, declared) in _WEIGHT_BEARING.items():
        folder = keys.get(name, name)
        if omit == name:
            continue
        _write_component(root, folder, declared, weights=True, populated=empty != name)
    for name, advertised in _TOKENIZERS.items():
        folder = keys.get(name, name)
        if omit == name:
            continue
        _write_component(root, folder, {"tokenizer_class": advertised}, weights=False, populated=empty != name)

    return root


def _mod(root: Path) -> MagicMock:
    mod = MagicMock()
    mod.path = root
    mod.name = "sdnq-flux1"
    return mod


def _discover(root: Path) -> Main_SDNQ_Diffusers_FLUX_Config:
    return Main_SDNQ_Diffusers_FLUX_Config.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})


_ALL_SLOTS = {
    SubModelType.Transformer,
    SubModelType.TextEncoder,
    SubModelType.TextEncoder2,
    SubModelType.VAE,
    SubModelType.Tokenizer,
    SubModelType.Tokenizer2,
}

_SLOT_FOR_COMPONENT = {
    "text_encoder": SubModelType.TextEncoder,
    "text_encoder_2": SubModelType.TextEncoder2,
    "vae": SubModelType.VAE,
    "tokenizer": SubModelType.Tokenizer,
    "tokenizer_2": SubModelType.Tokenizer2,
}


def test_a_complete_pipeline_records_every_component(tmp_path: Path) -> None:
    config = _discover(_make_flux1_pipeline(tmp_path / "complete"))

    assert config.submodels is not None
    assert set(config.submodels) == _ALL_SLOTS


@pytest.mark.parametrize("component", sorted(_SLOT_FOR_COMPONENT))
def test_a_component_whose_directory_is_missing_is_not_recorded(tmp_path: Path, component: str) -> None:
    """A partial download keeps the full index; the missing component must not be advertised as present."""
    config = _discover(_make_flux1_pipeline(tmp_path / f"missing-{component}", omit=component))

    assert config.submodels is not None
    assert _SLOT_FOR_COMPONENT[component] not in config.submodels


@pytest.mark.parametrize("component", sorted(_SLOT_FOR_COMPONENT))
def test_a_component_whose_directory_is_empty_is_not_recorded(tmp_path: Path, component: str) -> None:
    """An interrupted download leaves the folders created but empty — nothing there to load."""
    config = _discover(_make_flux1_pipeline(tmp_path / f"empty-{component}", empty=component))

    assert config.submodels is not None
    assert _SLOT_FOR_COMPONENT[component] not in config.submodels


def test_a_component_that_contradicts_its_advertised_class_is_not_recorded(tmp_path: Path) -> None:
    root = _make_flux1_pipeline(tmp_path / "mismatched-vae")
    (root / "vae" / "config.json").write_text(json.dumps({"_class_name": "T5EncoderModel"}), encoding="utf-8")

    config = _discover(root)

    assert config.submodels is not None
    assert SubModelType.VAE not in config.submodels


def test_a_weight_bearing_component_that_names_no_class_is_not_recorded(tmp_path: Path) -> None:
    root = _make_flux1_pipeline(tmp_path / "undeclared-encoder")
    (root / "text_encoder" / "config.json").write_text(json.dumps({"hidden_size": 768}), encoding="utf-8")

    config = _discover(root)

    assert config.submodels is not None
    assert SubModelType.TextEncoder not in config.submodels


def test_nonstandard_index_keys_are_recorded_at_the_path_they_actually_live_at(tmp_path: Path) -> None:
    """The index may name a component anything. Discovery records the folder it found, so loading
    follows it instead of reconstructing a conventional name that does not exist here."""
    keys = {"text_encoder": "clip_encoder", "vae": "autoencoder"}
    root = _make_flux1_pipeline(tmp_path / "nonstandard", component_keys=keys)

    config = _discover(root)

    assert config.submodels is not None
    assert set(config.submodels) == _ALL_SLOTS
    assert config.submodels[SubModelType.TextEncoder].path_or_prefix.endswith("clip_encoder")
    assert config.submodels[SubModelType.VAE].path_or_prefix.endswith("autoencoder")
    # And the conventional folders genuinely do not exist, so a reconstructing loader would miss.
    assert not (root / "text_encoder").exists()
    assert not (root / "vae").exists()


def test_a_serialized_submodel_map_does_not_survive_a_deleted_component(tmp_path: Path) -> None:
    """Rehydration must revalidate against current files.

    A config persisted while the pipeline was complete carries the full submodel map. Replaying it
    would keep reporting a component that has since been deleted, so the pipeline would still look
    self-contained and the failure would only surface when a loader opens the missing folder.
    """
    root = _make_flux1_pipeline(tmp_path / "rehydrate")
    complete = _discover(root)
    assert complete.submodels is not None and SubModelType.VAE in complete.submodels

    # The component goes away after the config was written.
    for entry in (root / "vae").iterdir():
        entry.unlink()
    (root / "vae").rmdir()

    rehydrated = Main_SDNQ_Diffusers_FLUX_Config.from_model_on_disk(
        _mod(root),
        {**_REQUIRED_FIELDS, "path": root.as_posix(), "submodels": complete.submodels},
    )

    assert rehydrated.submodels is not None
    assert SubModelType.VAE not in rehydrated.submodels
