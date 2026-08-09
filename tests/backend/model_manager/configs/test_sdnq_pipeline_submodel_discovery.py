"""Tests that SDNQ pipeline submodel discovery recognizes the compatible Qwen encoder/tokenizer
classes the loader can actually load.

_get_submodels() must record the TextEncoder / Tokenizer submodels for the text-only Qwen causal-LM
classes and the slow/fast Qwen2 tokenizer classes; otherwise a valid SDNQ pipeline whose
model_index.json advertises e.g. Qwen2ForCausalLM or Qwen2TokenizerFast is mis-recorded as partial
and is_self_contained_sdnq_pipeline() wrongly returns False, forcing separate VAE/Qwen3 sources.

It must NOT record a TextEncoder for Qwen2VLForConditionalGeneration: that is a multimodal Qwen-VL
model (with a visual tower), but the SDNQ pipeline loaders instantiate a text-only Qwen3ForCausalLM,
so treating it as self-contained would mark the pipeline complete even though the loader would fail
on the visual-tower weights.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.app.invocations.model import is_self_contained_sdnq_pipeline
from invokeai.backend.model_manager.configs.main import (
    Main_SDNQ_Diffusers_Flux2_Config,
    Main_SDNQ_Diffusers_ZImage_Config,
)
from invokeai.backend.model_manager.taxonomy import SubModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "file_size": 1000,
    "name": "sdnq-pipeline",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

# Only Qwen3ForCausalLM is loadable by the pipeline loader (it builds a text-only Qwen3ForCausalLM).
# Qwen2ForCausalLM (missing Qwen3 q/k-norm params) and the multimodal Qwen2VLForConditionalGeneration
# (visual tower) are not loadable and must not be recorded as self-contained TextEncoders.
_ENCODER_CLASSES = ["Qwen3ForCausalLM"]
_UNLOADABLE_ENCODER_CLASSES = ["Qwen2ForCausalLM", "Qwen2VLForConditionalGeneration"]


def _write_sdnq_transformer(root: Path, transformer_config: dict, class_name: str = "Flux2Transformer2DModel") -> None:
    transformer_dir = root / "transformer"
    transformer_dir.mkdir()
    # Real diffusers component configs carry `_class_name` (save_config writes it). Discovery requires
    # a weight-bearing component to name itself, so the fixture must too.
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": class_name, **transformer_config}), encoding="utf-8"
    )
    (transformer_dir / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")
    save_file(
        {
            "transformer_blocks.0.attn.to_q.weight": torch.zeros(64, 32, dtype=torch.uint8),
            "transformer_blocks.0.attn.to_q.scale": torch.zeros(64, 1, dtype=torch.float32),
        },
        str(transformer_dir / "diffusion_pytorch_model.safetensors"),
    )


# The component folders discovery must find populated on disk before recording a submodel.
# model_index.json advertising them is not enough — a partial download can keep the index while its
# component folders are missing, and an interrupted one leaves them created but empty.
_PIPELINE_COMPONENT_DIRS = ("vae", "text_encoder", "tokenizer")


# What each component's own config declares about itself, as `save_pretrained` would write it:
# `architectures` for transformers models, `_class_name` for diffusers ones. `AutoencoderKL` is
# accepted for both pipelines (FLUX.2 also allows the `AutoencoderKLFlux2` spelling).
_COMPONENT_SELF_DECLARED_CONFIG = {
    "vae": {"_class_name": "AutoencoderKL"},
    "text_encoder": {"architectures": ["Qwen3ForCausalLM"]},
}


def _write_component_dirs(
    root: Path, components: tuple[str, ...] = _PIPELINE_COMPONENT_DIRS, *, populated: bool = True
) -> None:
    """Create the given component subfolders.

    With ``populated=True`` each folder gets the files its loader needs (config + weights, or the
    tokenizer's config for the weightless tokenizer folder). ``populated=False`` creates the bare
    directories an interrupted download leaves behind — discovery must not record those.

    The weight-bearing configs name their own class, because that is what a real `save_pretrained`
    writes and what discovery now requires: "some config plus some weights" is the shape an unrelated
    model has too, so it is no longer accepted on its own.
    """
    for name in components:
        component_dir = root / name
        component_dir.mkdir(exist_ok=True)
        if not populated:
            continue
        if name == "tokenizer":
            (component_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "Qwen2TokenizerFast"}), encoding="utf-8"
            )
            (component_dir / "tokenizer.json").write_text(json.dumps({}), encoding="utf-8")
            continue
        (component_dir / "config.json").write_text(
            json.dumps(_COMPONENT_SELF_DECLARED_CONFIG.get(name, {})), encoding="utf-8"
        )
        save_file({"weight": torch.zeros(4, 4, dtype=torch.float32)}, str(component_dir / "model.safetensors"))


def _write_qwen_vl_text_encoder(root: Path) -> None:
    """Write a text_encoder/ folder that actually contains a Qwen-VL model: a config declaring the
    Qwen-VL architecture and SDNQ weights that include both language (model.*) and visual-tower
    (visual.*) keys. The pipeline loader's text-only Qwen3ForCausalLM cannot consume these."""
    te_dir = root / "text_encoder"
    te_dir.mkdir(exist_ok=True)
    (te_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2VLForConditionalGeneration"], "hidden_size": 2560}), encoding="utf-8"
    )
    (te_dir / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(1000, 2560, dtype=torch.uint8),
            "model.embed_tokens.scale": torch.zeros(1000, 1, dtype=torch.float32),
            "visual.patch_embed.proj.weight": torch.zeros(64, 32, dtype=torch.uint8),
            "visual.patch_embed.proj.scale": torch.zeros(64, 1, dtype=torch.float32),
        },
        str(te_dir / "model.safetensors"),
    )


def _make_flux2_pipeline(
    root: Path,
    encoder_class: str,
    components: tuple[str, ...] = _PIPELINE_COMPONENT_DIRS,
    *,
    populated: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2KleinPipeline",
                "transformer": ["diffusers", "Flux2Transformer2DModel"],
                "text_encoder": ["transformers", encoder_class],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            }
        ),
        encoding="utf-8",
    )
    _write_sdnq_transformer(root, {"attention_head_dim": 128, "num_attention_heads": 24, "joint_attention_dim": 7680})
    _write_component_dirs(root, components, populated=populated)
    return root


def _make_zimage_pipeline(
    root: Path,
    encoder_class: str,
    components: tuple[str, ...] = _PIPELINE_COMPONENT_DIRS,
    *,
    populated: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ZImagePipeline",
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "text_encoder": ["transformers", encoder_class],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )
    _write_sdnq_transformer(root, {"_class_name": "ZImageTransformer2DModel"})
    scheduler_dir = root / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps({"shift": 3.0}), encoding="utf-8")
    _write_component_dirs(root, components, populated=populated)
    return root


def _mod(root: Path) -> MagicMock:
    mod = MagicMock()
    mod.path = root
    mod.name = "sdnq-klein-4b"  # no "base" substring -> distilled variant
    return mod


def _assert_complete_pipeline(config) -> None:
    assert config.submodels is not None
    assert SubModelType.Transformer in config.submodels
    assert SubModelType.TextEncoder in config.submodels
    assert SubModelType.Tokenizer in config.submodels
    assert SubModelType.VAE in config.submodels
    assert is_self_contained_sdnq_pipeline(config)


@pytest.mark.parametrize("encoder_class", _ENCODER_CLASSES)
def test_flux2_sdnq_pipeline_records_compatible_encoder_and_fast_tokenizer(tmp_path: Path, encoder_class: str):
    root = _make_flux2_pipeline(tmp_path / "flux2", encoder_class)
    config = Main_SDNQ_Diffusers_Flux2_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    _assert_complete_pipeline(config)


@pytest.mark.parametrize("encoder_class", _ENCODER_CLASSES)
def test_zimage_sdnq_pipeline_records_compatible_encoder_and_fast_tokenizer(tmp_path: Path, encoder_class: str):
    root = _make_zimage_pipeline(tmp_path / "zimage", encoder_class)
    config = Main_SDNQ_Diffusers_ZImage_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    _assert_complete_pipeline(config)


def _assert_pipeline_not_self_contained(config) -> None:
    # The VAE and Tokenizer are still recorded, but the unloadable text encoder is NOT — so the
    # pipeline is not self-contained and readiness/invocation must require an explicit text-only Qwen3
    # source instead of selecting the main model (which the loader could not load).
    assert config.submodels is not None
    assert SubModelType.TextEncoder not in config.submodels
    assert SubModelType.VAE in config.submodels
    assert not is_self_contained_sdnq_pipeline(config)


@pytest.mark.parametrize("encoder_class", _UNLOADABLE_ENCODER_CLASSES)
def test_flux2_sdnq_pipeline_with_unloadable_encoder_is_not_self_contained(tmp_path: Path, encoder_class: str):
    root = _make_flux2_pipeline(tmp_path / "flux2-unloadable", encoder_class)
    _write_qwen_vl_text_encoder(root)
    config = Main_SDNQ_Diffusers_Flux2_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    _assert_pipeline_not_self_contained(config)


@pytest.mark.parametrize("encoder_class", _UNLOADABLE_ENCODER_CLASSES)
def test_zimage_sdnq_pipeline_with_unloadable_encoder_is_not_self_contained(tmp_path: Path, encoder_class: str):
    root = _make_zimage_pipeline(tmp_path / "zimage-unloadable", encoder_class)
    _write_qwen_vl_text_encoder(root)
    config = Main_SDNQ_Diffusers_ZImage_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    _assert_pipeline_not_self_contained(config)


# --- Partial download: complete model_index.json but component folders missing on disk ---
# is_self_contained_sdnq_pipeline must key on what actually ships, not on what the index advertises,
# otherwise the loaders later request fixed vae/ text_encoder/ tokenizer/ subfolders that aren't there.


@pytest.mark.parametrize("missing", ["vae", "text_encoder", "tokenizer"])
def test_flux2_sdnq_pipeline_with_missing_component_dir_is_not_self_contained(tmp_path: Path, missing: str):
    present = tuple(c for c in _PIPELINE_COMPONENT_DIRS if c != missing)
    root = _make_flux2_pipeline(tmp_path / f"flux2-missing-{missing}", "Qwen3ForCausalLM", components=present)
    config = Main_SDNQ_Diffusers_Flux2_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    assert config.submodels is not None
    assert not is_self_contained_sdnq_pipeline(config)
    # The absent component is not recorded, even though model_index.json still advertises it.
    submodel_for = {
        "vae": SubModelType.VAE,
        "text_encoder": SubModelType.TextEncoder,
        "tokenizer": SubModelType.Tokenizer,
    }
    assert submodel_for[missing] not in config.submodels


@pytest.mark.parametrize("missing", ["vae", "text_encoder", "tokenizer"])
def test_zimage_sdnq_pipeline_with_missing_component_dir_is_not_self_contained(tmp_path: Path, missing: str):
    present = tuple(c for c in _PIPELINE_COMPONENT_DIRS if c != missing)
    root = _make_zimage_pipeline(tmp_path / f"zimage-missing-{missing}", "Qwen3ForCausalLM", components=present)
    config = Main_SDNQ_Diffusers_ZImage_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    assert config.submodels is not None
    assert not is_self_contained_sdnq_pipeline(config)
    submodel_for = {
        "vae": SubModelType.VAE,
        "text_encoder": SubModelType.TextEncoder,
        "tokenizer": SubModelType.Tokenizer,
    }
    assert submodel_for[missing] not in config.submodels


@pytest.mark.parametrize(
    ("factory", "root_name"),
    [
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-empty-components"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-empty-components"),
    ],
)
def test_sdnq_pipeline_with_empty_component_dirs_is_not_self_contained(tmp_path: Path, factory, root_name: str):
    """An interrupted download leaves the component folders created but empty. Their mere existence
    must not make the pipeline self-contained — the loaders would find nothing to read."""
    maker = _make_flux2_pipeline if factory is Main_SDNQ_Diffusers_Flux2_Config else _make_zimage_pipeline
    root = maker(tmp_path / root_name, "Qwen3ForCausalLM", populated=False)
    config = factory.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    assert not is_self_contained_sdnq_pipeline(config)
    assert config.submodels is not None
    for submodel in (SubModelType.VAE, SubModelType.TextEncoder, SubModelType.Tokenizer):
        assert submodel not in config.submodels


@pytest.mark.parametrize(
    ("factory", "root_name"),
    [
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-missing-index-transformer"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-missing-index-transformer"),
    ],
)
def test_sdnq_pipeline_without_index_transformer_is_not_self_contained(tmp_path: Path, factory, root_name: str):
    """A malformed index can advertise the components while omitting the transformer every loader
    requests. The components alone must not satisfy the self-contained check."""
    maker = _make_flux2_pipeline if factory is Main_SDNQ_Diffusers_Flux2_Config else _make_zimage_pipeline
    root = maker(tmp_path / root_name, "Qwen3ForCausalLM")
    model_index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    del model_index["transformer"]
    (root / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")

    config = factory.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    assert config.submodels is not None
    assert SubModelType.Transformer not in config.submodels
    assert not is_self_contained_sdnq_pipeline(config)


# --- Mismatched components: the index advertises a loadable class over a folder holding something else ---
# model_index.json is a claim about each folder, not a fact about it. A repo can advertise the one
# class the loader supports and ship a different model, which passes both the class-name check and the
# file-presence check. The pipeline is then recorded as self-contained and the mismatch only surfaces
# at generation time, when the loader builds the advertised class against an incompatible state dict.


@pytest.mark.parametrize(
    ("factory", "root_name"),
    [
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-mismatched-encoder"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-mismatched-encoder"),
    ],
)
def test_sdnq_pipeline_with_a_mismatched_text_encoder_folder_is_not_self_contained(
    tmp_path: Path, factory, root_name: str
):
    """Index says Qwen3ForCausalLM; the folder holds a multimodal Qwen-VL model."""
    maker = _make_flux2_pipeline if factory is Main_SDNQ_Diffusers_Flux2_Config else _make_zimage_pipeline
    root = maker(tmp_path / root_name, "Qwen3ForCausalLM")
    _write_qwen_vl_text_encoder(root)

    config = factory.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    _assert_pipeline_not_self_contained(config)


@pytest.mark.parametrize(
    ("factory", "root_name"),
    [
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-mismatched-vae"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-mismatched-vae"),
    ],
)
def test_sdnq_pipeline_with_a_mismatched_vae_folder_is_not_self_contained(tmp_path: Path, factory, root_name: str):
    """Index says the VAE slot holds an AutoencoderKL; the folder declares an unrelated model."""
    maker = _make_flux2_pipeline if factory is Main_SDNQ_Diffusers_Flux2_Config else _make_zimage_pipeline
    root = maker(tmp_path / root_name, "Qwen3ForCausalLM")
    (root / "vae" / "config.json").write_text(json.dumps({"_class_name": "Qwen2ForCausalLM"}), encoding="utf-8")

    config = factory.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    assert config.submodels is not None
    assert SubModelType.VAE not in config.submodels
    assert not is_self_contained_sdnq_pipeline(config)


@pytest.mark.parametrize(
    ("factory", "root_name", "component"),
    [
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-undeclared-encoder", "text_encoder"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-undeclared-encoder", "text_encoder"),
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-undeclared-vae", "vae"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-undeclared-vae", "vae"),
    ],
)
def test_sdnq_pipeline_with_a_weight_bearing_component_that_names_no_class_is_not_self_contained(
    tmp_path: Path, factory, root_name: str, component: str
):
    """Silence is not evidence.

    "Some config.json plus some weight file" is exactly the shape an unrelated model has, so a
    weight-bearing component that names no class cannot be confirmed and must not be recorded.
    Accepting it would leave the hole open for every mismatched component whose config happens to
    omit the key — which is the reviewer's "any directory containing a config file and weight file".
    """
    maker = _make_flux2_pipeline if factory is Main_SDNQ_Diffusers_Flux2_Config else _make_zimage_pipeline
    root = maker(tmp_path / root_name, "Qwen3ForCausalLM")
    (root / component / "config.json").write_text(json.dumps({"hidden_size": 2560}), encoding="utf-8")

    config = factory.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    submodel = SubModelType.TextEncoder if component == "text_encoder" else SubModelType.VAE
    assert config.submodels is not None
    assert submodel not in config.submodels
    assert not is_self_contained_sdnq_pipeline(config)


@pytest.mark.parametrize(
    ("factory", "root_name"),
    [
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-undeclared-tokenizer"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-undeclared-tokenizer"),
    ],
)
def test_sdnq_pipeline_tokenizer_without_a_declared_class_is_still_recorded(tmp_path: Path, factory, root_name: str):
    """The tokenizer stays lenient: it carries no weights, so nothing can be mis-instantiated against
    it, and `tokenizer_class` is less consistently written than `_class_name`/`architectures`."""
    maker = _make_flux2_pipeline if factory is Main_SDNQ_Diffusers_Flux2_Config else _make_zimage_pipeline
    root = maker(tmp_path / root_name, "Qwen3ForCausalLM")
    (root / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({}), encoding="utf-8")

    config = factory.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    _assert_complete_pipeline(config)


@pytest.mark.parametrize("suffix", [".gguf", ".bin", ".pt"])
@pytest.mark.parametrize(
    ("factory", "root_name"),
    [
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-nonsafetensors-encoder"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-nonsafetensors-encoder"),
    ],
)
def test_a_quantized_component_without_safetensors_is_not_recorded(
    tmp_path: Path, factory, root_name: str, suffix: str
):
    """`sdnq_sd_loader` globs `*.safetensors` and raises when it finds none.

    A folder that declares the right class and holds a `.gguf`/`.bin` looks populated to a generic
    file check, so the pipeline was recorded as self-contained and the failure moved to load time.
    """
    maker = _make_flux2_pipeline if factory is Main_SDNQ_Diffusers_Flux2_Config else _make_zimage_pipeline
    root = maker(tmp_path / f"{root_name}{suffix}", "Qwen3ForCausalLM")

    encoder = root / "text_encoder"
    (encoder / "model.safetensors").unlink()
    # Marking it SDNQ is what routes it to the safetensors-only loader.
    (encoder / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")
    (encoder / f"model{suffix}").write_bytes(b"\x00")

    config = factory.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    assert config.submodels is not None
    assert SubModelType.TextEncoder not in config.submodels
    assert not is_self_contained_sdnq_pipeline(config)


@pytest.mark.parametrize(
    ("factory", "root_name"),
    [
        (Main_SDNQ_Diffusers_Flux2_Config, "flux2-bin-vae"),
        (Main_SDNQ_Diffusers_ZImage_Config, "zimage-bin-vae"),
    ],
)
def test_an_unquantized_component_may_still_ship_a_non_safetensors_weight(tmp_path: Path, factory, root_name: str):
    """SDNQ exports leave the VAE unquantized; it goes through `from_pretrained`, which reads more
    than safetensors. Narrowing every component to safetensors would reject those pipelines."""
    maker = _make_flux2_pipeline if factory is Main_SDNQ_Diffusers_Flux2_Config else _make_zimage_pipeline
    root = maker(tmp_path / root_name, "Qwen3ForCausalLM")

    vae = root / "vae"
    (vae / "model.safetensors").unlink()
    (vae / "diffusion_pytorch_model.bin").write_bytes(b"\x00")  # no quantization_config.json here

    config = factory.from_model_on_disk(_mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    assert config.submodels is not None
    assert SubModelType.VAE in config.submodels
