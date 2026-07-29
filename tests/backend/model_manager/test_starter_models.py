"""Tests for the Krea-2 starter-model bundle and its GGUF dependency wiring.

A single-file / GGUF Krea-2 transformer ships *only* the transformer, so it is unusable without a
standalone Qwen-Image VAE and Qwen3-VL text encoder. These tests assert that the Krea-2 launchpad
bundle exists, exposes both the diffusers and GGUF options, and that each GGUF entry declares the two
standalone dependencies so installing it also pulls the pieces needed to run it.
"""

from invokeai.backend.model_manager.starter_models import (
    STARTER_BUNDLES,
    STARTER_MODELS,
    StarterModel,
)
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Krea2VariantType,
    ModelFormat,
    ModelType,
)


def _krea2_bundle_by_source() -> dict[str, StarterModel]:
    bundle = STARTER_BUNDLES[BaseModelType.Krea2]
    return {model.source: model for model in bundle.models}


def test_krea2_bundle_is_registered() -> None:
    assert BaseModelType.Krea2 in STARTER_BUNDLES
    assert STARTER_BUNDLES[BaseModelType.Krea2].name == "Krea-2"


def test_krea2_bundle_contains_diffusers_gguf_and_standalone_components() -> None:
    by_source = _krea2_bundle_by_source()
    # Diffusers pipelines (self-contained), GGUF transformers, and the standalone VAE + encoder that the
    # GGUF transformers need.
    assert "krea/Krea-2-Turbo" in by_source
    assert "krea/Krea-2-Raw" in by_source
    assert any(s.endswith("krea2_turbo-Q4_K_M.gguf") for s in by_source)
    assert any(s.endswith("krea2_turbo-Q8_0.gguf") for s in by_source)
    # Standalone components (also declared as GGUF dependencies).
    assert any(m.type is ModelType.VAE for m in by_source.values())
    assert any(m.type is ModelType.Qwen3VLEncoder for m in by_source.values())


def test_krea2_diffusers_variants() -> None:
    by_source = _krea2_bundle_by_source()
    # Turbo (distilled) vs Raw (undistilled Base) must be tagged so defaults/scheduling differ.
    assert by_source["krea/Krea-2-Turbo"].variant is Krea2VariantType.Turbo
    assert by_source["krea/Krea-2-Raw"].variant is Krea2VariantType.Base


def test_krea2_gguf_entries_declare_vae_and_encoder_dependencies() -> None:
    gguf_models = [
        m
        for m in _krea2_bundle_by_source().values()
        if m.format is ModelFormat.GGUFQuantized and m.base is BaseModelType.Krea2
    ]
    assert len(gguf_models) == 2

    for model in gguf_models:
        assert model.variant is Krea2VariantType.Turbo
        assert model.dependencies is not None, f"{model.name} must declare its standalone dependencies"
        dep_types = {dep.type for dep in model.dependencies}
        # GGUF ships only the transformer -> it must pull a VAE and a Qwen3-VL encoder.
        assert ModelType.VAE in dep_types, f"{model.name} is missing a VAE dependency"
        assert ModelType.Qwen3VLEncoder in dep_types, f"{model.name} is missing a Qwen3-VL encoder dependency"


def test_krea2_bundle_models_are_registered_in_starter_models() -> None:
    starter_sources = {m.source for m in STARTER_MODELS}
    for model in STARTER_BUNDLES[BaseModelType.Krea2].models:
        assert model.source in starter_sources, f"{model.name} is not registered in STARTER_MODELS"


def test_krea2_gguf_dependency_models_are_registered_in_starter_models() -> None:
    # Every dependency source must itself be an installable starter model.
    starter_sources = {m.source for m in STARTER_MODELS}
    for model in STARTER_BUNDLES[BaseModelType.Krea2].models:
        for dep in model.dependencies or []:
            assert dep.source in starter_sources, f"dependency {dep.name} is not registered in STARTER_MODELS"
