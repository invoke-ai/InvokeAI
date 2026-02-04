"""
Tests for missing model detection (_scan_for_missing_models) and bulk deletion.
"""

import gc
from pathlib import Path

import pytest

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_install import ModelInstallServiceBase
from invokeai.app.services.model_records import UnknownModelException
from invokeai.backend.model_manager.configs.textual_inversion import TI_File_SD1_Config
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelSourceType,
    ModelType,
)
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403


class TestScanForMissingModels:
    """Tests for ModelInstallService._scan_for_missing_models()."""

    def test_no_missing_models(
        self, mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
    ) -> None:
        """When all registered models exist on disk, _scan_for_missing_models returns an empty list."""
        mm2_installer.register_path(embedding_file)
        missing = mm2_installer._scan_for_missing_models()
        assert len(missing) == 0

    def test_detects_missing_model(
        self, mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
    ) -> None:
        """A model whose path does not exist on disk is reported as missing."""
        # Register a real model first, then add a fake one with a non-existent path
        mm2_installer.register_path(embedding_file)

        fake_config = TI_File_SD1_Config(
            key="missing-model-key-1",
            path="/nonexistent/path/missing_model.safetensors",
            name="MissingModel",
            base=BaseModelType.StableDiffusion1,
            type=ModelType.TextualInversion,
            format=ModelFormat.EmbeddingFile,
            hash="FAKEHASH1",
            file_size=1024,
            source="test/source",
            source_type=ModelSourceType.Path,
        )
        mm2_installer.record_store.add_model(fake_config)

        missing = mm2_installer._scan_for_missing_models()
        assert len(missing) == 1
        assert missing[0].key == "missing-model-key-1"

    def test_mix_of_existing_and_missing(
        self,
        mm2_installer: ModelInstallServiceBase,
        embedding_file: Path,
        diffusers_dir: Path,
        mm2_app_config: InvokeAIAppConfig,
    ) -> None:
        """With multiple models, only the ones with missing files are returned."""
        key_existing = mm2_installer.register_path(embedding_file)
        mm2_installer.register_path(diffusers_dir)

        # Add two models with non-existent paths
        fake1 = TI_File_SD1_Config(
            key="missing-key-1",
            path="/nonexistent/missing1.safetensors",
            name="Missing1",
            base=BaseModelType.StableDiffusion1,
            type=ModelType.TextualInversion,
            format=ModelFormat.EmbeddingFile,
            hash="FAKEHASH_A",
            file_size=1024,
            source="test/source1",
            source_type=ModelSourceType.Path,
        )
        fake2 = TI_File_SD1_Config(
            key="missing-key-2",
            path="/nonexistent/missing2.safetensors",
            name="Missing2",
            base=BaseModelType.StableDiffusion1,
            type=ModelType.TextualInversion,
            format=ModelFormat.EmbeddingFile,
            hash="FAKEHASH_B",
            file_size=2048,
            source="test/source2",
            source_type=ModelSourceType.Path,
        )
        mm2_installer.record_store.add_model(fake1)
        mm2_installer.record_store.add_model(fake2)

        missing = mm2_installer._scan_for_missing_models()
        missing_keys = {m.key for m in missing}
        assert len(missing) == 2
        assert "missing-key-1" in missing_keys
        assert "missing-key-2" in missing_keys
        assert key_existing not in missing_keys

    def test_empty_store_returns_empty(self, mm2_installer: ModelInstallServiceBase) -> None:
        """With no models registered, _scan_for_missing_models returns an empty list."""
        missing = mm2_installer._scan_for_missing_models()
        assert len(missing) == 0


class TestBulkDelete:
    """Tests for bulk model deletion."""

    def test_delete_installed_model(
        self, mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
    ) -> None:
        """Deleting an installed model removes it from the store and disk."""
        key = mm2_installer.install_path(embedding_file)
        record = mm2_installer.record_store.get_model(key)
        model_path = mm2_app_config.models_path / record.path
        assert model_path.exists()
        assert mm2_installer.record_store.exists(key)

        gc.collect()
        mm2_installer.delete(key)

        with pytest.raises(UnknownModelException):
            mm2_installer.record_store.get_model(key)

    def test_unregister_missing_model(
        self, mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
    ) -> None:
        """Unregistering a model whose file is missing removes it from the DB."""
        fake_config = TI_File_SD1_Config(
            key="missing-to-delete",
            path="/nonexistent/path/gone.safetensors",
            name="GoneModel",
            base=BaseModelType.StableDiffusion1,
            type=ModelType.TextualInversion,
            format=ModelFormat.EmbeddingFile,
            hash="FAKEHASH_GONE",
            file_size=1024,
            source="test/source",
            source_type=ModelSourceType.Path,
        )
        mm2_installer.record_store.add_model(fake_config)
        assert mm2_installer.record_store.exists("missing-to-delete")

        # Unregister removes it from DB without touching disk
        mm2_installer.unregister("missing-to-delete")

        with pytest.raises(UnknownModelException):
            mm2_installer.record_store.get_model("missing-to-delete")

    def test_delete_unknown_key_raises(self, mm2_installer: ModelInstallServiceBase) -> None:
        """Deleting a model with an unknown key raises UnknownModelException."""
        with pytest.raises(UnknownModelException):
            mm2_installer.delete("nonexistent-key-12345")

    def test_scan_then_unregister_clears_missing(
        self, mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
    ) -> None:
        """After unregistering all missing models, _scan_for_missing_models returns empty."""
        # Add two models with non-existent paths
        for i in range(2):
            config = TI_File_SD1_Config(
                key=f"missing-bulk-{i}",
                path=f"/nonexistent/bulk_{i}.safetensors",
                name=f"BulkMissing{i}",
                base=BaseModelType.StableDiffusion1,
                type=ModelType.TextualInversion,
                format=ModelFormat.EmbeddingFile,
                hash=f"BULKHASH{i}",
                file_size=1024,
                source=f"test/bulk{i}",
                source_type=ModelSourceType.Path,
            )
            mm2_installer.record_store.add_model(config)

        missing = mm2_installer._scan_for_missing_models()
        assert len(missing) == 2

        # Unregister all missing (simulates bulk delete for missing models)
        for model in missing:
            mm2_installer.unregister(model.key)

        assert len(mm2_installer._scan_for_missing_models()) == 0

    def test_bulk_unregister_does_not_affect_existing_models(
        self,
        mm2_installer: ModelInstallServiceBase,
        embedding_file: Path,
        mm2_app_config: InvokeAIAppConfig,
    ) -> None:
        """Unregistering missing models does not affect models that exist on disk."""
        existing_key = mm2_installer.register_path(embedding_file)

        fake_config = TI_File_SD1_Config(
            key="missing-selective",
            path="/nonexistent/selective.safetensors",
            name="SelectiveMissing",
            base=BaseModelType.StableDiffusion1,
            type=ModelType.TextualInversion,
            format=ModelFormat.EmbeddingFile,
            hash="SELECTIVEHASH",
            file_size=1024,
            source="test/selective",
            source_type=ModelSourceType.Path,
        )
        mm2_installer.record_store.add_model(fake_config)

        # Only unregister the missing one
        missing = mm2_installer._scan_for_missing_models()
        assert len(missing) == 1
        for model in missing:
            mm2_installer.unregister(model.key)

        # Existing model should still be there
        assert mm2_installer.record_store.exists(existing_key)
        assert len(mm2_installer._scan_for_missing_models()) == 0
