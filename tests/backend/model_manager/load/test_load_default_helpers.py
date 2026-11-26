"""Two invariants that belong to model loading in general, not to any one loader.

Both were repeatedly got wrong per-loader, so they now live at the single choke point every loader
passes through (`put_in_eval_mode`) or in one shared resolver (`resolve_submodel_path`).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_FLUX_Config
from invokeai.backend.model_manager.load.load_default import put_in_eval_mode, resolve_submodel_path
from invokeai.backend.model_manager.load.model_cache.model_cache import CacheClearResult, ModelCache
from invokeai.backend.model_manager.taxonomy import FluxVariantType, ModelFormat, ModelType, SubModelType


class TestPutInEvalMode:
    def test_a_hand_built_module_tree_comes_back_in_inference_mode(self) -> None:
        """`init_empty_weights` + `load_state_dict` leaves `training` True, unlike `from_pretrained`."""
        module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(2))
        module.train()
        assert module.training

        returned = put_in_eval_mode(module)

        assert returned is module
        assert all(not child.training for child in module.modules())

    def test_it_is_idempotent(self) -> None:
        """Loaders that already call `.eval()` themselves must be unaffected."""
        module = torch.nn.Linear(2, 2).eval()

        assert not put_in_eval_mode(module).training

    @pytest.mark.parametrize(
        "value",
        [
            "a tokenizer stand-in",
            {"scheduler": "config"},
            SimpleNamespace(name="an IP-Adapter wrapper"),
            None,
        ],
    )
    def test_non_modules_pass_through_untouched(self, value) -> None:
        """Loaders also return tokenizers, schedulers and plain wrappers."""
        assert put_in_eval_mode(value) is value


class TestResolveSubmodelPath:
    def _config(self, submodels: dict | None) -> SimpleNamespace:
        return SimpleNamespace(path="/models/pipeline", submodels=submodels)

    def test_a_relative_record_follows_the_model_folder(self) -> None:
        """Components are recorded relative to the model, so moving the model moves them too."""
        config = self._config({SubModelType.TextEncoder: SimpleNamespace(path_or_prefix="clip_encoder")})

        resolved = resolve_submodel_path(config, SubModelType.TextEncoder, Path("/models/pipeline/text_encoder"))

        assert resolved == Path("/models/pipeline/clip_encoder")

    def test_it_rebases_a_legacy_absolute_record_after_the_pipeline_is_moved(self, tmp_path: Path) -> None:
        """Older records hold the identification-time path: the download's `tmpinstall_*` staging dir."""
        staging_root = tmp_path / "tmpinstall_pipeline"
        discovered_component = staging_root / "clip_encoder"
        discovered_component.mkdir(parents=True)

        installed_root = tmp_path / "models" / "pipeline"
        installed_root.parent.mkdir()
        staging_root.rename(installed_root)
        config = SimpleNamespace(
            # The loader makes `config.path` absolute before any component is resolved.
            path=installed_root.as_posix(),
            submodels={SubModelType.TextEncoder: SimpleNamespace(path_or_prefix=discovered_component.as_posix())},
        )

        resolved = resolve_submodel_path(config, SubModelType.TextEncoder, installed_root / "text_encoder")

        assert resolved == installed_root / "clip_encoder"

    def test_a_legacy_absolute_record_loses_to_the_models_own_component(self, tmp_path: Path) -> None:
        """A copy left at the recorded location (an install source, a copied models dir) may have
        been replaced since; the model's own folder is the one that belongs to this record."""
        elsewhere = tmp_path / "Downloads" / "pipeline" / "text_encoder"
        own = tmp_path / "models" / "key" / "text_encoder"
        elsewhere.mkdir(parents=True)
        own.mkdir(parents=True)
        config = SimpleNamespace(
            path=str(tmp_path / "models" / "key"),
            submodels={SubModelType.TextEncoder: SimpleNamespace(path_or_prefix=elsewhere.as_posix())},
        )

        assert resolve_submodel_path(config, SubModelType.TextEncoder, Path("/unused")) == own

    def test_a_legacy_record_of_a_symlinked_component_keeps_its_target(self, tmp_path: Path) -> None:
        """Identification resolved symlinks, so the record names the target, which the model folder lacks."""
        target = tmp_path / "shared" / "te_weights"
        target.mkdir(parents=True)
        model = tmp_path / "pipeline"
        model.mkdir()
        (model / "text_encoder").symlink_to(target)
        config = SimpleNamespace(
            path=str(model),
            submodels={SubModelType.TextEncoder: SimpleNamespace(path_or_prefix=target.as_posix())},
        )

        assert resolve_submodel_path(config, SubModelType.TextEncoder, Path("/unused")) == target

    def test_the_model_folder_is_the_root_whatever_the_fallback(self, tmp_path: Path) -> None:
        """The transformer loaders fall back to the model folder itself when it has no `transformer/`,
        so the fallback's parent is the models dir; the root must come from the config instead."""
        (tmp_path / "models" / "key" / "dit").mkdir(parents=True)
        (tmp_path / "models" / "dit").mkdir()  # a namesake beside the model must not be picked up
        model = tmp_path / "models" / "key"
        stale = (tmp_path / "models" / "tmpinstall_x" / "pipeline" / "dit").as_posix()
        config = SimpleNamespace(
            path=str(model), submodels={SubModelType.Transformer: SimpleNamespace(path_or_prefix=stale)}
        )

        assert resolve_submodel_path(config, SubModelType.Transformer, model) == model / "dit"

    def test_a_missing_component_is_reported_inside_the_model_folder(self, tmp_path: Path) -> None:
        """When neither location exists, the failure names the installed model's folder rather than a
        long-gone staging path, which Transformers would misreport as an invalid Hub repo id."""
        missing_component = tmp_path / "tmpinstall_x" / "clip_encoder"
        installed_root = tmp_path / "pipeline"
        installed_root.mkdir()
        config = SimpleNamespace(
            path=installed_root.as_posix(),
            submodels={SubModelType.TextEncoder: SimpleNamespace(path_or_prefix=missing_component.as_posix())},
        )

        resolved = resolve_submodel_path(config, SubModelType.TextEncoder, installed_root / "text_encoder")

        assert resolved == installed_root / "clip_encoder"

    def test_it_falls_back_when_the_slot_was_not_discovered(self) -> None:
        """Configs persisted before submodel discovery existed carry no map."""
        fallback = Path("/models/pipeline/text_encoder")

        assert resolve_submodel_path(self._config(None), SubModelType.TextEncoder, fallback) == fallback
        assert resolve_submodel_path(self._config({}), SubModelType.TextEncoder, fallback) == fallback

    def test_it_falls_back_for_a_slot_the_map_does_not_cover(self) -> None:
        """A partial pipeline records only some components; the rest keep the conventional name."""
        config = self._config({SubModelType.VAE: SimpleNamespace(path_or_prefix="/models/pipeline/autoencoder")})
        fallback = Path("/models/pipeline/text_encoder")

        assert resolve_submodel_path(config, SubModelType.TextEncoder, fallback) == fallback

    def test_it_tolerates_a_config_without_a_submodels_attribute(self) -> None:
        """Most model configs have no `submodels` field at all."""
        fallback = Path("/models/single-file")

        assert resolve_submodel_path(SimpleNamespace(path="/x"), SubModelType.Transformer, fallback) == fallback


class TestGetSizeFs:
    """Cache admission must size the folder loading will actually read.

    `make_room()` runs before the component is constructed, so a size of 0 reserves nothing and the
    real model read lands on top of whatever was already resident.
    """

    def _loader(self):
        from invokeai.backend.model_manager.load.load_default import ModelLoader

        return ModelLoader.__new__(ModelLoader)

    def _component(self, root: Path, name: str, size: int) -> Path:
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "model.safetensors").write_bytes(b"\x00" * size)
        return folder

    def test_it_sizes_a_nonstandard_component_key(self, tmp_path: Path) -> None:
        self._component(tmp_path, "clip_encoder", 4096)
        config = SimpleNamespace(
            path=str(tmp_path),
            repo_variant=None,
            submodels={SubModelType.TextEncoder: SimpleNamespace(path_or_prefix="clip_encoder")},
        )

        size = self._loader().get_size_fs(config, tmp_path, SubModelType.TextEncoder)

        assert size >= 4096

    def test_the_conventional_layout_is_unchanged(self, tmp_path: Path) -> None:
        self._component(tmp_path, "text_encoder", 2048)
        config = SimpleNamespace(path=str(tmp_path), repo_variant=None, submodels=None)

        size = self._loader().get_size_fs(config, tmp_path, SubModelType.TextEncoder)

        assert size >= 2048

    def test_a_missing_component_still_sizes_to_zero(self, tmp_path: Path) -> None:
        """A diffusers pipeline that merely lacks a subfolder must keep returning 0."""
        (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
        config = SimpleNamespace(path=str(tmp_path), repo_variant=None, submodels=None)

        assert self._loader().get_size_fs(config, tmp_path, SubModelType.TextEncoder) == 0

    def test_qwen_vl_checkpoint_tokenizer_does_not_size_the_checkpoint(self, tmp_path: Path) -> None:
        """The checkpoint is unrelated to the vendored tokenizer loaded for this component."""
        checkpoint = tmp_path / "qwen_vl.safetensors"
        checkpoint.write_bytes(b"\x00" * 8192)
        config = SimpleNamespace(
            path=str(checkpoint),
            repo_variant=None,
            type=ModelType.QwenVLEncoder,
            format=ModelFormat.Checkpoint,
        )

        assert self._loader().get_size_fs(config, checkpoint, SubModelType.Tokenizer) == 0
        assert self._loader().get_size_fs(config, checkpoint, SubModelType.TextEncoder) == checkpoint.stat().st_size

    @pytest.mark.parametrize("model_format", [ModelFormat.Checkpoint, ModelFormat.GGUFQuantized])
    def test_qwen_vl_single_file_admits_cold_vendored_tokenizer_parse_then_skips_warm_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_format: ModelFormat
    ) -> None:
        from invokeai.backend.model_manager.load.model_util import calc_tokenizer_size_by_fs
        from invokeai.backend.qwen2_5_vl import qwen2_5_vl_assets

        checkpoint = tmp_path / "qwen_vl.safetensors"
        checkpoint.write_bytes(b"\x00" * 8192)
        config = SimpleNamespace(
            path=str(checkpoint),
            repo_variant=None,
            type=ModelType.QwenVLEncoder,
            format=model_format,
        )
        loader = self._loader()
        cold_parse_bytes = calc_tokenizer_size_by_fs(qwen2_5_vl_assets._TOKENIZER_DIR, include_runtime_overhead=True)
        monkeypatch.setattr(
            qwen2_5_vl_assets.load_bundled_qwen2_5_vl_tokenizer, "cache_info", lambda: SimpleNamespace(currsize=0)
        )

        assert cold_parse_bytes > 100 * 2**20
        assert loader.get_size_fs(config, checkpoint, SubModelType.Tokenizer) == 0
        assert loader._get_host_bytes_needed(config, checkpoint, SubModelType.Tokenizer) == cold_parse_bytes

        monkeypatch.setattr(
            qwen2_5_vl_assets.load_bundled_qwen2_5_vl_tokenizer, "cache_info", lambda: SimpleNamespace(currsize=1)
        )
        assert loader.get_size_fs(config, checkpoint, SubModelType.Tokenizer) == 0
        assert loader._get_host_bytes_needed(config, checkpoint, SubModelType.Tokenizer) == 0

    def test_flux_bundle_sizes_only_the_selected_transformer_payload(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "flux-bundle.safetensors"
        marker = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
        save_file(
            {
                marker: torch.zeros(4),
                "model.diffusion_model.img_in.weight": torch.zeros(32),
                "text_encoders.clip.weight": torch.zeros(4096),
                "vae.decoder.weight": torch.zeros(4096),
            },
            checkpoint,
        )
        config = Main_Checkpoint_FLUX_Config.model_construct(
            path=str(checkpoint), variant=FluxVariantType.Dev, repo_variant=None, name="bundle"
        )

        selected = self._loader().get_size_fs(config, checkpoint, SubModelType.Transformer)

        assert selected == (4 + 32) * torch.float32.itemsize

    def test_flux_bundle_reserves_selected_scale_conversion_bytes_separately(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "flux-bundle.safetensors"
        marker = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
        save_file(
            {
                marker: torch.zeros(4),
                "model.diffusion_model.double_blocks.0.img_attn.norm.query.scale": torch.zeros(32),
                "text_encoders.clip.scale": torch.zeros(4096),
                "vae.decoder.scale": torch.zeros(4096),
            },
            checkpoint,
        )
        config = Main_Checkpoint_FLUX_Config.model_construct(
            path=str(checkpoint), variant=FluxVariantType.Dev, repo_variant=None
        )

        selected_source_bytes = (4 + 32) * torch.float32.itemsize
        assert self._loader()._get_host_bytes_needed(config, checkpoint, SubModelType.Transformer) == (
            selected_source_bytes + (4 + 32) * 2
        )

    def test_flux_non_bundle_does_not_reserve_scale_conversion_bytes(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "flux.safetensors"
        save_file({"double_blocks.0.img_attn.norm.key_norm.scale": torch.zeros(32)}, checkpoint)
        config = Main_Checkpoint_FLUX_Config.model_construct(
            path=str(checkpoint), variant=FluxVariantType.Dev, repo_variant=None
        )

        assert self._loader()._get_host_bytes_needed(config, checkpoint, SubModelType.Transformer) == 0

    def test_flux_bundle_marker_counts_scale_conversion_even_when_all_keys_are_selected(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "flux-bundle.safetensors"
        marker = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
        save_file(
            {
                marker: torch.zeros(4),
                "model.diffusion_model.single_blocks.0.linear.scale": torch.zeros(8),
            },
            checkpoint,
        )
        config = Main_Checkpoint_FLUX_Config.model_construct(
            path=str(checkpoint), variant=FluxVariantType.Dev, repo_variant=None
        )

        selected_source_bytes = (4 + 8) * torch.float32.itemsize
        assert self._loader()._get_host_bytes_needed(config, checkpoint, SubModelType.Transformer) == (
            selected_source_bytes + (4 + 8) * torch.bfloat16.itemsize
        )

    def test_flux_bundle_passes_total_host_peak_to_initial_admission(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "flux-bundle.safetensors"
        marker = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
        save_file(
            {
                marker: torch.zeros(4),
                "model.diffusion_model.img_in.weight": torch.zeros(32),
                "model.diffusion_model.single_blocks.0.linear.scale": torch.zeros(8),
                "text_encoders.clip.weight": torch.zeros(4096),
            },
            checkpoint,
        )
        config = Main_Checkpoint_FLUX_Config.model_construct(
            path=str(checkpoint), variant=FluxVariantType.Dev, repo_variant=None, name="bundle"
        )
        loader = self._loader()
        loader._ram_cache = MagicMock()
        loader._ram_cache.shared_cpu_weights = None
        loader._ram_cache.get_with_first_use_claim.side_effect = [IndexError, IndexError, (MagicMock(), None)]
        loader._ram_cache.make_room.return_value = CacheClearResult(0, 0, False, True)
        loader._ram_cache.put.return_value = MagicMock()
        loader._app_config = SimpleNamespace(log_memory_usage=False)
        loader._logger = MagicMock()
        loader.config_check = None
        loader._get_model_path = lambda _config: checkpoint  # type: ignore[method-assign]
        loader._get_execution_device = lambda *_args: torch.device("cpu")  # type: ignore[method-assign]
        loader._load_model = MagicMock(return_value=object())  # type: ignore[method-assign]

        loader._load_and_cache(config, SubModelType.Transformer)

        selected = loader.get_size_fs(config, checkpoint, SubModelType.Transformer)
        converted = (4 + 8) * torch.bfloat16.itemsize
        loader._ram_cache.make_room.assert_called_once_with(selected, host_bytes_needed=selected + converted)


def test_cold_load_refuses_before_model_construction_when_host_headroom_is_insufficient(tmp_path: Path) -> None:
    """A fixed cache cap must not let a large CPU construction run into a smaller live host budget."""
    from invokeai.backend.model_manager.load.load_default import ModelLoader

    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
    )
    loader = ModelLoader.__new__(ModelLoader)
    loader._ram_cache = cache
    loader._app_config = SimpleNamespace(log_memory_usage=False)
    loader._logger = MagicMock()
    loader._get_model_path = lambda _config: tmp_path  # type: ignore[method-assign]
    loader.get_size_fs = lambda *_args: cache.local_ram_cache_size_bytes + 1  # type: ignore[method-assign]
    loader._load_model = MagicMock(return_value=object())  # type: ignore[method-assign]
    cache.make_room = MagicMock(return_value=CacheClearResult(0, 0, False, False))  # type: ignore[method-assign]
    config = SimpleNamespace(
        base="base",
        type="type",
        name="model",
        key="model",
        path=str(tmp_path),
        repo_variant=None,
        submodels=None,
    )

    try:
        with pytest.raises(MemoryError, match="host memory"):
            loader._load_and_cache(config)
        loader._load_model.assert_not_called()  # type: ignore[attr-defined]
    finally:
        cache.shutdown()
