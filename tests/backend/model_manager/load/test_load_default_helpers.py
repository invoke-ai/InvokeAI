"""Two invariants that belong to model loading in general, not to any one loader.

Both were repeatedly got wrong per-loader, so they now live at the single choke point every loader
passes through (`put_in_eval_mode`) or in one shared resolver (`resolve_submodel_path`).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.model_manager.load.load_default import put_in_eval_mode, resolve_submodel_path
from invokeai.backend.model_manager.taxonomy import SubModelType


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

    def test_it_prefers_the_path_discovery_recorded(self) -> None:
        """`model_index.json` may name a component anything; discovery records the folder it found."""
        config = self._config(
            {SubModelType.TextEncoder: SimpleNamespace(path_or_prefix="/models/pipeline/clip_encoder")}
        )

        resolved = resolve_submodel_path(config, SubModelType.TextEncoder, Path("/models/pipeline/text_encoder"))

        assert resolved == Path("/models/pipeline/clip_encoder")

    def test_it_rebases_a_discovered_path_after_the_pipeline_is_moved(self, tmp_path: Path) -> None:
        """Remote installs identify a pipeline in a staging dir before moving it into models/."""
        staging_root = tmp_path / "tmpinstall_pipeline"
        discovered_component = staging_root / "clip_encoder"
        discovered_component.mkdir(parents=True)

        installed_root = tmp_path / "models" / "pipeline"
        installed_root.parent.mkdir()
        staging_root.rename(installed_root)
        config = SimpleNamespace(
            # Managed model records store paths relative to the models directory. The loader's
            # fallback carries the resolved absolute model root.
            path=installed_root.name,
            submodels={SubModelType.TextEncoder: SimpleNamespace(path_or_prefix=discovered_component.as_posix())},
        )

        resolved = resolve_submodel_path(config, SubModelType.TextEncoder, installed_root / "text_encoder")

        assert resolved == installed_root / "clip_encoder"

    def test_it_keeps_an_existing_discovered_path_outside_the_pipeline(self, tmp_path: Path) -> None:
        """A valid external component must not be replaced merely because the pipeline has a namesake."""
        external_component = tmp_path / "shared" / "clip_encoder"
        external_component.mkdir(parents=True)
        installed_component = tmp_path / "pipeline" / "clip_encoder"
        installed_component.mkdir(parents=True)
        config = SimpleNamespace(
            path=installed_component.parent.as_posix(),
            submodels={SubModelType.TextEncoder: SimpleNamespace(path_or_prefix=external_component.as_posix())},
        )

        resolved = resolve_submodel_path(config, SubModelType.TextEncoder, installed_component.parent / "text_encoder")

        assert resolved == external_component

    def test_it_keeps_a_missing_discovered_path_when_no_relocated_component_exists(self, tmp_path: Path) -> None:
        """Do not hide a genuinely missing component behind an invented path under the model root."""
        missing_component = tmp_path / "staging" / "clip_encoder"
        installed_root = tmp_path / "pipeline"
        installed_root.mkdir()
        config = SimpleNamespace(
            path=installed_root.as_posix(),
            submodels={SubModelType.TextEncoder: SimpleNamespace(path_or_prefix=missing_component.as_posix())},
        )

        resolved = resolve_submodel_path(config, SubModelType.TextEncoder, installed_root / "text_encoder")

        assert resolved == missing_component

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
    real multi-GB read lands on top of whatever was already resident.
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
            submodels={
                SubModelType.TextEncoder: SimpleNamespace(path_or_prefix=(tmp_path / "clip_encoder").as_posix())
            },
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
