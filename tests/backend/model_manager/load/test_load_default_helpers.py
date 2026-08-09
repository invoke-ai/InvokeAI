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
