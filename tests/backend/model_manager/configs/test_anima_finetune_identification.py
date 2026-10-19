"""Identification of the depth-expanded Anima finetunes and of the Qwen3.5 encoder Anima-3.8B reads.

Written against small safetensors files carrying the real key names (and, where identification reads
it, the real header), because identification reads both through `ModelOnDisk`.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Anima_Config, _get_anima_variant
from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_Checkpoint_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import AnimaVariantType, Qwen35VariantType
from tests.backend.model_manager.load.state_dicts.anima_3_8b_connector_keys import metadata as anima_3_8b_metadata

_ANIMA_KEYS = [
    "net.llm_adapter.blocks.0.cross_attn.k_norm.weight",
    "net.blocks.0.adaln_modulation_cross_attn.1.weight",
    "net.t_embedder.1.linear_1.weight",
    "net.x_embedder.proj.1.weight",
    "net.final_layer.adaln_modulation.1.weight",
]
_CONNECTOR_KEY = "net.anima_v2_connector.quality_anchor.layer_mix_logits"


def _write(path: Path, keys: dict[str, list[int]], metadata: dict[str, str] | None = None) -> ModelOnDisk:
    save_file({k: torch.zeros(shape) for k, shape in keys.items()}, path, metadata=metadata)
    return ModelOnDisk(path)


def _identify(mod: ModelOnDisk, override_fields: dict | None = None) -> AnyModelConfig | None:
    """Identify through the real factory: every config class gets its say, so a double match shows."""
    result = ModelConfigFactory.from_model_on_disk(mod, override_fields, allow_unknown=False)
    matched = [type(v).__name__ for v in result.details.values() if not isinstance(v, Exception)]
    assert len(matched) <= 1, f"more than one config matched: {matched}"
    return result.config


def _anima(tmp_path: Path, *, connector: bool, metadata: dict[str, str] | None = None) -> ModelOnDisk:
    keys = {k: [1] for k in _ANIMA_KEYS}
    if connector:
        keys[_CONNECTOR_KEY] = [6, 4]
    return _write(tmp_path / "anima.safetensors", keys, metadata)


class TestAnimaVariant:
    @pytest.mark.parametrize("prefix", ["", "net.", "model.diffusion_model."])
    def test_a_bundled_connector_needs_qwen3_5(self, prefix: str) -> None:
        sd = {f"{prefix}blocks.0.mlp.layer1.weight": None, f"{prefix}anima_v2_connector.query_tokens": None}
        assert _get_anima_variant(sd) is AnimaVariantType.Qwen35

    def test_everything_else_is_qwen3_only(self) -> None:
        # Any depth: the official 28 blocks and Anima-2.9B's 40 are the same variant.
        assert _get_anima_variant({"net.blocks.39.mlp.layer1.weight": None}) is AnimaVariantType.Qwen3

    def test_official_release_and_2_9b_identify_as_qwen3(self, tmp_path: Path) -> None:
        config = _identify(_anima(tmp_path, connector=False))
        assert isinstance(config, Main_Checkpoint_Anima_Config)
        assert config.variant is AnimaVariantType.Qwen3

    def test_3_8b_v1_1_identifies_as_qwen35(self, tmp_path: Path) -> None:
        mod = _anima(tmp_path, connector=True, metadata=anima_3_8b_metadata)
        config = _identify(mod)
        assert isinstance(config, Main_Checkpoint_Anima_Config)
        assert config.variant is AnimaVariantType.Qwen35
        # Default settings follow the variant (the reference workflow's CFG 6 / 40 steps).
        assert config.default_settings is not None
        assert (config.default_settings.steps, config.default_settings.cfg_scale) == (40, 6.0)

    def test_3_8b_v1_0_without_its_adapter_is_refused_not_installed(self, tmp_path: Path) -> None:
        # The v1.0 DiT is a plain 52-block Anima on the outside; only its header says it was trained
        # jointly with a Qwen3.5 adapter it does not carry.
        mod = _anima(tmp_path, connector=False, metadata={"qwen35_joint_dit_blocks": "[0, 1, 2]"})
        result = ModelConfigFactory.from_model_on_disk(mod, allow_unknown=True)
        # Final, not "unknown": nothing is written to the database.
        assert result.config is None
        refusal = result.details[Main_Checkpoint_Anima_Config.__name__]
        assert isinstance(refusal, InvalidMatchError)
        assert "v1.0" in str(refusal)

    def test_an_explicit_variant_override_wins(self, tmp_path: Path) -> None:
        mod = _anima(tmp_path, connector=False)
        config = _identify(mod, {"variant": AnimaVariantType.Qwen35})
        assert isinstance(config, Main_Checkpoint_Anima_Config)
        assert config.variant is AnimaVariantType.Qwen35


# The keys of Anima-3.8B's `qwen35_4b.safetensors` that identification reads, at token extents except
# where the width is the signal.
_QWEN3_5_KEYS = {
    "embed_tokens.weight": [4, 2560],
    "layers.0.linear_attn.in_proj_qkv.weight": [4, 4],
    "layers.3.self_attn.q_proj.weight": [4, 4],
    "layers.3.self_attn.q_norm.weight": [4],
}


class TestQwen35Encoder:
    @pytest.mark.parametrize("prefix", ["", "model.", "model.language_model."])
    def test_qwen3_5_4b_is_identified(self, tmp_path: Path, prefix: str) -> None:
        mod = _write(tmp_path / "q35.safetensors", {f"{prefix}{k}": s for k, s in _QWEN3_5_KEYS.items()})
        config = _identify(mod)
        assert isinstance(config, Qwen35Encoder_Checkpoint_Config)
        assert config.variant is Qwen35VariantType.Qwen35_4B

    def test_an_unknown_width_is_not_a_match(self, tmp_path: Path) -> None:
        keys = {**_QWEN3_5_KEYS, "embed_tokens.weight": [4, 1024]}
        assert _identify(_write(tmp_path / "q35.safetensors", keys)) is None

    def test_a_qwen3_encoder_is_not_a_match(self, tmp_path: Path) -> None:
        keys = {"model.embed_tokens.weight": [4, 2560], "model.layers.0.self_attn.q_norm.weight": [4]}
        assert isinstance(_identify(_write(tmp_path / "q3.safetensors", keys)), Qwen3Encoder_Checkpoint_Config)

    def test_the_qwen3_config_refuses_qwen3_5_at_the_same_width(self, tmp_path: Path) -> None:
        # Both 4Bs are 2560 wide and a `model.`-prefixed Qwen3.5 export satisfies every Qwen3 heuristic;
        # only the linear-attention layers tell them apart.
        mod = _write(tmp_path / "q35.safetensors", {f"model.{k}": s for k, s in _QWEN3_5_KEYS.items()})
        assert isinstance(_identify(mod), Qwen35Encoder_Checkpoint_Config)
