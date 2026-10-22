"""Tests for Qwen3-VL text-encoder identification (used by Krea-2).

A single-file Qwen3-VL encoder is distinguished from the text-only ``Qwen3Encoder`` (Z-Image /
FLUX.2 Klein) by the presence of the Qwen3-VL **visual tower** (``visual.*`` / ``model.visual.*``).
Both have a Qwen3 text decoder (``model.layers.*``), so the visual tower is the deciding signal.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_Checkpoint_Config,
    Qwen3VLEncoder_Qwen3VLEncoder_Config,
    _is_qwen3_vl_encoder_state_dict,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, Qwen3VLVariantType
from tests.fixtures.quantized_payloads import comfy_quant_marker, nvfp4_signed_tensors, quantize_scaled_fp8

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/qwen3vl.safetensors",
    "file_size": 1000,
    "name": "qwen3vl-encoder",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


class TestIsQwen3VLEncoderStateDict:
    def test_text_decoder_plus_visual_tower_matches(self) -> None:
        # ComfyUI single-file layout (implicit LM prefix): model.layers.* + model.visual.*
        sd = {
            "model.layers.0.self_attn.q_proj.weight": object(),
            "model.visual.blocks.0.attn.qkv.weight": object(),
        }
        assert _is_qwen3_vl_encoder_state_dict(sd) is True

    def test_explicit_language_model_prefix_matches(self) -> None:
        # Alternative single-file layout (explicit LM prefix): model.language_model.layers.* + model.visual.*
        sd = {
            "model.language_model.layers.0.self_attn.q_proj.weight": object(),
            "model.visual.blocks.0.attn.qkv.weight": object(),
        }
        assert _is_qwen3_vl_encoder_state_dict(sd) is True

    def test_text_only_decoder_does_not_match(self) -> None:
        # Z-Image / FLUX.2 Klein Qwen3 text encoder: text decoder but NO visual tower.
        sd = {
            "model.layers.0.self_attn.q_proj.weight": object(),
            "model.layers.0.mlp.down_proj.weight": object(),
            "model.norm.weight": object(),
        }
        assert _is_qwen3_vl_encoder_state_dict(sd) is False

    def test_visual_tower_only_does_not_match(self) -> None:
        sd = {"model.visual.blocks.0.attn.qkv.weight": object()}
        assert _is_qwen3_vl_encoder_state_dict(sd) is False

    def test_ignores_non_string_keys(self) -> None:
        sd: dict = {0: object(), 1: object()}
        assert _is_qwen3_vl_encoder_state_dict(sd) is False


class TestQwen3VLEncoderCheckpointConfig:
    def _make_mock_mod(self, state_dict: dict, suffix: str = ".safetensors") -> MagicMock:
        mod = MagicMock()
        mod.path = Path(f"/fake/qwen3vl{suffix}")
        mod.load_state_dict.return_value = state_dict
        return mod

    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_for_override_fields")
    def test_matches_vl_single_file(self, _rfo, _rif) -> None:
        mod = self._make_mock_mod(
            {
                "model.embed_tokens.weight": MagicMock(shape=(151936, 2560)),
                "model.layers.35.self_attn.q_proj.weight": object(),
                "model.visual.blocks.0.attn.qkv.weight": object(),
            }
        )
        config = Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.type == ModelType.Qwen3VLEncoder
        assert config.base == BaseModelType.Any
        assert config.format == ModelFormat.Checkpoint
        assert config.variant is Qwen3VLVariantType.Qwen3VL_4B

    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_for_override_fields")
    def test_rejects_text_only_encoder(self, _rfo, _rif) -> None:
        mod = self._make_mock_mod(
            {
                "model.layers.0.self_attn.q_proj.weight": object(),
                "model.norm.weight": object(),
            }
        )
        with pytest.raises(NotAMatchError):
            Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})

    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_for_override_fields")
    def test_rejects_non_safetensors_checkpoint(self, _rfo, _rif) -> None:
        mod = self._make_mock_mod(
            {
                "model.layers.35.self_attn.q_proj.weight": object(),
                "model.visual.blocks.0.attn.qkv.weight": object(),
            },
            suffix=".bin",
        )

        with pytest.raises(NotAMatchError, match="safetensors"):
            Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})

    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_for_override_fields")
    def test_records_the_8b_variant(self, _rfo, _rif) -> None:
        # Ideogram 4's encoder. Same type, same layout, twice the width -- and the loader builds the
        # architecture from a HuggingFace config chosen by this variant, so a mislabelled one fails
        # on shapes thousands of tensors in.
        mod = self._make_mock_mod(
            {
                "model.embed_tokens.weight": MagicMock(shape=(151936, 4096)),
                "model.layers.35.self_attn.q_proj.weight": object(),
                "model.visual.blocks.0.attn.qkv.weight": object(),
            }
        )

        config = Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})

        assert config.variant is Qwen3VLVariantType.Qwen3VL_8B

    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_for_override_fields")
    def test_rejects_an_unsupported_width(self, _rfo, _rif) -> None:
        # Qwen3-VL-32B (hidden 5120): a real encoder with no architecture here. Only MiniMax H3's
        # truncated build of it is supported, and that has its own config class.
        mod = self._make_mock_mod(
            {
                "model.embed_tokens.weight": MagicMock(shape=(151936, 5120)),
                "model.layers.35.self_attn.q_proj.weight": object(),
                "model.visual.blocks.0.attn.qkv.weight": object(),
            }
        )

        with pytest.raises(NotAMatchError, match="hidden size"):
            Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})

    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_for_override_fields")
    def test_rejects_a_truncated_language_stack(self, _rfo, _rif) -> None:
        mod = self._make_mock_mod(
            {
                "model.embed_tokens.weight": MagicMock(shape=(151936, 4096)),
                "model.layers.20.self_attn.q_proj.weight": object(),
                "model.visual.blocks.0.attn.qkv.weight": object(),
            }
        )

        with pytest.raises(NotAMatchError, match="layer 35"):
            Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})


_PACKED = "model.layers.0.self_attn.q_proj"


class TestQwen3VLEncoderQuantizedCheckpoints:
    """What the loader would refuse of a quantized single file is refused at install time; the rest installs.

    Written as real files and read through `ModelOnDisk`, because identification sees tensors on the meta
    device -- dtypes and shapes, no bytes -- so whatever names a layer nvfp4 has to be read from the file.
    """

    @classmethod
    def _identify(
        cls, tmp_path: Path, mutate=lambda _sd: None, *, naming: str = "marker"
    ) -> Qwen3VLEncoder_Checkpoint_Config:
        packed, _ = nvfp4_signed_tensors(_PACKED, torch.ones(128, 64, dtype=torch.bool))
        state_dict = {
            "model.embed_tokens.weight": torch.zeros(2, 2560, dtype=torch.bfloat16),
            "model.layers.35.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.visual.blocks.0.attn.qkv.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
            **packed,
        }
        metadata = None
        if naming == "marker":
            state_dict[f"{_PACKED}.comfy_quant"] = comfy_quant_marker({"format": "nvfp4"})
        else:
            metadata = {"_quantization_metadata": json.dumps({"layers": {_PACKED: {"format": "nvfp4"}}})}
        mutate(state_dict)
        path = tmp_path / "qwen3vl_4b.safetensors"
        save_file(state_dict, str(path), metadata=metadata)
        return Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(ModelOnDisk(path), {**_REQUIRED_FIELDS})

    @pytest.mark.parametrize("naming", ["marker", "header"])
    def test_a_comfy_nvfp4_build_installs(self, tmp_path: Path, naming: str) -> None:
        assert self._identify(tmp_path, naming=naming).variant is Qwen3VLVariantType.Qwen3VL_4B

    def _replace_packed_layer(self, state_dict: dict, **tensors: torch.Tensor) -> None:
        for key in [key for key in state_dict if key.startswith(f"{_PACKED}.")]:
            del state_dict[key]
        state_dict.update({f"{_PACKED}.{suffix}": tensor for suffix, tensor in tensors.items()})

    def test_scaled_fp8_and_int8_builds_still_install(self, tmp_path: Path) -> None:
        # Both pair a weight with a `weight_scale` too, and neither is 4-bit.
        fp8 = quantize_scaled_fp8(torch.randn(128, 64))
        assert (
            self._identify(
                tmp_path, lambda sd: self._replace_packed_layer(sd, weight=fp8.codes, weight_scale=fp8.scale)
            ).variant
            is Qwen3VLVariantType.Qwen3VL_4B
        )
        int8 = {
            "weight": torch.randint(-127, 128, (128, 64), dtype=torch.int8),
            "weight_scale": torch.ones(128, 1),
            "comfy_quant": comfy_quant_marker({"format": "int8_tensorwise"}),
        }
        assert (
            self._identify(tmp_path, lambda sd: self._replace_packed_layer(sd, **int8)).variant
            is Qwen3VLVariantType.Qwen3VL_4B
        )

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            # ModelOpt-style: the nvfp4 tensors, but nothing naming the layout ComfyUI's.
            pytest.param(lambda sd: sd.pop(f"{_PACKED}.comfy_quant"), "no ComfyUI `comfy_quant` marker", id="unnamed"),
            # A custom RTN repack published for this encoder: packed codes and a block scale only.
            pytest.param(lambda sd: sd.pop(f"{_PACKED}.weight_scale_2"), "no weight_scale_2", id="no-global-scale"),
            pytest.param(lambda sd: sd.update({f"{_PACKED}.pre_quant_scale": torch.ones(64)}), "AWQ", id="awq"),
        ],
    )
    def test_a_layer_the_loader_would_refuse_is_refused_at_install(self, tmp_path: Path, mutate, message: str) -> None:
        with pytest.raises(InvalidMatchError, match=message):
            self._identify(tmp_path, mutate)

    def test_a_repack_packing_its_embedding_too_is_refused_rather_than_unknown(self, tmp_path: Path) -> None:
        # The variant is read off the embedding's width, which a packed table halves. Checked after that, such a
        # file would fail there instead and register as an unknown model, its reason only in the log.
        def packed_embedding_without_global_scale(state_dict: dict) -> None:
            state_dict["model.embed_tokens.weight"] = torch.zeros(2, 1280, dtype=torch.uint8)
            state_dict["model.embed_tokens.weight_scale"] = torch.zeros(2, 160).to(torch.float8_e4m3fn)

        with pytest.raises(InvalidMatchError, match="no weight_scale_2"):
            self._identify(tmp_path, packed_embedding_without_global_scale)

    def test_a_layer_the_loader_never_reads_cannot_get_a_file_refused(self, tmp_path: Path) -> None:
        # The loader drops the visual tower and the LM head before it reads a scale, so their layout is not this
        # file's to answer for.
        def unread_half_state_layers(state_dict: dict) -> None:
            for path in ("model.visual.blocks.0.mlp.linear_fc1", "lm_head"):
                state_dict[f"{path}.weight"] = torch.zeros(128, 32, dtype=torch.uint8)
                state_dict[f"{path}.weight_scale"] = torch.zeros(128, 4).to(torch.float8_e4m3fn)

        assert self._identify(tmp_path, unread_half_state_layers).variant is Qwen3VLVariantType.Qwen3VL_4B


class TestQwen3VLEncoderDirectoryConfig:
    @staticmethod
    def _write_config(path: Path, *, hidden_size: int = 2560, num_hidden_layers: int = 36) -> None:
        path.write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3VLModel"],
                    "text_config": {"hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers},
                }
            )
        )

    @staticmethod
    def _fields(path: Path) -> dict:
        return {**_REQUIRED_FIELDS, "path": path.as_posix()}

    def test_accepts_direct_layout_with_weights_and_tokenizer(self, tmp_path: Path) -> None:
        self._write_config(tmp_path / "config.json")
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "tokenizer.json").touch()

        config = Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

        assert config.type is ModelType.Qwen3VLEncoder

    @pytest.mark.parametrize(("include_weights", "include_tokenizer"), [(False, True), (True, False), (False, False)])
    def test_rejects_incomplete_direct_layout(
        self, tmp_path: Path, include_weights: bool, include_tokenizer: bool
    ) -> None:
        self._write_config(tmp_path / "config.json")
        if include_weights:
            (tmp_path / "model.safetensors").touch()
        if include_tokenizer:
            (tmp_path / "tokenizer.json").touch()

        with pytest.raises(NotAMatchError, match="weights|tokenizer"):
            Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

    def test_accepts_nested_layout_with_weights_and_tokenizer(self, tmp_path: Path) -> None:
        text_encoder = tmp_path / "text_encoder"
        tokenizer = tmp_path / "tokenizer"
        text_encoder.mkdir()
        tokenizer.mkdir()
        self._write_config(text_encoder / "config.json")
        (text_encoder / "model.safetensors").touch()
        (tokenizer / "tokenizer.json").touch()

        config = Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

        assert config.format is ModelFormat.Qwen3VLEncoder

    def test_records_the_8b_variant_from_the_directory_config(self, tmp_path: Path) -> None:
        self._write_config(tmp_path / "config.json", hidden_size=4096)
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "tokenizer.json").touch()

        config = Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

        assert config.variant is Qwen3VLVariantType.Qwen3VL_8B

    @pytest.mark.parametrize(("hidden_size", "num_hidden_layers"), [(5120, 36), (2560, 28)])
    def test_rejects_an_unsupported_directory_config(
        self, tmp_path: Path, hidden_size: int, num_hidden_layers: int
    ) -> None:
        self._write_config(tmp_path / "config.json", hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "tokenizer.json").touch()

        with pytest.raises(NotAMatchError, match="hidden size|layers"):
            Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

    def test_rejects_malformed_text_config(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3VLModel"], "text_config": []}))
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "tokenizer.json").touch()

        with pytest.raises(NotAMatchError, match="text_config"):
            Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

    @pytest.mark.parametrize("artifact", ["adapter_model.safetensors", "training_args.bin"])
    def test_rejects_unrecognized_weight_artifact(self, tmp_path: Path, artifact: str) -> None:
        self._write_config(tmp_path / "config.json")
        (tmp_path / artifact).touch()
        (tmp_path / "tokenizer.json").touch()

        with pytest.raises(NotAMatchError, match="weights"):
            Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

    def test_rejects_incomplete_sharded_checkpoint(self, tmp_path: Path) -> None:
        self._write_config(tmp_path / "config.json")
        (tmp_path / "model-00001-of-00002.safetensors").touch()
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "language_model.layers.0.weight": "model-00001-of-00002.safetensors",
                        "language_model.layers.35.weight": "model-00002-of-00002.safetensors",
                    }
                }
            )
        )
        (tmp_path / "tokenizer.json").touch()

        with pytest.raises(NotAMatchError, match="missing|weights"):
            Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

    @pytest.mark.parametrize("index_name", ["model.safetensors.index.json", "pytorch_model.bin.index.json"])
    def test_accepts_complete_sharded_checkpoint(self, tmp_path: Path, index_name: str) -> None:
        self._write_config(tmp_path / "config.json")
        suffix = "safetensors" if index_name.startswith("model") else "bin"
        shard_names = [f"model-00001-of-00002.{suffix}", f"model-00002-of-00002.{suffix}"]
        for shard_name in shard_names:
            (tmp_path / shard_name).touch()
        (tmp_path / index_name).write_text(
            json.dumps(
                {
                    "weight_map": {
                        "language_model.layers.0.weight": shard_names[0],
                        "language_model.layers.35.weight": shard_names[1],
                    }
                }
            )
        )
        (tmp_path / "tokenizer.json").touch()

        config = Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

        assert config.type is ModelType.Qwen3VLEncoder

    @pytest.mark.parametrize("bad_filename", [7, "../outside.safetensors", "/tmp/outside.safetensors"])
    def test_rejects_unsafe_or_non_string_shard_names(self, tmp_path: Path, bad_filename: object) -> None:
        self._write_config(tmp_path / "config.json")
        valid_shard = "model-00001-of-00002.safetensors"
        (tmp_path / valid_shard).touch()
        outside = tmp_path.parent / "outside.safetensors"
        outside.touch()
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "language_model.layers.0.weight": valid_shard,
                        "language_model.layers.35.weight": bad_filename,
                    }
                }
            )
        )
        (tmp_path / "tokenizer.json").touch()

        with pytest.raises(NotAMatchError, match="weights|shard|index"):
            Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))

    def test_rejects_existing_absolute_shard_path(self, tmp_path: Path) -> None:
        self._write_config(tmp_path / "config.json")
        outside = tmp_path.parent / "absolute-outside.safetensors"
        outside.touch()
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"language_model.layers.0.weight": outside.as_posix()}})
        )
        (tmp_path / "tokenizer.json").touch()

        with pytest.raises(NotAMatchError, match="weights|shard|index"):
            Qwen3VLEncoder_Qwen3VLEncoder_Config.from_model_on_disk(ModelOnDisk(tmp_path), self._fields(tmp_path))
