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

from invokeai.backend.model_manager.configs.base import ModelRole
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_Checkpoint_Config,
    Qwen3VLEncoder_Qwen3VLEncoder_Config,
    _is_qwen3_vl_encoder_state_dict,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, Qwen3VLVariantType
from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_config_dict
from tests.fixtures.quantized_payloads import (
    comfy_quant_marker,
    mxfp8_tensors,
    nvfp4_signed_tensors,
    quantize_scaled_fp8,
)

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

    def test_a_multimodal_qwen3_5_export_does_not_match(self) -> None:
        # MOSS-Image's encoder (transformers `Qwen3_5Model`): a decoder and a visual tower, like Qwen3-VL,
        # but with Gated DeltaNet layers, which no Qwen3-VL has -- whatever its depth.
        sd = {
            "language_model.layers.0.linear_attn.in_proj_qkv.weight": object(),
            "language_model.layers.35.self_attn.q_proj.weight": object(),
            "visual.blocks.0.attn.qkv.weight": object(),
        }
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


def _replace_packed_layer(state_dict: dict, **tensors: torch.Tensor) -> None:
    for key in [key for key in state_dict if key.startswith(f"{_PACKED}.")]:
        del state_dict[key]
    state_dict.update({f"{_PACKED}.{suffix}": tensor for suffix, tensor in tensors.items()})


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

    def test_scaled_fp8_and_int8_builds_still_install(self, tmp_path: Path) -> None:
        # Both pair a weight with a `weight_scale` too, and neither is 4-bit.
        fp8 = quantize_scaled_fp8(torch.randn(128, 64))
        assert (
            self._identify(
                tmp_path, lambda sd: _replace_packed_layer(sd, weight=fp8.codes, weight_scale=fp8.scale)
            ).variant
            is Qwen3VLVariantType.Qwen3VL_4B
        )
        int8 = {
            "weight": torch.randint(-127, 128, (128, 64), dtype=torch.int8),
            "weight_scale": torch.ones(128, 1),
            "comfy_quant": comfy_quant_marker({"format": "int8_tensorwise"}),
        }
        assert (
            self._identify(tmp_path, lambda sd: _replace_packed_layer(sd, **int8)).variant
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
            # A marker that names no format names nothing; only the header could, and here it does not.
            *(
                pytest.param(
                    lambda sd, marker=marker: sd.update({f"{_PACKED}.comfy_quant": comfy_quant_marker(marker)}),
                    "no ComfyUI `comfy_quant` marker",
                    id=f"unnamed-{name}-marker",
                )
                for name, marker in (("empty", {}), ("format-less", {"full_precision_matrix_mult": False}))
            ),
            # Marked nvfp4 over a dense weight: no `weight_scale_2` anywhere in the file, and no packed half-state for
            # the structural check to catch, so only the marker's own bytes say what the loader will refuse.
            pytest.param(
                lambda sd: _replace_packed_layer(
                    sd,
                    weight=torch.zeros(128, 128, dtype=torch.bfloat16),
                    comfy_quant=comfy_quant_marker({"format": "nvfp4"}),
                ),
                "has no weight_scale_2, so it cannot be decoded",
                id="nvfp4-marker-on-a-dense-weight",
            ),
        ],
    )
    def test_a_layer_the_loader_would_refuse_is_refused_at_install(self, tmp_path: Path, mutate, message: str) -> None:
        with pytest.raises(InvalidMatchError, match=message):
            self._identify(tmp_path, mutate)
        with pytest.raises(ValueError, match=message):
            self._loader_finds_nvfp4_layers(tmp_path / "qwen3vl_4b.safetensors")

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

    @staticmethod
    def _loader_finds_nvfp4_layers(path: Path) -> list[str]:
        """The loader's own check, over the real tensors and header of the file `_identify` wrote."""
        from safetensors.torch import load_file

        from invokeai.backend.model_manager.util.qwen3_vl import drop_qwen3vl_unloaded_keys
        from invokeai.backend.quantization.fp8_scaled import parse_quantization_metadata, read_safetensors_metadata
        from invokeai.backend.quantization.nvfp4 import find_nvfp4_layers

        state_dict = drop_qwen3vl_unloaded_keys(load_file(str(path)))
        return find_nvfp4_layers(state_dict, parse_quantization_metadata(read_safetensors_metadata(path)))

    @pytest.mark.parametrize("marker", [{}, {"full_precision_matrix_mult": False}], ids=["empty", "format_less"])
    def test_a_header_named_layer_installs_beside_a_marker_naming_no_format(self, tmp_path: Path, marker) -> None:
        # The loader names a layer nvfp4 when its marker or its header entry says so. Laid over the header, a marker
        # without a format erased the header's entry, and a file the loader takes was refused at install.
        def add_marker(state_dict: dict) -> None:
            state_dict[f"{_PACKED}.comfy_quant"] = comfy_quant_marker(marker)

        assert self._identify(tmp_path, add_marker, naming="header").variant is Qwen3VLVariantType.Qwen3VL_4B
        assert self._loader_finds_nvfp4_layers(tmp_path / "qwen3vl_4b.safetensors") == [_PACKED]

    def test_a_marker_contradicting_a_header_named_layer_is_refused_at_install_as_at_load(self, tmp_path: Path) -> None:
        def add_marker(state_dict: dict) -> None:
            state_dict[f"{_PACKED}.comfy_quant"] = comfy_quant_marker({"format": "float8_e4m3fn"})

        with pytest.raises(InvalidMatchError, match="contradicts its tensors"):
            self._identify(tmp_path, add_marker, naming="header")
        with pytest.raises(ValueError, match="contradicts its tensors"):
            self._loader_finds_nvfp4_layers(tmp_path / "qwen3vl_4b.safetensors")


_HIDDEN_BY_VARIANT = {Qwen3VLVariantType.Qwen3VL_4B: 2560, Qwen3VLVariantType.Qwen3VL_8B: 4096}
_MX_HEADER = {"_quantization_metadata": json.dumps({"layers": {"lm_head": {"format": "mxfp8"}}})}


def _mx_head(rows: int = 128) -> dict[str, torch.Tensor]:
    return mxfp8_tensors("lm_head", torch.full((rows, 4096 // 32), 127))[0]


class TestQwen3VLEncoderServesAsATextLLM:
    """A single file that is the whole Instruct chat model also serves as a Text LLM; one that cannot generate does not.

    Real files through `ModelOnDisk`, so the decision is made on the meta tensors identification holds, by the rule
    the Text LLM loader refuses by.
    """

    @staticmethod
    def _roles(
        tmp_path: Path,
        variant: Qwen3VLVariantType,
        head: dict[str, torch.Tensor] | None = None,
        *,
        language_model: str = "model.",
        head_prefix: str = "lm_head.",
        vocab: int = 2,
        metadata: dict[str, str] | None = None,
    ) -> list[ModelRole]:
        hidden = _HIDDEN_BY_VARIANT[variant]
        state_dict = {
            f"{language_model}embed_tokens.weight": torch.zeros(vocab, hidden, dtype=torch.bfloat16),
            f"{language_model}layers.35.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
            f"{language_model}norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
            # One tower weight: enough to identify the encoder, too few for the tower to load.
            "model.visual.blocks.0.attn.qkv.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
            **{f"{head_prefix}{key.removeprefix('lm_head.')}": tensor for key, tensor in (head or {}).items()},
        }
        path = tmp_path / "qwen3vl.safetensors"
        save_file(state_dict, str(path), metadata=metadata)
        config = Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(ModelOnDisk(path), {**_REQUIRED_FIELDS})
        assert config.variant is variant
        return config.roles

    @staticmethod
    def _role(variant: Qwen3VLVariantType, *, reads_images: bool = False) -> list[ModelRole]:
        return [
            ModelRole(
                type=ModelType.TextLLM, format=ModelFormat.Qwen3VLEncoder, variant=variant, supports_images=reads_images
            )
        ]

    def test_a_4b_file_generates_through_its_tied_embedding_without_an_lm_head(self, tmp_path: Path) -> None:
        assert self._roles(tmp_path, Qwen3VLVariantType.Qwen3VL_4B) == self._role(Qwen3VLVariantType.Qwen3VL_4B)

    @pytest.mark.parametrize(
        "head",
        [
            pytest.param({"lm_head.weight": torch.zeros(2, 4096, dtype=torch.bfloat16)}, id="bf16"),
            # Comfy-Org's nvfp4 8B keeps its head in scaled fp8, named by a marker.
            pytest.param(
                {
                    "lm_head.weight": quantize_scaled_fp8(torch.randn(2, 4096)).codes,
                    "lm_head.weight_scale": torch.tensor(0.01),
                    "lm_head.comfy_quant": comfy_quant_marker({"format": "float8_e4m3fn"}),
                },
                id="scaled-fp8",
            ),
        ],
    )
    def test_an_8b_file_with_an_lm_head_the_loader_widens_serves_as_a_text_llm(self, tmp_path: Path, head) -> None:
        assert self._roles(tmp_path, Qwen3VLVariantType.Qwen3VL_8B, head) == self._role(Qwen3VLVariantType.Qwen3VL_8B)

    @pytest.mark.parametrize(
        ("language_model", "head_prefix"),
        [("model.", "model.lm_head."), ("model.language_model.", "lm_head.")],
        ids=["model-prefixed-head", "transformers-layout"],
    )
    def test_the_lm_head_is_found_in_either_spelling(
        self, tmp_path: Path, language_model: str, head_prefix: str
    ) -> None:
        head = {"lm_head.weight": torch.zeros(2, 4096, dtype=torch.bfloat16)}

        roles = self._roles(
            tmp_path, Qwen3VLVariantType.Qwen3VL_8B, head, language_model=language_model, head_prefix=head_prefix
        )

        assert roles == self._role(Qwen3VLVariantType.Qwen3VL_8B)

    def test_an_mxfp8_head_serves_when_the_header_names_it(self, tmp_path: Path) -> None:
        roles = self._roles(tmp_path, Qwen3VLVariantType.Qwen3VL_8B, _mx_head(), vocab=128, metadata=_MX_HEADER)

        assert roles == self._role(Qwen3VLVariantType.Qwen3VL_8B)

    @pytest.mark.parametrize(
        ("head", "vocab"),
        [
            pytest.param({}, 2, id="no-lm-head"),
            pytest.param({"lm_head.weight": torch.zeros(2, 2048, dtype=torch.uint8)}, 2, id="packed-lm-head"),
            pytest.param({"lm_head.weight": torch.zeros(3, 4096, dtype=torch.bfloat16)}, 2, id="head-wider-than-vocab"),
            pytest.param(
                {
                    "lm_head.weight": torch.zeros(2, 4096, dtype=torch.bfloat16),
                    "lm_head.bias": torch.zeros(2, dtype=torch.bfloat16),
                },
                2,
                id="unread-head-key",
            ),
            pytest.param(_mx_head(), 128, id="mx-grid-nothing-names"),
        ],
    )
    def test_an_8b_file_that_cannot_generate_is_only_an_encoder(self, tmp_path: Path, head, vocab: int) -> None:
        assert self._roles(tmp_path, Qwen3VLVariantType.Qwen3VL_8B, head, vocab=vocab) == []


def _release_tower_shapes(variant: Qwen3VLVariantType) -> dict[str, list[int]]:
    """The vision tower's weights and shapes, worked out from the release's vision config rather than from a model."""
    config = load_bundled_qwen3_vl_config_dict(variant)["vision_config"]
    hidden, inner, out = config["hidden_size"], config["intermediate_size"], config["out_hidden_size"]
    merged = hidden * config["spatial_merge_size"] ** 2
    patch = config["patch_size"]
    shapes = {
        "patch_embed.proj.weight": [hidden, config["in_channels"], config["temporal_patch_size"], patch, patch],
        "patch_embed.proj.bias": [hidden],
        "pos_embed.weight": [config["num_position_embeddings"], hidden],
    }
    for block in range(config["depth"]):
        for name, shape in {
            "norm1.weight": [hidden],
            "norm1.bias": [hidden],
            "norm2.weight": [hidden],
            "norm2.bias": [hidden],
            "attn.qkv.weight": [3 * hidden, hidden],
            "attn.qkv.bias": [3 * hidden],
            "attn.proj.weight": [hidden, hidden],
            "attn.proj.bias": [hidden],
            "mlp.linear_fc1.weight": [inner, hidden],
            "mlp.linear_fc1.bias": [inner],
            "mlp.linear_fc2.weight": [hidden, inner],
            "mlp.linear_fc2.bias": [hidden],
        }.items():
            shapes[f"blocks.{block}.{name}"] = shape
    # The final merger normalizes each patch before merging 2 x 2 of them; the deepstack mergers normalize after.
    mergers = [("merger", hidden)] + [
        (f"deepstack_merger_list.{index}", merged) for index in range(len(config["deepstack_visual_indexes"]))
    ]
    for merger, norm in mergers:
        shapes.update(
            {
                f"{merger}.norm.weight": [norm],
                f"{merger}.norm.bias": [norm],
                f"{merger}.linear_fc1.weight": [merged, merged],
                f"{merger}.linear_fc1.bias": [merged],
                f"{merger}.linear_fc2.weight": [out, merged],
                f"{merger}.linear_fc2.bias": [out],
            }
        )
    return shapes


_QKV = "blocks.0.attn.qkv"


class TestQwen3VLEncoderTextLLMReadsImages:
    """The Text LLM role reads images when the file's vision tower would load, by the tower loader's rules.

    Header-only files at the release's shapes: identification reads headers and markers, never tensor data.
    """

    @staticmethod
    def _reads_images(
        tmp_path: Path,
        variant: Qwen3VLVariantType,
        tower: dict[str, tuple[list[int], str]],
        markers: dict[str, dict] | None = None,
    ) -> bool | None:
        from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors

        hidden = _HIDDEN_BY_VARIANT[variant]
        keys = {
            "model.embed_tokens.weight": ([2, hidden], "BF16"),
            "model.layers.35.input_layernorm.weight": ([4], "BF16"),
            "model.norm.weight": ([hidden], "BF16"),
            "lm_head.weight": ([2, hidden], "BF16"),
            **tower,
        }
        path = write_header_only_safetensors(tmp_path / "qwen3vl.safetensors", keys, markers)
        config = Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(ModelOnDisk(path), {**_REQUIRED_FIELDS})
        (role,) = config.roles
        assert (role.type, role.variant) == (ModelType.TextLLM, variant)
        return role.supports_images

    @staticmethod
    def _tower(variant: Qwen3VLVariantType, prefix: str = "model.visual.") -> dict[str, tuple[list[int], str]]:
        return {f"{prefix}{name}": (shape, "BF16") for name, shape in _release_tower_shapes(variant).items()}

    @pytest.mark.parametrize("variant", [Qwen3VLVariantType.Qwen3VL_4B, Qwen3VLVariantType.Qwen3VL_8B])
    @pytest.mark.parametrize("prefix", ["model.visual.", "visual."], ids=["comfy", "bare"])
    def test_a_whole_tower_in_either_spelling_reads_images(
        self, tmp_path: Path, variant: Qwen3VLVariantType, prefix: str
    ) -> None:
        assert self._reads_images(tmp_path, variant, self._tower(variant, prefix)) is True

    def test_a_scaled_fp8_tower_layer_reads_images(self, tmp_path: Path) -> None:
        tower = self._tower(Qwen3VLVariantType.Qwen3VL_4B)
        shape, _ = tower[f"model.visual.{_QKV}.weight"]
        tower[f"model.visual.{_QKV}.weight"] = (shape, "F8_E4M3")
        tower[f"model.visual.{_QKV}.weight_scale"] = ([], "F32")

        assert self._reads_images(tmp_path, Qwen3VLVariantType.Qwen3VL_4B, tower) is True

    def test_an_int8_tower_layer_reads_images_when_its_marker_claims_it(self, tmp_path: Path) -> None:
        tower = self._tower(Qwen3VLVariantType.Qwen3VL_4B)
        shape, _ = tower[f"model.visual.{_QKV}.weight"]
        tower[f"model.visual.{_QKV}.weight"] = (shape, "I8")
        tower[f"model.visual.{_QKV}.weight_scale"] = ([shape[0], 1], "F32")
        marker = {f"model.visual.{_QKV}": {"format": "int8_tensorwise"}}

        assert self._reads_images(tmp_path, Qwen3VLVariantType.Qwen3VL_4B, tower, marker) is True
        assert self._reads_images(tmp_path, Qwen3VLVariantType.Qwen3VL_4B, tower) is False

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda tower: tower.pop("model.visual.merger.linear_fc2.weight"), id="a-weight-missing"),
            pytest.param(
                lambda tower: tower.update({f"model.visual.{_QKV}.weight": ([3072, 1000], "BF16")}),
                id="a-weight-of-another-shape",
            ),
            pytest.param(
                lambda tower: tower.update({"visual.merger.norm.weight": ([1024], "BF16")}), id="a-key-spelled-twice"
            ),
        ],
    )
    def test_a_tower_its_loader_refuses_leaves_the_text_llm_text_only(self, tmp_path: Path, mutate) -> None:
        tower = self._tower(Qwen3VLVariantType.Qwen3VL_4B)
        mutate(tower)

        assert self._reads_images(tmp_path, Qwen3VLVariantType.Qwen3VL_4B, tower) is False

    def test_a_tower_with_an_nvfp4_layer_its_loader_refuses_leaves_the_text_llm_text_only(self, tmp_path: Path) -> None:
        """ModelOpt-style: nvfp4 tensors, but nothing naming the layout ComfyUI's. The encoder still installs."""
        tower = self._tower(Qwen3VLVariantType.Qwen3VL_4B)
        layer = "model.visual.blocks.0.mlp.linear_fc1"
        packed, _ = nvfp4_signed_tensors(layer, torch.ones(128, 64, dtype=torch.bool))
        dtypes = {torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}
        tower.update({key: (list(tensor.shape), dtypes[tensor.dtype]) for key, tensor in packed.items()})

        assert self._reads_images(tmp_path, Qwen3VLVariantType.Qwen3VL_4B, tower) is False


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
