"""Regression tests for Qwen3 Encoder config probing.

See https://github.com/invoke-ai/InvokeAI/issues/9090

`Qwen2.5-1.5B-Instruct` (a standalone causal LM) was being misidentified as a
`Qwen3Encoder` because the diffusers-style config check matched any directory with
`config.json` at the root and a Qwen* class name. A complete causal LM also bundles
tokenizer files at the root, while standalone text_encoder downloads do not — we
use that to disambiguate.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
    Qwen3Encoder_GGUF_Config,
    Qwen3Encoder_Qwen3Encoder_Config,
    Qwen3Encoder_SDNQ_Folder_Config,
    _has_gemma2_keys,
    _has_qwen_vl_visual_tower,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.quantization.sdnq.loaders import _parse_quantization_config, sdnq_sd_loader

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test-model",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _write_config(path: Path, hidden_size: int = 2560, architecture: str = "Qwen2ForCausalLM") -> None:
    path.write_text(json.dumps({"architectures": [architecture], "hidden_size": hidden_size}))


@pytest.mark.parametrize("tokenizer_file", ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"])
def test_complete_causal_lm_is_rejected(tokenizer_file: str) -> None:
    """A directory with config.json + tokenizer files at root is a TextLLM, not a Qwen3 encoder."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_config(root / "config.json")
        (root / tokenizer_file).write_text("{}")

        mod = MagicMock()
        mod.path = root

        with pytest.raises(NotAMatchError, match="complete causal LM"):
            Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_standalone_text_encoder_subfolder_still_matches() -> None:
    """A standalone text_encoder download (config.json at root, no tokenizer files) should still match."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_config(root / "config.json")

        mod = MagicMock()
        mod.path = root

        config = Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "qwen3_encoder"


class TestQwen3EncoderRejectsVisualTower:
    """Regression for the single-file Qwen3-VL dual-match bug.

    A nested-layout single-file Qwen3-VL 4B encoder (``model.layers.*`` + ``model.visual.*``) previously
    matched BOTH the text-only Qwen3Encoder checkpoint config and the Qwen3VLEncoder config, because the
    reject helper only looked for bare ``visual.blocks.*``. The tie-break is nondeterministic, so the
    encoder could register as Qwen3Encoder and vanish from Krea-2's Qwen3-VL dropdown - hard-blocking the
    single-file/GGUF Krea-2 install path. The reject helper now matches the nested ``model.visual.*``
    layout too, keeping the two configs mutually exclusive.
    """

    @pytest.mark.parametrize(
        "visual_key",
        [
            "model.visual.blocks.0.attn.qkv.weight",  # ComfyUI single-file nested layout
            "visual.blocks.0.attn.qkv.weight",  # already-split (transformers) layout
            "model.visual.patch_embed.proj.weight",
        ],
    )
    def test_matches_nested_and_split_visual_layouts(self, visual_key: str) -> None:
        assert _has_qwen_vl_visual_tower({visual_key: object()}) is True

    def test_ignores_text_only_decoder(self) -> None:
        assert _has_qwen_vl_visual_tower({"model.layers.0.self_attn.q_proj.weight": object()}) is False

    @patch("invokeai.backend.model_manager.configs.qwen3_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_encoder.raise_for_override_fields")
    def test_checkpoint_config_rejects_nested_qwen3_vl(self, _rfo, _rif) -> None:
        mod = MagicMock()
        mod.path = Path("/fake/qwen3vl_4b.safetensors")
        mod.load_state_dict.return_value = {
            "model.layers.0.self_attn.q_proj.weight": object(),
            "model.visual.blocks.0.attn.qkv.weight": object(),
        }
        with pytest.raises(NotAMatchError, match="visual tower"):
            Qwen3Encoder_Checkpoint_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_nested_text_encoder_with_root_tokenizer_still_matches() -> None:
    """A model with text_encoder/config.json should match even if tokenizer files exist at root.

    The tokenizer-at-root heuristic only applies to the standalone (root-level config.json) case.
    """
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "text_encoder").mkdir()
        _write_config(root / "text_encoder" / "config.json")
        (root / "tokenizer.json").write_text("{}")

        mod = MagicMock()
        mod.path = root

        config = Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "qwen3_encoder"


def test_has_gemma2_keys_detects_post_norms() -> None:
    """Gemma-2/3 GGUFs are identified by their post-attention / post-feedforward norms."""
    assert _has_gemma2_keys({"token_embd.weight": 0, "blk.0.post_attention_norm.weight": 0}) is True
    assert _has_gemma2_keys({"blk.5.post_ffw_norm.weight": 0}) is True
    # A Qwen3-style state dict (attn q/k norms, no post-norms) is not mistaken for Gemma.
    assert _has_gemma2_keys({"token_embd.weight": 0, "blk.0.attn_q_norm.weight": 0}) is False


def test_gguf_config_rejects_gemma_state_dict() -> None:
    """A Gemma-2 GGUF satisfies the generic Qwen3 key heuristic but must be rejected by the Qwen3 GGUF
    config so it is not re-identified as a Qwen3 encoder (it should classify as Gemma2Encoder)."""
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "token_embd.weight": 0,
        "blk.0.attn_norm.weight": 0,
        "blk.0.post_attention_norm.weight": 0,
        "blk.0.post_ffw_norm.weight": 0,
    }
    with pytest.raises(NotAMatchError, match="Gemma-2"):
        Qwen3Encoder_GGUF_Config._validate_looks_like_qwen3_model(mod)


class TestShardedQwen3EncoderFolder:
    """A sharded text_encoder folder must still identify as a Qwen3 encoder.

    The starter-model download
    `black-forest-labs/FLUX.2-klein-{4B,9B}::text_encoder+tokenizer` lands the encoder as several
    `model-0000N-of-0000M.safetensors` shards. The SDNQ rejection guard used to call
    `mod.load_state_dict()`, which raises ValueError ("Multiple weight files found") rather than a
    NotAMatchError when a folder holds more than one weight file. That aborted this config's probe,
    so the model was stored as `unknown`.
    """

    @staticmethod
    def _make_sharded_encoder(root: Path, *, sdnq: bool = False) -> Path:
        text_encoder = root / "text_encoder"
        text_encoder.mkdir(parents=True)
        _write_config(text_encoder / "config.json", hidden_size=4096, architecture="Qwen3ForCausalLM")

        shards: list[dict[str, torch.Tensor]] = [
            {"model.embed_tokens.weight": torch.zeros(8, 4096, dtype=torch.uint8 if sdnq else torch.float32)},
            {"model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.uint8 if sdnq else torch.float32)},
        ]
        if sdnq:
            shards[1]["model.layers.0.self_attn.q_proj.scale"] = torch.zeros(8, 1, dtype=torch.float32)

        for i, shard in enumerate(shards, start=1):
            save_file(shard, str(text_encoder / f"model-0000{i}-of-00002.safetensors"))

        # Tokenizer files live in their own subfolder for the `text_encoder+tokenizer` download layout.
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer_config.json").write_text("{}")
        return root

    def test_sharded_encoder_matches(self, tmp_path: Path) -> None:
        root = self._make_sharded_encoder(tmp_path / "klein-9b-encoder")
        config = Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

        assert config.type.value == "qwen3_encoder"
        assert config.variant.value == "qwen3_8b"

    def test_sharded_sdnq_encoder_is_still_rejected(self, tmp_path: Path) -> None:
        """The shard-safe check must keep detecting SDNQ weight+scale pairs across shards."""
        root = self._make_sharded_encoder(tmp_path / "klein-9b-encoder-sdnq", sdnq=True)
        with pytest.raises(NotAMatchError, match="SDNQ"):
            Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

    def test_folder_probe_never_loads_a_state_dict(self, tmp_path: Path) -> None:
        """Identifying a folder encoder must not go through `load_state_dict()` at all.

        A folder holding more than one weight file has no single state dict to load: the call raises
        ValueError, which is not a NotAMatchError, and the probe is abandoned. Everything this config
        needs is in `config.json` and the safetensors headers, so make the absence of that call a
        property of the test suite rather than of today's implementation.
        """
        root = self._make_sharded_encoder(tmp_path / "klein-9b-encoder-nosd")

        def _explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("identification must not load the state dict of a folder model")

        with patch.object(ModelOnDisk, "load_state_dict", _explode):
            config = Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

        assert config.variant.value == "qwen3_8b"


class TestSdnqQwen3EncoderFolder:
    """`Qwen3Encoder_Qwen3Encoder_Config` and `Qwen3Encoder_SDNQ_Folder_Config` must partition folders.

    The two used to ask different questions about the same directory: the unquantized config looked
    for the SDNQ marker in `text_encoder/` as well and for SDNQ keys across every shard, while the
    SDNQ config looked only at the root and only at safetensors sitting directly in it. An SDNQ
    encoder in the nested `text_encoder/` layout was therefore rejected by *both* - the exact shape
    that lands a model in `unknown`.
    """

    @staticmethod
    def _make_sdnq_encoder(root: Path, *, marker_in: str | None = None, architecture: str | None = None) -> Path:
        text_encoder = root / "text_encoder"
        text_encoder.mkdir(parents=True)

        config: dict[str, object] = {"hidden_size": 4096}
        if architecture is not None:
            config["architectures"] = [architecture]
        (text_encoder / "config.json").write_text(json.dumps(config))

        # q_norm/k_norm are the Qwen3-only marker the config falls back to when config.json declares
        # no architecture; they are split across shards on purpose.
        save_file(
            {"model.embed_tokens.weight": torch.zeros(8, 4096, dtype=torch.uint8)},
            str(text_encoder / "model-00001-of-00002.safetensors"),
        )
        save_file(
            {
                "model.layers.0.self_attn.q_norm.weight": torch.zeros(8, 8, dtype=torch.uint8),
                "model.layers.0.self_attn.q_norm.scale": torch.zeros(8, 1, dtype=torch.float32),
            },
            str(text_encoder / "model-00002-of-00002.safetensors"),
        )

        if marker_in is not None:
            (root / marker_in / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}))
        return root

    def test_nested_markerless_sdnq_encoder_is_claimed(self, tmp_path: Path) -> None:
        """Detected by key shape across the shards of `text_encoder/`, with no marker file at all."""
        root = self._make_sdnq_encoder(tmp_path / "sdnq-nested", architecture="Qwen3ForCausalLM")

        config = Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

        assert config.format.value == "sdnq_quantized"
        assert config.variant.value == "qwen3_8b"
        # ...and the unquantized config declines the same folder, so exactly one of them matches.
        with pytest.raises(NotAMatchError, match="SDNQ"):
            Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

    def test_marker_in_text_encoder_subfolder_is_honored(self, tmp_path: Path) -> None:
        root = self._make_sdnq_encoder(
            tmp_path / "sdnq-marker-nested", marker_in="text_encoder", architecture="Qwen3ForCausalLM"
        )

        config = Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))
        assert config.format.value == "sdnq_quantized"

    def test_unreadable_marker_falls_through_to_the_key_check(self, tmp_path: Path) -> None:
        """A corrupt `quantization_config.json` used to abort this probe with a JSONDecodeError."""
        root = self._make_sdnq_encoder(tmp_path / "sdnq-bad-marker", architecture="Qwen3ForCausalLM")
        (root / "quantization_config.json").write_text("{not json")

        config = Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))
        assert config.format.value == "sdnq_quantized"

    def test_sharded_folder_without_declared_architecture_is_claimed(self, tmp_path: Path) -> None:
        """The Qwen3-only q_norm/k_norm fallback must read shard headers, not a single state dict.

        With no `architectures` in config.json the config falls back to tensor names. Reading them
        via `load_state_dict()` yielded nothing for a sharded folder, so this shape was rejected.
        """
        root = self._make_sdnq_encoder(tmp_path / "sdnq-no-arch")

        config = Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))
        assert config.format.value == "sdnq_quantized"


class TestSdnqEncoderIdentificationMatchesTheLoader:
    """Identification must not accept a folder `Qwen3EncoderSDNQLoader` cannot open.

    Claiming the nested `text_encoder/` layout without teaching the loader about it would turn
    "unknown at install" into "installed, then ValueError at generation time" - the worse of the two,
    and the very failure `Main_SDNQ_Diffusers_ZImage_Config._validate_has_sdnq_transformer` exists to
    prevent. `sdnq_sd_loader` globs one directory and reads the `quantization_config.json` beside it,
    so the loader has to resolve the layout the same way the config did.
    """

    @staticmethod
    def _make_nested_bundle(root: Path, *, group_size: int = 64) -> Path:
        text_encoder = root / "text_encoder"
        text_encoder.mkdir(parents=True)
        (text_encoder / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3ForCausalLM"], "hidden_size": 2560})
        )
        (text_encoder / "quantization_config.json").write_text(
            json.dumps({"quant_method": "sdnq", "group_size": group_size})
        )
        save_file(
            {
                "model.embed_tokens.weight": torch.randint(-128, 127, (64, 64), dtype=torch.int8),
                "model.embed_tokens.scale": torch.tensor([0.01], dtype=torch.float32),
            },
            str(text_encoder / "model-00001-of-00002.safetensors"),
        )
        save_file(
            {
                "model.layers.0.self_attn.q_norm.weight": torch.randint(-128, 127, (64, 64), dtype=torch.int8),
                "model.layers.0.self_attn.q_norm.scale": torch.tensor([0.02], dtype=torch.float32),
            },
            str(text_encoder / "model-00002-of-00002.safetensors"),
        )
        (root / "tokenizer").mkdir()
        (root / "tokenizer" / "tokenizer_config.json").write_text("{}")
        return root

    def test_the_loader_can_open_what_identification_accepted(self, tmp_path: Path) -> None:
        root = self._make_nested_bundle(tmp_path / "sdnq-nested-loadable")

        config = Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))
        assert config.format.value == "sdnq_quantized"

        # The bug: handing the loader the registered path finds no weights at all.
        with pytest.raises(ValueError, match="No safetensors files found"):
            sdnq_sd_loader(root, compute_dtype=torch.float32)

        # The fix: the same resolution identification used points at the shards *and* at the marker
        # next to them, so a real group_size is honoured instead of silently defaulting.
        resolved = Qwen3Encoder_SDNQ_Folder_Config.resolve_text_encoder_dir(root)
        assert resolved == root / "text_encoder"
        assert _parse_quantization_config(resolved / "quantization_config.json").get("group_size") == 64

        sd = sdnq_sd_loader(resolved, compute_dtype=torch.float32)
        assert "model.embed_tokens.weight" in sd
        assert "model.layers.0.self_attn.q_norm.weight" in sd

    def test_standalone_layout_still_resolves_to_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "sdnq-standalone"
        root.mkdir()
        assert Qwen3Encoder_SDNQ_Folder_Config.resolve_text_encoder_dir(root) == root


class TestSdnqPipelineBundleIsNotAQwen3Encoder:
    """A full SDNQ pipeline bundle must not also match `Qwen3Encoder_SDNQ_Folder_Config`.

    Its transformer and VAE are SDNQ-quantized too, so a recursive key scan answers "yes" at the
    bundle root. While `Main_SDNQ_Diffusers_*` matches, the factory's type sort hides that; when it
    declines - an interrupted download, a corrupt `model_index.json` - the whole bundle registers as
    a Qwen3 encoder at a path the SDNQ encoder loader cannot open.
    """

    @staticmethod
    def _make_bundle(root: Path, *, model_index: str | None = None, transformer_shards: bool = True) -> Path:
        transformer = root / "transformer"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text(json.dumps({"_class_name": "ZImageTransformer2DModel"}))
        (transformer / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}))
        if transformer_shards:
            save_file(
                {
                    "transformer_blocks.0.attn.to_q.weight": torch.randint(-128, 127, (64, 64), dtype=torch.int8),
                    "transformer_blocks.0.attn.to_q.scale": torch.tensor([0.01], dtype=torch.float32),
                },
                str(transformer / "diffusion_pytorch_model.safetensors"),
            )
        if model_index is not None:
            (root / "model_index.json").write_text(model_index)
        return root

    def test_complete_bundle_is_rejected(self, tmp_path: Path) -> None:
        root = self._make_bundle(tmp_path / "bundle", model_index=json.dumps({"_class_name": "ZImagePipeline"}))
        with pytest.raises(NotAMatchError, match="full diffusers pipeline"):
            Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

    def test_interrupted_download_is_rejected(self, tmp_path: Path) -> None:
        """Transformer marker and config present, shards not downloaded yet: still not an encoder."""
        root = self._make_bundle(tmp_path / "bundle-partial", transformer_shards=False)
        with pytest.raises(NotAMatchError, match="full diffusers pipeline"):
            Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

    def test_corrupt_model_index_is_rejected(self, tmp_path: Path) -> None:
        root = self._make_bundle(tmp_path / "bundle-corrupt", model_index="{not json")
        with pytest.raises(NotAMatchError, match="full diffusers pipeline"):
            Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

    def test_sdnq_keys_outside_the_encoder_dirs_do_not_count(self, tmp_path: Path) -> None:
        """The scan is scoped to the directories the loaders read, not an rglob over the tree.

        No `transformer/` and no `model_index.json` here, so the pipeline guard does not fire - only
        the scope keeps a folder whose sole SDNQ weights live in `vae/` from reading as an encoder.
        """
        root = tmp_path / "vae-only"
        vae = root / "vae"
        vae.mkdir(parents=True)
        save_file(
            {
                "decoder.conv_in.weight": torch.randint(-128, 127, (64, 64), dtype=torch.int8),
                "decoder.conv_in.scale": torch.tensor([0.01], dtype=torch.float32),
            },
            str(vae / "diffusion_pytorch_model.safetensors"),
        )
        with pytest.raises(NotAMatchError, match="does not look like an SDNQ-quantized Qwen3 encoder"):
            Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))
