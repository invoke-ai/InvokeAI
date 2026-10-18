"""Loading MOSS-Image: the transformer from a single file or a pipeline folder, the pipeline's other submodels, and
FP8 Storage on the vendored transformer.

The geometry tests run against the real key layout (`moss_image_dit_keys`) on the meta device; everything that moves
values uses a tiny transformer at float32 on the CPU.
"""

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import accelerate
import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_MossImage_Config,
    Main_Diffusers_MossImage_Config,
)
from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.load.model_loaders import moss_image as loader_module
from invokeai.backend.model_manager.load.model_loaders.moss_image import (
    MOSS_IMAGE_FP8_SKIP_PATTERNS,
    MossImageCheckpointModel,
    MossImageDiffusersModel,
    moss_image_component_shards,
    moss_image_transformer_config,
)
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder
from tests.backend.model_manager.load.state_dicts.moss_image_dit_keys import state_dict_keys as moss_dit_keys
from tests.backend.model_manager.load.state_dicts.moss_image_vae_keys import state_dict_keys as moss_vae_keys

# `OpenMOSS-Team/MOSS-Image-Base`'s `transformer/config.json` at 1ceede80badddd6a7812069619edaa5633587192, verbatim
# but for `rope_base`, which no tensor records. A literal, so this fails when the detection drifts rather than agreeing
# with whatever the loader computes.
RELEASED_GEOMETRY = {
    "dim": 3072,
    "ffn_dim": 8192,
    "freq_dim": 256,
    "latent_channels": 128,
    "num_heads": 24,
    "num_layers": 42,
    "text_dim": 4096,
}

TINY_GEOMETRY = {
    "dim": 64,
    "ffn_dim": 96,
    "num_heads": 2,
    "num_layers": 3,
    "text_dim": 24,
    "freq_dim": 16,
    "latent_channels": 8,
}

_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}


def _meta_state_dict(keys: dict[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.empty(shape, dtype=_SAFETENSORS_DTYPES[dtype], device="meta") for key, (shape, dtype) in keys.items()
    }


def _tiny_reference() -> MOSSImageTransformer:
    torch.manual_seed(0)
    reference = MOSSImageTransformer(**TINY_GEOMETRY).eval()
    # Non-trivial norms, so a weight that does not arrive shows.
    for name, parameter in reference.named_parameters():
        if "norm" in name:
            torch.nn.init.normal_(parameter, mean=1.0, std=0.1)
    return reference


def _tiny_state_dict(reference: MOSSImageTransformer) -> dict[str, torch.Tensor]:
    return {key: value.clone().contiguous() for key, value in reference.state_dict().items()}


def _forward(model: MOSSImageTransformer) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, TINY_GEOMETRY["latent_channels"], 4, 4, generator=generator)
    text = torch.randn(1, 5, TINY_GEOMETRY["text_dim"], generator=generator)
    with torch.no_grad():
        return model(latents, torch.tensor(500.0), text).sample


def _loader(cls: type, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setattr(loader_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        loader_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.float32)
    )
    loader = object.__new__(cls)
    loader._ram_cache = MagicMock()
    loader._logger = logging.getLogger(__name__)
    loader._torch_device = torch.device("cpu")
    loader._torch_dtype = torch.float32
    return loader


def _assert_loaded(model: torch.nn.Module, reference: MOSSImageTransformer) -> None:
    assert [name for name, param in model.named_parameters() if param.is_meta] == []
    expected = reference.state_dict()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected[name]), name
    # Derived on load, at float32, whatever the checkpoint stored.
    assert model.rope.bands.dtype is torch.float32


def _write_index(path: Path, shards: dict[str, list[str]]) -> None:
    weight_map = {key: name for name, keys in shards.items() for key in keys}
    path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


class TestRealLayout:
    def test_the_geometry_read_from_the_checkpoint_is_the_released_one(self) -> None:
        geometry = moss_image_transformer_config(_meta_state_dict(moss_dit_keys))

        assert geometry == RELEASED_GEOMETRY

    def test_a_model_built_from_it_takes_every_key_and_leaves_none_missing(self) -> None:
        sd = _meta_state_dict(moss_dit_keys)
        with accelerate.init_empty_weights():
            model = MOSSImageTransformer(**moss_image_transformer_config(sd))

        result = model.load_state_dict(sd, strict=False, assign=True)

        assert (result.missing_keys, result.unexpected_keys) == ([], [])

    def test_the_fp8_skip_patterns_name_real_modules(self) -> None:
        """Every pattern matches a Linear of the released model, and only the ones the forward reads the dtype of."""
        with accelerate.init_empty_weights():
            model = MOSSImageTransformer(**RELEASED_GEOMETRY)
        skipped = {
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and any(re.search(p, name) for p in MOSS_IMAGE_FP8_SKIP_PATTERNS)
        }

        expected = {
            "video_embedder",
            "time_embedding.0",
            "shared_block_modulation",
            "video_head_modulation",
            "final_linear_video",
            "text_embedder.proj",
        } | {f"layers.{index}.proj.q" for index in range(42)}
        assert skipped == expected

    def test_the_vae_loads_whole_through_the_flux2_vae_checkpoint_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MOSS-Image's VAE (and its ComfyUI export) identifies as a FLUX.2 VAE with the small decoder. The loader that
        identification hands it to must build that decoder and take every key: checked on the meta device, at the
        real shapes."""
        from invokeai.backend.model_manager.load.model_loaders import flux as flux_module

        monkeypatch.setattr(flux_module, "load_file", lambda _path: _meta_state_dict(moss_vae_keys))
        seen: dict[str, list[str]] = {}
        load = flux_module.load_state_dict_ignoring_extras

        def spy(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], **kwargs: Any) -> list[str]:
            expected = set(model.state_dict())
            seen["missing"] = sorted(expected - set(state_dict))
            seen["unexpected"] = sorted(set(state_dict) - expected)
            return load(model, state_dict, **kwargs)

        monkeypatch.setattr(flux_module, "load_state_dict_ignoring_extras", spy)
        loader = object.__new__(flux_module.Flux2VAELoader)
        loader._torch_device = torch.device("cpu")
        loader._torch_dtype = torch.float32
        loader._logger = logging.getLogger(__name__)
        config = VAE_Checkpoint_Flux2_Config.model_construct(path="moss-image-vae.safetensors", name="moss-image-vae")

        vae = loader._load_model(config)

        assert seen == {"missing": [], "unexpected": []}
        assert vae.encoder.conv_in.out_channels == 128
        assert vae.decoder.conv_in.out_channels == 384


class TestTransformerGeometry:
    def test_a_gap_in_the_blocks_is_refused(self) -> None:
        sd = {key: value for key, value in _meta_state_dict(moss_dit_keys).items() if not key.startswith("layers.7.")}

        with pytest.raises(ValueError, match=r"gaps .*\[7\]"):
            moss_image_transformer_config(sd)

    def test_a_checkpoint_without_its_geometry_tensors_is_refused(self) -> None:
        sd = {key: value for key, value in _meta_state_dict(moss_dit_keys).items() if key != "video_embedder.weight"}

        with pytest.raises(ValueError, match="video_embedder.weight"):
            moss_image_transformer_config(sd)


class TestCheckpointLoader:
    @pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
    @pytest.mark.parametrize("with_rope_bands", [True, False])
    def test_a_single_file_loads_into_the_model_it_describes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prefix: str, with_rope_bands: bool
    ) -> None:
        reference = _tiny_reference()
        sd = _tiny_state_dict(reference)
        if not with_rope_bands:
            del sd["rope.bands"]
        path = tmp_path / "moss-image-dit.safetensors"
        save_file({f"{prefix}{key}": value for key, value in sd.items()}, str(path))
        loader = _loader(MossImageCheckpointModel, monkeypatch)
        config = Main_Checkpoint_MossImage_Config.model_construct(path=str(path), name="moss")

        model = loader._load_model(config, SubModelType.Transformer)

        _assert_loaded(model, reference)
        assert torch.equal(_forward(model), _forward(reference))
        # Reserved once, before anything was cast or assigned.
        loader._ram_cache.make_room.assert_called_once()

    def test_a_quantization_side_channel_is_refused(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        sd = _tiny_state_dict(_tiny_reference())
        sd["layers.0.proj.k.weight_scale"] = torch.ones(())
        path = tmp_path / "moss-fp8.safetensors"
        save_file(sd, str(path))
        loader = _loader(MossImageCheckpointModel, monkeypatch)
        config = Main_Checkpoint_MossImage_Config.model_construct(path=str(path), name="moss")

        with pytest.raises(ValueError, match="quantization side channel"):
            loader._load_model(config, SubModelType.Transformer)
        loader._ram_cache.make_room.assert_not_called()

    def test_the_checkpoint_refuses_submodels_it_does_not_contain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = _loader(MossImageCheckpointModel, monkeypatch)
        config = Main_Checkpoint_MossImage_Config.model_construct(path="moss.safetensors", name="moss")

        with pytest.raises(ValueError, match="holds only the transformer"):
            loader._load_model(config, SubModelType.VAE)


# A tiny Qwen3.5 text config in the shape of a transformers `Qwen3_5Model` `config.json`, as MOSS-Image ships it.
_TINY_TEXT_ENCODER_CONFIG = {
    "architectures": ["Qwen3_5Model"],
    "model_type": "qwen3_5",
    "text_config": {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
    },
}


class TestDiffusersLoader:
    def _config(self, path: Path) -> Main_Diffusers_MossImage_Config:
        return Main_Diffusers_MossImage_Config.model_construct(path=str(path), name="MOSS-Image-Base")

    def _transformer_dir(self, root: Path) -> Path:
        transformer_dir = root / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(
            json.dumps({"_class_name": "MOSSImageTransformer", "_diffusers_version": "0.38.0", **TINY_GEOMETRY})
        )
        return transformer_dir

    def test_the_sharded_transformer_loads_what_its_index_names_at_its_declared_geometry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        reference = _tiny_reference()
        sd = _tiny_state_dict(reference)
        transformer_dir = self._transformer_dir(tmp_path)
        keys = sorted(sd)
        shards = {
            "diffusion_pytorch_model-00001-of-00002.safetensors": keys[::2],
            "diffusion_pytorch_model-00002-of-00002.safetensors": keys[1::2],
        }
        for name, shard_keys in shards.items():
            save_file({k: sd[k] for k in shard_keys}, str(transformer_dir / name))
        _write_index(transformer_dir / "diffusion_pytorch_model.safetensors.index.json", shards)
        # Left behind by an earlier, differently sharded download: it sorts between the two and overlaps both.
        save_file(
            {key: torch.zeros_like(value) for key, value in sd.items()},
            str(transformer_dir / "diffusion_pytorch_model-00001-of-00003.safetensors"),
        )
        loader = _loader(MossImageDiffusersModel, monkeypatch)

        model = loader._load_model(self._config(tmp_path), SubModelType.Transformer)

        _assert_loaded(model, reference)
        loader._ram_cache.make_room.assert_called_once()

    def test_an_unsharded_transformer_needs_no_index(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        reference = _tiny_reference()
        transformer_dir = self._transformer_dir(tmp_path)
        save_file(_tiny_state_dict(reference), str(transformer_dir / "diffusion_pytorch_model.safetensors"))
        loader = _loader(MossImageDiffusersModel, monkeypatch)

        model = loader._load_model(self._config(tmp_path), SubModelType.Transformer)

        _assert_loaded(model, reference)

    def test_shards_without_an_index_are_refused(self, tmp_path: Path) -> None:
        for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            save_file({"weight": torch.zeros(1)}, str(tmp_path / name))

        with pytest.raises(ValueError, match="no safetensors index, so its weights must be one file, but it holds 2"):
            moss_image_component_shards(tmp_path)

    @pytest.mark.parametrize(
        ("shard_name", "message"),
        [("model-00002-of-00002.safetensors", "missing shard"), ("../model.safetensors", "not a file in")],
        ids=["missing_shard", "outside_the_folder"],
    )
    def test_an_index_naming_what_the_folder_does_not_hold_is_refused(
        self, tmp_path: Path, shard_name: str, message: str
    ) -> None:
        save_file({"a": torch.zeros(1)}, str(tmp_path / "model-00001-of-00002.safetensors"))
        _write_index(
            tmp_path / "model.safetensors.index.json",
            {"model-00001-of-00002.safetensors": ["a"], shard_name: ["b"]},
        )

        with pytest.raises(ValueError, match=message):
            moss_image_component_shards(tmp_path)

    def test_the_text_encoder_is_the_language_model_read_after_its_final_norm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from invokeai.backend.qwen3_5.qwen3_5_encoder import qwen3_5_text_config_from_config_json

        torch.manual_seed(0)
        reference = Qwen35Encoder(qwen3_5_text_config_from_config_json(_TINY_TEXT_ENCODER_CONFIG), final_norm=True)
        weights = reference.state_dict()
        text_encoder_dir = tmp_path / "text_encoder"
        text_encoder_dir.mkdir()
        (text_encoder_dir / "config.json").write_text(json.dumps(_TINY_TEXT_ENCODER_CONFIG))
        checkpoint = {
            **{f"language_model.{key}": value.contiguous() for key, value in weights.items()},
            "visual.blocks.0.attn.qkv.weight": torch.zeros(48, 16),
        }
        save_file(checkpoint, str(text_encoder_dir / "model-00001-of-00001.safetensors"))
        _write_index(
            text_encoder_dir / "model.safetensors.index.json", {"model-00001-of-00001.safetensors": list(checkpoint)}
        )
        # A stale shard the index does not name. Read along with the named one, it would replace the embedding.
        save_file(
            {"language_model.embed_tokens.weight": torch.zeros_like(weights["embed_tokens.weight"])},
            str(text_encoder_dir / "model-00001-of-00002.safetensors"),
        )
        loader = _loader(MossImageDiffusersModel, monkeypatch)

        encoder = loader._load_model(self._config(tmp_path), SubModelType.TextEncoder)

        assert isinstance(encoder, Qwen35Encoder)
        assert encoder.norm is not None
        assert encoder.state_dict().keys() == weights.keys()
        for key, value in weights.items():
            assert torch.equal(encoder.state_dict()[key], value), key

    def test_the_vae_never_loads_in_float16(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from diffusers import AutoencoderKLFlux2

        AutoencoderKLFlux2(
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(32,),
            layers_per_block=1,
            mid_block_add_attention=False,
        ).save_pretrained(tmp_path / "vae")
        loader = _loader(MossImageDiffusersModel, monkeypatch)
        # `precision: auto` picks float16 on CUDA and MPS, in which the FLUX autoencoders break.
        loader._torch_dtype = torch.float16

        vae = loader._load_model(self._config(tmp_path), SubModelType.VAE)

        assert isinstance(vae, AutoencoderKLFlux2)
        assert vae.dtype in (torch.bfloat16, torch.float32)

    def test_the_scheduler_is_not_a_submodel_it_loads(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        loader = _loader(MossImageDiffusersModel, monkeypatch)

        with pytest.raises(ValueError, match="not scheduler"):
            loader._load_model(self._config(tmp_path), SubModelType.Scheduler)

    def test_a_foreign_config_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = _loader(MossImageDiffusersModel, monkeypatch)

        with pytest.raises(ValueError, match="Main_Diffusers_MossImage_Config"):
            loader._load_model(SimpleNamespace(path="whatever"), SubModelType.Transformer)  # type: ignore[arg-type]


class TestFp8Storage:
    """FP8 Storage on the vendored transformer, through the loader's own load path.

    The vendored forward casts activations to some layers' weight dtype before calling them, so those layers must stay
    in the compute dtype. Float32 compute on the CPU: what is under test is the dtype plumbing, not bf16 numerics.
    """

    def _load_with_fp8_storage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, skip_patterns: tuple[str, ...]
    ) -> torch.nn.Module:
        monkeypatch.setattr(loader_module, "MOSS_IMAGE_FP8_SKIP_PATTERNS", skip_patterns)
        path = tmp_path / "moss.safetensors"
        save_file(_tiny_state_dict(_tiny_reference()), str(path))
        loader = _loader(MossImageCheckpointModel, monkeypatch)
        loader._should_use_fp8 = lambda _config, _submodel=None: True
        config = Main_Checkpoint_MossImage_Config.model_construct(path=str(path), name="moss")
        return loader._load_model(config, SubModelType.Transformer)

    def test_the_cast_model_runs_and_stays_close_to_the_full_precision_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        model = self._load_with_fp8_storage(monkeypatch, tmp_path, MOSS_IMAGE_FP8_SKIP_PATTERNS)

        stored_fp8 = {name for name, p in model.named_parameters() if p.dtype is torch.float8_e4m3fn}
        # The bulk of the blocks is stored in fp8; what the forward reads the dtype of is not.
        assert "layers.0.proj.k.weight" in stored_fp8
        assert "layers.0.proj.ffn_up_gate.weight" in stored_fp8
        assert "layers.0.proj.q.weight" not in stored_fp8
        assert "video_embedder.weight" not in stored_fp8

        output = _forward(model)
        expected = _forward(_tiny_reference())
        assert torch.isfinite(output).all()
        # e4m3 keeps three mantissa bits: a few percent per weight, which a forward of three blocks keeps that order.
        relative_error = (output - expected).norm() / expected.norm()
        assert relative_error < 0.1

    @pytest.mark.parametrize("dropped", MOSS_IMAGE_FP8_SKIP_PATTERNS)
    def test_every_skip_pattern_is_needed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dropped: str) -> None:
        model = self._load_with_fp8_storage(
            monkeypatch, tmp_path, tuple(p for p in MOSS_IMAGE_FP8_SKIP_PATTERNS if p != dropped)
        )

        with pytest.raises(RuntimeError):
            _forward(model)
