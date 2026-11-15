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
    Main_GGUF_MossImage_Config,
)
from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import moss_image as loader_module
from invokeai.backend.model_manager.load.model_loaders.moss_image import (
    MOSS_IMAGE_FP8_SKIP_PATTERNS,
    MossImageCheckpointModel,
    MossImageDiffusersModel,
    MossImageGGUFModel,
    moss_image_component_shards,
    moss_image_transformer_config,
)
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder
from tests.backend.model_manager.load.moss_image_fixtures import (
    TINY_GEOMETRY,
    forward,
    fuse_qkv,
    int8_convrot_state_dict,
    tiny_reference,
    with_state_dict,
    write_moss_image_gguf,
)
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

_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}


def _meta_state_dict(keys: dict[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.empty(shape, dtype=_SAFETENSORS_DTYPES[dtype], device="meta") for key, (shape, dtype) in keys.items()
    }


def _tiny_state_dict(reference: MOSSImageTransformer) -> dict[str, torch.Tensor]:
    return {key: value.clone().contiguous() for key, value in reference.state_dict().items()}


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

    def test_the_skip_patterns_hit_the_same_layers_whichever_way_a_path_matches_them(self) -> None:
        """FP8 Storage matches them with `re.search` on module names, the quantized paths with `in` on tensor keys.
        Both must pick exactly the Linears whose dtype the forward reads, on the real module and key lists."""
        with accelerate.init_empty_weights():
            model = MOSSImageTransformer(**RELEASED_GEOMETRY)
        expected = {
            "video_embedder",
            "time_embedding.0",
            "shared_block_modulation",
            "video_head_modulation",
            "final_linear_video",
            "text_embedder.proj",
        } | {f"layers.{index}.proj.q" for index in range(42)}
        linears = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

        by_regex = {name for name in linears if any(re.search(p, name) for p in MOSS_IMAGE_FP8_SKIP_PATTERNS)}
        by_substring_of_the_module = {
            name for name in linears if any(p in f"{name}.weight" for p in MOSS_IMAGE_FP8_SKIP_PATTERNS)
        }
        by_substring_of_real_keys = {
            key.removesuffix(".weight")
            for key in moss_dit_keys
            if key.endswith(".weight") and any(p in key for p in MOSS_IMAGE_FP8_SKIP_PATTERNS)
        }

        assert by_regex == by_substring_of_the_module == expected
        # The checkpoint's own key list, which also holds the norms and the biases' siblings: nothing else hits.
        assert by_substring_of_real_keys == expected
        # The near misses the patterns are written to leave quantizable.
        for name in ("time_embedding.2", "text_embedder.blocks.0.proj", "layers.0.proj.k", "layers.0.proj.ffn_down"):
            assert name in linears
            assert name not in by_regex

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
        reference = tiny_reference()
        sd = _tiny_state_dict(reference)
        if not with_rope_bands:
            del sd["rope.bands"]
        path = tmp_path / "moss-image-dit.safetensors"
        save_file({f"{prefix}{key}": value for key, value in sd.items()}, str(path))
        loader = _loader(MossImageCheckpointModel, monkeypatch)
        config = Main_Checkpoint_MossImage_Config.model_construct(path=str(path), name="moss")

        model = loader._load_model(config, SubModelType.Transformer)

        # Bit for bit: every parameter and the recomputed float32 RoPE frequencies.
        _assert_loaded(model, reference)
        # The same function, to float32 rounding rather than bit for bit. `assign=True` leaves the parameters where
        # safetensors put them, at the file's data offsets: only 8-byte aligned when the header length and the
        # 16-byte `rope.bands` fall that way. Linux x86 CPU kernels take a different, equally exact accumulation order
        # for such operands, so identical weights there give results a few ULP apart from the reference's freshly
        # allocated ones.
        torch.testing.assert_close(forward(model), forward(reference))
        # Reserved once, before anything was cast or assigned.
        loader._ram_cache.make_room.assert_called_once()

    def test_a_quantization_side_channel_is_refused(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        sd = _tiny_state_dict(tiny_reference())
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
        reference = tiny_reference()
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
        reference = tiny_reference()
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
        save_file(_tiny_state_dict(tiny_reference()), str(path))
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

        output = forward(model)
        expected = forward(tiny_reference())
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
            forward(model)


class TestFusedAndInt8Checkpoints:
    """Comfy-Org's two single files, fused `proj.qkv` in both, on a tiny transformer."""

    def _load(self, monkeypatch: pytest.MonkeyPatch, path: Path, *, fp8_storage: bool = False) -> tuple[Any, Any]:
        loader = _loader(MossImageCheckpointModel, monkeypatch)
        if fp8_storage:
            loader._should_use_fp8 = lambda _config, _submodel=None: True
        config = Main_Checkpoint_MossImage_Config.model_construct(path=str(path), name="moss")
        return loader._load_model(config, SubModelType.Transformer), loader

    @pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
    def test_a_fused_bf16_file_loads_as_the_split_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prefix: str
    ) -> None:
        reference = tiny_reference()
        path = tmp_path / "moss_image_base_bf16.safetensors"
        save_file({f"{prefix}{k}": v for k, v in fuse_qkv(_tiny_state_dict(reference)).items()}, str(path))

        model, loader = self._load(monkeypatch, path)

        _assert_loaded(model, reference)
        torch.testing.assert_close(forward(model), forward(reference))
        loader._ram_cache.make_room.assert_called_once()

    def test_an_int8_convrot_file_keeps_its_projections_int8_and_each_part_its_own_scale(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stored, meant = int8_convrot_state_dict(tiny_reference())
        path = tmp_path / "moss_image_base_int8_convrot.safetensors"
        save_file(stored, str(path))

        model, loader = self._load(monkeypatch, path)

        block = model.layers[1].proj
        # `proj` is a ModuleDict: the swap has to land in it, not on a stray attribute.
        for name in ("k", "v", "o", "ffn_up_gate", "ffn_down"):
            assert isinstance(block[name], Int8ConvrotLinear), name
        # What the forward reads the dtype of: dense, though the file holds it int8.
        assert type(block["q"]) is torch.nn.Linear
        assert block["q"].weight.dtype is torch.float32
        # k's rows and scales are the middle third of the fused tensor, untouched.
        assert torch.equal(block["k"].weight, stored["layers.1.proj.qkv.weight"][64:128])
        assert torch.equal(block["k"].weight_scale, stored["layers.1.proj.qkv.weight_scale"][64:128])
        assert torch.equal(block["v"].weight_scale, stored["layers.1.proj.qkv.weight_scale"][128:])
        # The text refiner and the stem were never quantized.
        assert type(model.text_embedder.blocks[0].qkv) is torch.nn.Linear
        # The same function as a dense model holding the values the file decodes to.
        torch.testing.assert_close(forward(model), forward(with_state_dict(meant)), rtol=1e-4, atol=1e-5)
        loader._ram_cache.make_room.assert_called_once()

    def test_an_int8_weight_without_its_marker_is_refused_not_cast_as_codes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stored, _ = int8_convrot_state_dict(tiny_reference())
        del stored["layers.0.proj.o.comfy_quant"]
        path = tmp_path / "moss-int8-unmarked.safetensors"
        save_file(stored, str(path))

        with pytest.raises(
            ValueError, match=r"no readable int8_tensorwise marker, e\.g\. \['layers\.0\.proj\.o\.weight'\]"
        ):
            self._load(monkeypatch, path)

    def test_fp8_storage_leaves_the_int8_layers_alone_and_still_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stored, meant = int8_convrot_state_dict(tiny_reference())
        path = tmp_path / "moss_image_base_int8_convrot.safetensors"
        save_file(stored, str(path))

        model, _ = self._load(monkeypatch, path, fp8_storage=True)

        assert all(isinstance(model.layers[0].proj[name], Int8ConvrotLinear) for name in ("k", "v", "o", "ffn_down"))
        assert model.layers[0].proj["q"].weight.dtype is torch.float32
        # The dense Linears nothing reads the dtype of are the ones that took the cast.
        assert model.text_embedder.blocks[0].qkv.weight.dtype is torch.float8_e4m3fn
        expected = forward(with_state_dict(meant))
        assert (forward(model) - expected).norm() / expected.norm() < 0.1


class TestGGUFLoader:
    """unsloth's split layout and leejet's fused, metadata-free one, as real tiny GGUF files."""

    def _load(self, monkeypatch: pytest.MonkeyPatch, path: Path) -> tuple[Any, Any]:
        loader = _loader(MossImageGGUFModel, monkeypatch)
        config = Main_GGUF_MossImage_Config.model_construct(path=str(path), name="moss")
        return loader._load_model(config, SubModelType.Transformer), loader

    @pytest.mark.parametrize(
        ("fused", "architecture"), [(False, "moss_image"), (True, None)], ids=["unsloth", "leejet"]
    )
    def test_the_gguf_computes_what_its_file_means(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fused: bool, architecture: str | None
    ) -> None:
        """Not bit-exact: torch's block kernels multiply codes by their fp16 scale in fp16, the reference reader in
        float32. A wrong third of the fused rows, a dropped norm or a missing key is off by far more."""
        path = tmp_path / "moss.gguf"
        meant = write_moss_image_gguf(path, tiny_reference(), fused=fused, architecture=architecture)

        model, loader = self._load(monkeypatch, path)
        apply_custom_layers_to_model(model)

        torch.testing.assert_close(forward(model), forward(with_state_dict(meant)), atol=1e-3, rtol=1e-3)
        loader._ram_cache.make_room.assert_called_once()

    @pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
    def test_only_the_projections_the_forward_never_reads_the_dtype_of_stay_packed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fused: bool
    ) -> None:
        path = tmp_path / "moss.gguf"
        write_moss_image_gguf(path, tiny_reference(), fused=fused, architecture=None if fused else "moss_image")

        model, _ = self._load(monkeypatch, path)

        block = model.layers[0].proj
        # leejet quantizes only the fused qkv and keeps the rest BF16, which unpacks at load; unsloth quantizes more.
        packed = ("k", "v") if fused else ("k", "v", "o", "ffn_up_gate", "ffn_down")
        for name in packed:
            assert isinstance(block[name].weight.data, GGMLTensor), name
        if not fused:
            assert isinstance(model.text_embedder.blocks[0].qkv.weight.data, GGMLTensor)
            # Quantized in the file, but the forward reads their dtype: the skip patterns unpack them.
            for module in (model.video_embedder, model.time_embedding[0], model.text_embedder.proj):
                assert type(module.weight.data) is torch.Tensor
        # `q` was quantized in the file (as part of the fused tensor, or on its own) and is dense here.
        assert type(block["q"].weight.data) is torch.Tensor
        assert block["q"].weight.dtype is torch.float32
        # Read outside a matmul, or BF16 in the file: nothing to keep packed.
        for module in (model.video_embedder, model.final_linear_video, model.shared_block_modulation):
            assert type(module.weight.data) is torch.Tensor
        assert type(model.layers[0].norm1.weight.data) is torch.Tensor

    def test_a_comfyui_prefixed_gguf_loads_like_an_unprefixed_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "moss.gguf"
        meant = write_moss_image_gguf(
            path, tiny_reference(), fused=True, architecture=None, prefix="model.diffusion_model."
        )

        model, _ = self._load(monkeypatch, path)
        apply_custom_layers_to_model(model)

        torch.testing.assert_close(forward(model), forward(with_state_dict(meant)), atol=1e-3, rtol=1e-3)

    def test_the_gguf_refuses_submodels_it_does_not_contain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = _loader(MossImageGGUFModel, monkeypatch)
        config = Main_GGUF_MossImage_Config.model_construct(path="moss.gguf", name="moss")

        with pytest.raises(ValueError, match="holds only the transformer"):
            loader._load_model(config, SubModelType.VAE)
