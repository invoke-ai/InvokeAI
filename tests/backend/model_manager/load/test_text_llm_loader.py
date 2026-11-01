"""Loading TextLLMs: a single-file Qwen3.5 vision-language model (the FLUX.3 prompt upsampler) and a model folder.

The geometry tests run the real layout of ComfyUI's builds (`flux3_pup_*_keys`) on the meta device against the
vendored 9B config. Everything that moves values runs a tiny `Qwen3_5ForConditionalGeneration` at float32 on the CPU,
through the loader's own entry point, with the vendored config swapped for the tiny one.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from transformers import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from invokeai.backend.model_manager.configs.text_llm import TextLLM_Checkpoint_Config, TextLLM_Diffusers_Config
from invokeai.backend.model_manager.load.model_loaders import text_llm as loader_module
from invokeai.backend.model_manager.load.model_loaders.text_llm import (
    TextLLMCheckpointLoader,
    TextLLMModelLoader,
    qwen3_5_vl_from_state_dict,
)
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.qwen3_5.qwen3_5_encoder import load_bundled_qwen3_5_tokenizer
from tests.backend.model_manager.load.state_dicts import flux3_pup_bf16_keys, flux3_pup_int8_convrot_keys
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot

Keys = dict[str, tuple[list[int], str]]

_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "I8": torch.int8, "U8": torch.uint8}

# Widened on load: the embedding is not a Linear, and the LM head is kept dense on purpose.
_WIDENED = {"lm_head", "model.language_model.embed_tokens"}


def _meta_state_dict(keys: Keys, markers: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    """The captured layout at full extents on the meta device, with the markers as the bytes a file holds."""
    sd: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in keys.items():
        if key.endswith(".comfy_quant"):
            sd[key] = comfy_quant_marker(markers[key.removesuffix(".comfy_quant")])
        else:
            sd[key] = torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
    return sd


def _nbytes(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(tensor.nelement() * tensor.element_size() for tensor in state_dict.values())


class TestTheReleasedLayouts:
    def test_the_int8_build_keeps_its_marked_linears_and_widens_the_embedding_and_lm_head(self) -> None:
        keys, markers = flux3_pup_int8_convrot_keys.state_dict_keys, flux3_pup_int8_convrot_keys.markers
        reserved: list[int] = []

        model = qwen3_5_vl_from_state_dict(
            _meta_state_dict(keys, markers), torch.bfloat16, reserve=reserved.append, what="pup", logger=MagicMock()
        )

        kept = {name: module for name, module in model.named_modules() if isinstance(module, Int8ConvrotLinear)}
        assert set(kept) == {path for path in markers if not path.startswith("mtp.")} - _WIDENED
        # The rotation group follows the input width: 256, 64 and 16 all occur, and each layer keeps its own.
        assert {module.group_size for module in kept.values()} == {256, 64, 16}
        assert all(
            module.convrot and module.group_size == markers[name]["convrot_groupsize"] for name, module in kept.items()
        )
        # The vision tower's quantized Linears keep their biases.
        assert all(kept[name].bias is not None for name in kept if name.startswith("model.visual."))
        assert type(model.lm_head) is torch.nn.Linear and model.lm_head.weight.dtype is torch.bfloat16
        assert model.model.language_model.embed_tokens.weight.dtype is torch.bfloat16

        # Every tensor of the file but the MTP head and the folded sidecars lands, at its stored shape.
        loaded = model.state_dict()
        expected = {
            key
            for key in keys
            if not key.startswith("mtp.")
            and not key.endswith(".comfy_quant")
            and key not in {f"{path}.weight_scale" for path in _WIDENED}
        }
        assert set(loaded) == expected
        assert all(list(loaded[key].shape) == keys[key][0] for key in expected)
        # One reservation, for exactly what the load ends with: the widened layers at bf16, the rest as stored.
        assert reserved == [_nbytes(loaded)]

    def test_the_bf16_build_loads_dense_with_every_key(self) -> None:
        keys = flux3_pup_bf16_keys.state_dict_keys
        reserved: list[int] = []

        model = qwen3_5_vl_from_state_dict(
            _meta_state_dict(keys, {}), torch.bfloat16, reserve=reserved.append, what="pup", logger=MagicMock()
        )

        assert not any(isinstance(module, Int8ConvrotLinear) for module in model.modules())
        loaded = model.state_dict()
        assert set(loaded) == set(keys)
        assert reserved == [_nbytes(loaded)]


# -- A tiny model, at the 9B's width so the width lookup is the real one, and float32 throughout. ------------------

_HIDDEN = 4096
_VOCAB = 64


def _tiny_config() -> Qwen3_5Config:
    text = Qwen3_5TextConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=256,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        out_hidden_size=_HIDDEN,
        num_position_embeddings=16,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
    )
    return Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict(), tie_word_embeddings=False)


# What the int8 build quantizes, one of each kind, with the group size its input width allows.
_QUANTIZED = {
    "model.language_model.embed_tokens": 256,
    "model.language_model.layers.0.linear_attn.in_proj_qkv": 256,
    "model.language_model.layers.0.mlp.gate_proj": 256,
    "model.language_model.layers.0.mlp.up_proj": 256,
    "model.language_model.layers.0.mlp.down_proj": 256,
    "model.language_model.layers.1.self_attn.q_proj": 256,
    "model.language_model.layers.1.self_attn.o_proj": 16,
    "model.visual.blocks.0.attn.qkv": 64,
    "model.visual.blocks.0.mlp.linear_fc2": 16,
    "lm_head": 256,
}


def _marker(group_size: int) -> torch.Tensor:
    return comfy_quant_marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size})


def _reference() -> Qwen3_5ForConditionalGeneration:
    torch.manual_seed(0)
    reference = Qwen3_5ForConditionalGeneration(_tiny_config()).eval()
    # transformers zero-initializes the norms (they scale by 1 + weight); random values make each one count.
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return reference


def _int8_checkpoint(reference: Qwen3_5ForConditionalGeneration) -> dict[str, torch.Tensor]:
    """The reference as the int8 build stores it. The reference then holds the round trip a correct load returns."""
    sd = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    for path, group_size in _QUANTIZED.items():
        payload = quantize_convrot(sd[f"{path}.weight"], group_size=group_size)
        sd[f"{path}.weight"], sd[f"{path}.weight_scale"] = payload.codes, payload.scale
        sd[f"{path}.comfy_quant"] = _marker(group_size)
        with torch.no_grad():
            reference.get_submodule(path).weight.copy_(payload.dequantized)
    # The multi-token-prediction head: a marked int8 layer the built model has no module for, and a dense tensor.
    mtp = quantize_convrot(torch.randn(8, _HIDDEN))
    sd["mtp.layers.0.mlp.gate_proj.weight"] = mtp.codes
    sd["mtp.layers.0.mlp.gate_proj.weight_scale"] = mtp.scale
    sd["mtp.layers.0.mlp.gate_proj.comfy_quant"] = _marker(256)
    sd["mtp.fc.weight"] = torch.randn(8, 8)
    return sd


def _seam(loader: type, compute_dtype: torch.dtype) -> Seam:
    """Either TextLLM loader, with `compute_dtype` as the configured precision."""
    return Seam(
        loader=loader, module=loader_module, entry="_load_model", compute_dtype=compute_dtype, casts_fp8_storage=False
    )


SEAM = _seam(TextLLMCheckpointLoader, torch.float32)
FOLDER_SEAM = _seam(TextLLMModelLoader, torch.float32)


def _config(tmp_path: Path, cpu_only: bool | None = None) -> TextLLM_Checkpoint_Config:
    path = tmp_path / "qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors"
    path.write_bytes(b"\0" * 64)
    return TextLLM_Checkpoint_Config.model_construct(path=str(path), name=path.stem, cpu_only=cpu_only)


def _tiny_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader_module, "qwen3_5_vl_config", lambda _variant: _tiny_config())


def _text_logits(model: Qwen3_5ForConditionalGeneration) -> torch.Tensor:
    input_ids = torch.tensor([[1, 5, 9, 33, 2, 63, 17, 40]])
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


def _image_features(model: Qwen3_5ForConditionalGeneration) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    # A 4x4 patch grid; each patch is 3 channels x 2 frames x 2 x 2 pixels.
    pixel_values = torch.randn(16, 24, generator=generator)
    with torch.no_grad():
        return model.model.visual(pixel_values, grid_thw=torch.tensor([[1, 4, 4]])).pooler_output


class TestTheCheckpointLoader:
    def test_an_int8_checkpoint_computes_what_its_dequantized_weights_compute(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        reference = _reference()
        run = prepare(SEAM, monkeypatch, state_dict=_int8_checkpoint(reference), geometry=_tiny_geometry)

        model = run.load(_config(tmp_path), None)

        kept = {name for name, module in model.named_modules() if isinstance(module, Int8ConvrotLinear)}
        assert kept == set(_QUANTIZED) - _WIDENED
        assert not any(name.startswith("mtp") for name in model.state_dict())
        # The same numbers through every route: text through the language model and the dense LM head, an image
        # through the vision tower's biased int8 Linears and its merger.
        expected_logits = _text_logits(reference)
        assert expected_logits.abs().max() > 0.1
        torch.testing.assert_close(_text_logits(model), expected_logits, rtol=1e-4, atol=1e-5)
        expected_features = _image_features(reference)
        assert expected_features.abs().max() > 0.1
        torch.testing.assert_close(_image_features(model), expected_features, rtol=1e-4, atol=1e-5)
        assert run.reserved == [_nbytes(model.state_dict())]

    def test_the_model_generates_with_the_releases_end_of_turn_tokens_and_real_rope_buffers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        reference = _reference()
        run = prepare(SEAM, monkeypatch, state_dict=_int8_checkpoint(reference), geometry=_tiny_geometry)

        model = run.load(_config(tmp_path), None)

        # `<|im_end|>` and `<|endoftext|>`: a model built from the config alone stops at the first only.
        assert model.generation_config.eos_token_id == [248046, 248044]
        assert model.generation_config.pad_token_id == 248044
        # RoPE frequencies are derived, never stored: they have to be real, and float32 like a fresh model's.
        for name, buffer in reference.named_buffers():
            assert torch.equal(model.get_buffer(name), buffer), name

    def test_a_dense_checkpoint_loads_as_it_was_saved(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        reference = _reference()
        state_dict = {key: value.detach().clone() for key, value in reference.state_dict().items()}
        run = prepare(SEAM, monkeypatch, state_dict=state_dict, geometry=_tiny_geometry)

        model = run.load(_config(tmp_path), None)

        assert not any(isinstance(module, Int8ConvrotLinear) for module in model.modules())
        torch.testing.assert_close(_text_logits(model), _text_logits(reference))
        assert run.reserved == [_nbytes(model.state_dict())]

    @pytest.mark.parametrize(
        ("cpu_only", "expected"), [(None, torch.bfloat16), (True, torch.float32)], ids=["on the device", "cpu only"]
    )
    def test_with_float16_configured_it_loads_in_a_dtype_gated_deltanet_runs_in(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_only: bool | None, expected: torch.dtype
    ) -> None:
        """Float16 is what `precision: auto` gives on CUDA, and Gated DeltaNet does not run in it; a CPU-only model
        takes float32, as CPU float16 is emulated. Loaded, not run: a CPU bfloat16 forward is not something every
        test runner can do."""
        state_dict = {key: value.detach().clone() for key, value in _reference().state_dict().items()}
        run = prepare(
            _seam(TextLLMCheckpointLoader, torch.float16), monkeypatch, state_dict=state_dict, geometry=_tiny_geometry
        )

        model = run.load(_config(tmp_path, cpu_only), None)

        assert {parameter.dtype for parameter in model.parameters()} == {expected}

    def test_a_scaled_fp8_checkpoint_is_refused_before_room_is_reserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        state_dict = {key: value.detach().clone() for key, value in _reference().state_dict().items()}
        path = "model.language_model.layers.0.mlp.gate_proj"
        state_dict[f"{path}.weight"] = state_dict[f"{path}.weight"].to(torch.float8_e4m3fn)
        state_dict[f"{path}.weight_scale"] = torch.tensor(0.01)
        run = prepare(SEAM, monkeypatch, state_dict=state_dict, geometry=_tiny_geometry)

        with pytest.raises(ValueError, match="quantization side channel"):
            run.load(_config(tmp_path), None)
        assert run.reserved == []


# -- The text side a single file cannot carry, pinned against the release's own tokenizer and processor. -----------

# `Write the caption.` / an image and `a red fox, 狐狸 🦊`, rendered with `enable_thinking=False`: the ids the release's
# tokenizer (`black-forest-labs/flux-3-image-klein-pup` @ 3f3edb52) produces, with the one image placeholder.
_MESSAGES_TEXT = [
    {"role": "system", "content": "Write the caption."},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "a red fox, 狐狸 🦊"}]},
]
_RELEASE_IDS = [
    248045, 8678, 198, 7734, 279, 16746, 13, 248046, 198, 248045, 846, 198, 248053, 248056, 248054, 64, 2438, 37550,
    11, 220, 113481, 10838, 99, 232, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
]  # fmt: skip
_IMAGE_PAD = 248056


class TestTheCheckpointsTextSide:
    """Each load serves a new tokenizer, never the object the Qwen3.5 encoders share: their truncating encodes
    reconfigure a fast tokenizer, and an encode running on it in another thread then fails with "Already borrowed"."""

    def test_the_tokenizer_renders_as_the_release_does(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        tokenizer = prepare(SEAM, monkeypatch).load(_config(tmp_path), SubModelType.Tokenizer)

        assert tokenizer is not load_bundled_qwen3_5_tokenizer()
        rendered = tokenizer.apply_chat_template(
            _MESSAGES_TEXT, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert tokenizer(rendered, add_special_tokens=False)["input_ids"] == _RELEASE_IDS

    def test_the_processor_expands_the_image_as_the_release_does(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        processor = prepare(SEAM, monkeypatch).load(_config(tmp_path), SubModelType.Processor)
        # Solid red with a little blue: each channel normalizes to one value, (v / 255 - 0.5) / 0.5.
        image = Image.new("RGB", (64, 48), (255, 0, 51))
        messages = [
            _MESSAGES_TEXT[0],
            {"role": "user", "content": [{"type": "image", "image": image}, _MESSAGES_TEXT[1]["content"][1]]},
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        assert processor.tokenizer is not load_bundled_qwen3_5_tokenizer()
        # The release's 65536-pixel floor scales 64x48 up to 320x224: a 20 x 14 grid of 16 px patches, merged 2x2.
        assert inputs["image_grid_thw"].tolist() == [[1, 14, 20]]
        image_tokens = [_IMAGE_PAD] * (14 * 20 // 4)
        placeholder = _RELEASE_IDS.index(_IMAGE_PAD)
        expected = _RELEASE_IDS[:placeholder] + image_tokens + _RELEASE_IDS[placeholder + 1 :]
        assert inputs["input_ids"].flatten().tolist() == expected
        assert torch.unique(inputs["pixel_values"]).tolist() == pytest.approx([-1.0, -0.6, 1.0])


# -- The folder loader picks the class that keeps a vision tower, in a dtype the model's attention can run in. -------


def _folder_config(path: Path, cpu_only: bool | None = None) -> TextLLM_Diffusers_Config:
    return TextLLM_Diffusers_Config.model_construct(path=str(path), name=path.name, cpu_only=cpu_only)


def _tiny_gemma_4() -> Gemma4ForConditionalGeneration:
    text = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,
    )
    vision = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        patch_size=4,
        position_embedding_size=64,
    )
    config = Gemma4Config(text_config=text.to_dict(), vision_config=vision.to_dict(), audio_config=None)
    return Gemma4ForConditionalGeneration(config)


def _tiny_qwen3_5_folder(path: Path) -> Path:
    tiny = _tiny_config()
    tiny.text_config.hidden_size = 64
    tiny.vision_config.out_hidden_size = 64
    Qwen3_5ForConditionalGeneration(tiny).save_pretrained(path)
    return path


class TestTheFolderLoader:
    @pytest.mark.parametrize(
        ("cpu_only", "expected"), [(None, torch.bfloat16), (True, torch.float32)], ids=["on the device", "cpu only"]
    )
    def test_a_qwen3_5_vision_language_folder_loads_with_its_vision_tower_never_in_float16(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_only: bool | None, expected: torch.dtype
    ) -> None:
        folder = _tiny_qwen3_5_folder(tmp_path / "pup")
        run = prepare(_seam(TextLLMModelLoader, torch.float16), monkeypatch)

        model = run.load(_folder_config(folder, cpu_only), None)

        # `AutoModelForCausalLM` would have built the text-only `Qwen3_5ForCausalLM`.
        assert type(model) is Qwen3_5ForConditionalGeneration
        assert model.dtype is expected

    def test_a_gemma_4_folder_loads_with_its_vision_tower(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Gemma-4 is not on the image-text-to-text list: `AutoModelForCausalLM` builds the whole model for it."""
        saved = _tiny_gemma_4()
        saved.save_pretrained(tmp_path / "gemma4")

        model = prepare(FOLDER_SEAM, monkeypatch).load(_folder_config(tmp_path / "gemma4"), None)

        assert type(model) is Gemma4ForConditionalGeneration
        tower = model.model.vision_tower.state_dict()
        assert tower.keys() == saved.model.vision_tower.state_dict().keys()
        assert all(torch.equal(tower[key], value) for key, value in saved.model.vision_tower.state_dict().items())

    @pytest.mark.parametrize("ships_one", [False, True], ids=["stock release", "own generation config"])
    def test_a_qwen3_5_folder_stops_at_the_end_of_the_assistants_turn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ships_one: bool
    ) -> None:
        """A stock Qwen3.5 release ships no generation config; derived from its config, generation would stop only
        at `<|endoftext|>`. A folder that ships one -- the FLUX.3 prompt upsampler does -- keeps its own."""
        folder = _tiny_qwen3_5_folder(tmp_path / "qwen3.5")
        if ships_one:
            (folder / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 8], "pad_token_id": 9}))
        else:
            (folder / "generation_config.json").unlink()

        model = prepare(FOLDER_SEAM, monkeypatch).load(_folder_config(folder), None)

        # `<|im_end|>` and `<|endoftext|>`, padded with the latter, as the upsampler's release generates.
        expected = ([7, 8], 9) if ships_one else ([248046, 248044], 248044)
        assert (model.generation_config.eos_token_id, model.generation_config.pad_token_id) == expected

    def test_any_other_causal_lm_keeps_the_configured_dtype_and_its_own_end_of_turn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
        )
        LlamaForCausalLM(config).save_pretrained(tmp_path / "llama")
        (tmp_path / "llama" / "generation_config.json").unlink()
        run = prepare(_seam(TextLLMModelLoader, torch.float16), monkeypatch)

        model = run.load(_folder_config(tmp_path / "llama"), None)

        assert type(model) is LlamaForCausalLM
        assert model.dtype is torch.float16
        assert model.generation_config.eos_token_id == config.eos_token_id

    @pytest.mark.parametrize("submodel", [SubModelType.Tokenizer, SubModelType.Processor])
    def test_the_text_side_reserves_no_cache_room_for_the_models_weights(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, submodel: SubModelType
    ) -> None:
        folder = tmp_path / "llm"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"\0" * 1024)
        config = _folder_config(folder)
        loader = prepare(FOLDER_SEAM, monkeypatch).loader

        assert loader.get_size_fs(config, folder, submodel) == 0
        assert loader.get_size_fs(config, folder, None) == 1024
