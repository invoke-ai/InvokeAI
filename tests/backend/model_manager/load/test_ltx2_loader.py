"""The LTX-2 loaders over toy checkpoints in the official key layout.

The transformer tests build the real ``LTX2VideoTransformer3DModel`` at toy size, write its weights
out under the *official* names (the inverse of the renames the loader applies) and hand that to the
loader: what is pinned is the whole conversion chain landing every tensor, the packed formats staying
packed under renamed paths too, and cache room being reserved before any weight widens -- the
contract the loader-seam driver records as ``order`` (reservations made before each observed step)
and ``dtypes_at_make_room`` (the working dict's dtypes when room was first made).
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from invokeai.backend.ltx2 import component_configs as cc
from invokeai.backend.model_manager.configs.gemma4_encoder import Gemma4Encoder_Gemma4Encoder_LTX2_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_LTX2_Config, Main_Diffusers_LTX2_Config
from invokeai.backend.model_manager.load.model_loaders import ltx2
from invokeai.backend.model_manager.load.model_loaders.ltx2 import (
    LTX2CheckpointModel,
    LTX2FolderModel,
    LTX2Gemma4EncoderModel,
)
from invokeai.backend.model_manager.taxonomy import LTX2VariantType, SubModelType
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, nvfp4_signed_tensors

COMPUTE_DTYPE = torch.bfloat16

# A one-block transformer with a 128-wide video stream (so an nvfp4 layer's 128-row tiles fit) and
# a 32-wide audio stream.
TINY_TRANSFORMER = {
    **cc.LTX2_5_TRANSFORMER_CONFIG,
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 64,
    "cross_attention_dim": 32,
    "audio_in_channels": 4,
    "audio_out_channels": 4,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 16,
    "audio_cross_attention_dim": 16,
    "num_layers": 1,
    "caption_channels": 8,
}

# diffusers attribute path -> official checkpoint spelling; ordered so a longer name is rewritten
# before the shorter name it contains. Written independently of the loader's tables: the test is
# that the loader's forward conversion lands every one of these back on the model.
_TO_OFFICIAL = (
    ("audio_time_embed.", "audio_adaln_single."),
    ("time_embed.", "adaln_single."),
    ("av_cross_attn_video_scale_shift", "av_ca_video_scale_shift_adaln_single"),
    ("av_cross_attn_video_a2v_gate", "av_ca_a2v_gate_adaln_single"),
    ("av_cross_attn_audio_scale_shift", "av_ca_audio_scale_shift_adaln_single"),
    ("av_cross_attn_audio_v2a_gate", "av_ca_v2a_gate_adaln_single"),
    ("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video"),
    ("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio"),
    ("audio_proj_in", "audio_patchify_proj"),
    ("proj_in", "patchify_proj"),
    ("norm_q", "q_norm"),
    ("norm_k", "k_norm"),
    ("audio_prompt_adaln.", "audio_prompt_adaln_single."),
    ("prompt_adaln.", "prompt_adaln_single."),
)
_EXCLUSIVE = {"audio_time_embed.", "audio_proj_in", "audio_prompt_adaln."}


def _official_name(diffusers_name: str, prefix: str = "model.diffusion_model.") -> str:
    name = diffusers_name
    for new, old in _TO_OFFICIAL:
        if new in name:
            name = name.replace(new, old, 1)
            if new in _EXCLUSIVE:
                break
    return prefix + name


def _tiny_transformer_checkpoint(
    *, keyframe_embedding: bool = True, prefix: str = "model.diffusion_model."
) -> dict[str, torch.Tensor]:
    from diffusers import LTX2VideoTransformer3DModel

    torch.manual_seed(0)
    model = LTX2VideoTransformer3DModel(**{**TINY_TRANSFORMER, "use_keyframes_abs_pos_embedding": keyframe_embedding})
    return {_official_name(k, prefix): v.detach().clone() for k, v in model.state_dict().items()}


def _tiny_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "LTX2_5_TRANSFORMER_CONFIG", TINY_TRANSFORMER)


def _transformer_seam() -> Seam:
    return Seam(
        loader=LTX2CheckpointModel,
        module=ltx2,
        load_file_host=safetensors.torch,
        compute_dtype=COMPUTE_DTYPE,
        patches_device=True,
    )


def _checkpoint_config(
    tmp_path: Path,
    sd: dict[str, torch.Tensor],
    name: str = "ltx-2.5-22b-dev_diffusion_model.safetensors",
    metadata: dict[str, str] | None = None,
):
    """The config for a checkpoint written to disk for real: the header readers open the path
    themselves, so a served state dict alone would leave them reading an empty file."""
    path = tmp_path / name
    safetensors.torch.save_file(sd, path, metadata=metadata)
    return Main_Checkpoint_LTX2_Config.model_construct(
        path=str(path), variant=LTX2VariantType.Dev, generation="2.5", fp8_storage=None
    )


OBSERVED = ("cast_state_dict", "install_int8_convrot_layers", "install_nvfp4_layers")

# Layers whose official and diffusers spellings differ: what the side channel has to follow.
INT8_LAYER = "proj_in"  # official: patchify_proj
NVFP4_LAYER = "time_embed.emb.timestep_embedder.linear_1"  # official: adaln_single.…


def _int8_checkpoint() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    sd = _tiny_transformer_checkpoint()
    key = _official_name(f"{INT8_LAYER}.weight")
    dense = sd[key]
    scale = dense.abs().amax(dim=1, keepdim=True) / 127.0
    sd[key] = torch.round(dense / scale).clamp(-127, 127).to(torch.int8)
    sd[_official_name(f"{INT8_LAYER}.weight_scale")] = scale.float()
    sd[_official_name(f"{INT8_LAYER}.comfy_quant")] = comfy_quant_marker(
        {"format": "int8_tensorwise", "convrot": False}
    )
    return sd, dense


def _room_was_made_before(run, step: str) -> bool:
    return any(name == step and rooms > 0 for name, rooms in run.order)


def test_a_dense_bf16_file_loads_every_tensor_reserving_before_the_cast(monkeypatch, tmp_path) -> None:
    sd = _tiny_transformer_checkpoint()
    run = prepare(
        _transformer_seam(), monkeypatch, state_dict=sd, metadata={}, geometry=_tiny_geometry, observe=OBSERVED
    )
    model = run.load(_checkpoint_config(tmp_path, sd))

    assert not any(p.is_meta for p in model.parameters())
    assert {p.dtype for p in model.parameters()} == {COMPUTE_DTYPE}
    assert model.config.use_keyframes_abs_pos_embedding is True
    assert run.casting_calls == 1, "a dense file is what the FP8 Storage policy applies to"
    assert _room_was_made_before(run, "cast_state_dict")
    # The working dict still held the on-disk dtype when room was made: reserved for what will
    # exist, not for what was already made.
    assert set(run.dtypes_at_make_room.values()) == {torch.float32}


def test_an_int8_convrot_layer_stays_packed_under_its_renamed_path(monkeypatch, tmp_path) -> None:
    sd, dense = _int8_checkpoint()
    run = prepare(
        _transformer_seam(), monkeypatch, state_dict=sd, metadata={}, geometry=_tiny_geometry, observe=OBSERVED
    )
    model = run.load(_checkpoint_config(tmp_path, sd))

    layer = model.get_submodule(INT8_LAYER)
    assert isinstance(layer, Int8ConvrotLinear)
    assert layer.weight.dtype == torch.int8
    torch.testing.assert_close(
        layer._dequantized_weight(torch.device("cpu"), torch.float32), dense, atol=0.05, rtol=0.05
    )
    assert run.casting_calls == 0, "packed formats are never re-encoded by the FP8 Storage pass"
    # The shared int8 helper makes the reservation itself, before its own split and cast: what is
    # observable from outside is that the working dict still held the int8 codes when it did.
    assert run.reserved and torch.int8 in run.dtypes_at_make_room.values()


def test_a_header_named_nvfp4_layer_stays_packed_under_its_renamed_path(monkeypatch, tmp_path) -> None:
    sd = _tiny_transformer_checkpoint()
    official = _official_name(f"{NVFP4_LAYER}.weight")
    shape = tuple(sd.pop(official).shape)
    assert shape[0] % 128 == 0 and shape[1] % 64 == 0, "the toy layer must fit nvfp4's tiles"
    module_path = official[len("model.diffusion_model.") : -len(".weight")]
    tensors, expected = nvfp4_signed_tensors(module_path, torch.randint(0, 2, shape, dtype=torch.bool))
    sd.update({f"model.diffusion_model.{k}": v for k, v in tensors.items()})
    metadata = {"_quantization_metadata": json.dumps({"layers": {module_path: {"format": "nvfp4"}}})}
    run = prepare(
        _transformer_seam(), monkeypatch, state_dict=sd, metadata=metadata, geometry=_tiny_geometry, observe=OBSERVED
    )
    model = run.load(_checkpoint_config(tmp_path, sd, metadata=metadata))

    layer = model.get_submodule(NVFP4_LAYER)
    assert isinstance(layer, NVFP4Linear)
    torch.testing.assert_close(
        layer._dequantized_weight(torch.device("cpu"), torch.float32), expected.float(), atol=0.0, rtol=0.0
    )
    assert not any(p.is_meta for p in model.parameters())
    assert _room_was_made_before(run, "install_nvfp4_layers")


def test_a_file_without_the_keyframe_embedding_builds_the_model_without_it(monkeypatch, tmp_path) -> None:
    sd = _tiny_transformer_checkpoint(keyframe_embedding=False)
    run = prepare(_transformer_seam(), monkeypatch, state_dict=sd, metadata={}, geometry=_tiny_geometry)
    model = run.load(_checkpoint_config(tmp_path, sd))
    assert model.config.use_keyframes_abs_pos_embedding is False
    assert getattr(model, "keyframes_abs_pos_embedding", None) is None


def test_a_bare_key_file_loads_the_same_as_a_prefixed_one(monkeypatch, tmp_path) -> None:
    """Identification accepts both spellings, so the loader has to as well."""
    sd = _tiny_transformer_checkpoint(prefix="")
    run = prepare(_transformer_seam(), monkeypatch, state_dict=sd, metadata={}, geometry=_tiny_geometry)
    model = run.load(_checkpoint_config(tmp_path, sd))
    assert not any(p.is_meta for p in model.parameters())


def test_bundled_component_tensors_are_ignored_not_refused(monkeypatch, tmp_path) -> None:
    sd = _tiny_transformer_checkpoint()
    sd["vae.encoder.conv_in.conv.weight"] = torch.zeros(4, 4)
    sd["model.diffusion_model.video_embeddings_connector.learnable_registers"] = torch.zeros(4, 4)
    run = prepare(_transformer_seam(), monkeypatch, state_dict=sd, metadata={}, geometry=_tiny_geometry)
    model = run.load(_checkpoint_config(tmp_path, sd))
    assert not any(p.is_meta for p in model.parameters())
    assert any("ignoring 2 tensor" in str(call) for call in run.loader._logger.info.call_args_list)


def test_a_header_declared_unsupported_format_is_refused_before_the_tensor_read(monkeypatch, tmp_path) -> None:
    sd = _tiny_transformer_checkpoint()
    header = {"_quantization_metadata": json.dumps({"layers": {"x": {"format": "fp8_scaled"}}})}
    config = _checkpoint_config(tmp_path, sd, metadata=header)
    run = prepare(_transformer_seam(), monkeypatch, state_dict=sd, metadata=header, geometry=_tiny_geometry)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda _p: pytest.fail("the tensors were read"))
    with pytest.raises(ValueError, match="fp8_scaled"):
        run.load(config)
    assert run.reserved == []


def test_undeclared_float8_weights_are_refused_by_name(monkeypatch, tmp_path) -> None:
    sd = _tiny_transformer_checkpoint()
    key = _official_name("transformer_blocks.0.attn1.to_v.weight")
    sd[key] = sd[key].to(torch.float8_e4m3fn)
    run = prepare(_transformer_seam(), monkeypatch, state_dict=sd, metadata={}, geometry=_tiny_geometry)
    with pytest.raises(ValueError, match="float8"):
        run.load(_checkpoint_config(tmp_path, sd))
    assert run.reserved == []


# --- component folder ----------------------------------------------------------------------------


def _folder_loader(reserved: list[int] | None = None) -> LTX2FolderModel:
    loader = object.__new__(LTX2FolderModel)
    loader._ram_cache = SimpleNamespace(make_room=(reserved.append if reserved is not None else lambda _n: None))
    loader._logger = SimpleNamespace(info=lambda *_a, **_k: None)
    loader._torch_device = torch.device("cpu")
    return loader


def _cpu(monkeypatch) -> None:
    monkeypatch.setattr(ltx2.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(ltx2.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: torch.float32))


def test_a_components_only_folder_refuses_the_transformer_with_the_next_step(tmp_path) -> None:
    config = Main_Diffusers_LTX2_Config.model_construct(
        path=str(tmp_path),
        components={"video_vae": "v.safetensors", "audio_vae": "a.safetensors"},
        components_only=True,
    )
    with pytest.raises(ValueError, match="components-only"):
        _folder_loader()._load_model(config, SubModelType.Transformer)


def test_a_missing_component_names_what_to_install(tmp_path) -> None:
    config = Main_Diffusers_LTX2_Config.model_construct(path=str(tmp_path), components={"video_vae": "v.safetensors"})
    with pytest.raises(ValueError, match="vocoder"):
        _folder_loader()._load_model(config, SubModelType.Vocoder)


def test_a_component_file_deleted_after_install_is_named_not_tracebacked(tmp_path) -> None:
    config = Main_Diffusers_LTX2_Config.model_construct(path=str(tmp_path), components={"vocoder": "gone.safetensors"})
    with pytest.raises(ValueError, match="gone.safetensors"):
        _folder_loader()._load_model(config, SubModelType.Vocoder)


def test_the_text_encoder_is_not_the_folder_s_to_serve(tmp_path) -> None:
    config = Main_Diffusers_LTX2_Config.model_construct(path=str(tmp_path), components={})
    with pytest.raises(ValueError, match="Gemma-4"):
        _folder_loader()._load_model(config, SubModelType.TextEncoder)


@pytest.mark.parametrize(
    ("tensors", "match"),
    [
        ({"encoder.conv_in.conv.weight": torch.zeros(2, 2, dtype=torch.float8_e4m3fn)}, "float8"),
        (
            {
                "encoder.conv_in.conv.weight": torch.zeros(2, 2, dtype=torch.int8),
                "encoder.conv_in.conv.weight_scale": torch.ones(2, 1),
            },
            "int8",
        ),
    ],
)
def test_a_quantized_component_file_is_refused_before_any_room_is_made(monkeypatch, tmp_path, tensors, match) -> None:
    safetensors.torch.save_file(tensors, tmp_path / "v.safetensors")
    config = Main_Diffusers_LTX2_Config.model_construct(path=str(tmp_path), components={"video_vae": "v.safetensors"})
    _cpu(monkeypatch)
    reserved: list[int] = []
    with pytest.raises(ValueError, match=match):
        _folder_loader(reserved)._load_model(config, SubModelType.VAE)
    assert reserved == []


def test_the_folder_sizes_the_files_a_submodel_is_read_from(tmp_path) -> None:
    """`_load_and_cache` reserves `get_size_fs` before the read; a folder has no `<submodel>/` dir
    to size, so the read would otherwise land on an unreserved cache."""
    for name, size in (
        ("p.safetensors", 100),
        ("vc.safetensors", 200),
        ("ac.safetensors", 300),
        ("v.safetensors", 1000),
    ):
        (tmp_path / name).write_bytes(b"\0" * size)
    config = Main_Diffusers_LTX2_Config.model_construct(
        path=str(tmp_path),
        components={
            "text_projection": "p.safetensors",
            "video_connector": "vc.safetensors",
            "audio_connector": "ac.safetensors",
            "video_vae": "v.safetensors",
        },
    )
    loader = _folder_loader()
    assert loader.get_size_fs(config, tmp_path, SubModelType.Connectors) == 600
    assert loader.get_size_fs(config, tmp_path, SubModelType.VAE) == 1000
    assert loader.get_size_fs(config, tmp_path, SubModelType.Vocoder) == 0


def test_the_connectors_are_read_out_of_a_bundled_transformer_file_by_key(monkeypatch, tmp_path) -> None:
    """The official transformer files carry the two connectors beside the transformer; only the
    connector tensors are read, and the strict connector load never sees the rest."""
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    tiny = {
        **cc.LTX2_5_CONNECTORS_CONFIG,
        "caption_channels": 4,
        "text_proj_in_factor": 2,
        "video_connector_num_attention_heads": 2,
        "video_connector_attention_head_dim": 4,
        "video_connector_num_layers": 1,
        "video_connector_num_learnable_registers": 2,
        "audio_connector_num_attention_heads": 2,
        "audio_connector_attention_head_dim": 2,
        "audio_connector_num_layers": 1,
        "audio_connector_num_learnable_registers": 2,
        "video_hidden_dim": 8,
        "audio_hidden_dim": 4,
    }
    monkeypatch.setattr(cc, "LTX2_5_CONNECTORS_CONFIG", tiny)
    torch.manual_seed(0)
    reference = LTX2TextConnectors(**tiny)
    official: dict[str, torch.Tensor] = {}
    for key, value in reference.state_dict().items():
        if key.startswith("video_text_proj_in."):
            official["text_embedding_projection.video_aggregate_embed." + key[len("video_text_proj_in.") :]] = value
        elif key.startswith("audio_text_proj_in."):
            official["text_embedding_projection.audio_aggregate_embed." + key[len("audio_text_proj_in.") :]] = value
        else:
            branch, rest = key.split(".", 1)
            rest = (
                rest.replace("transformer_blocks.", "transformer_1d_blocks.")
                .replace("norm_q", "q_norm")
                .replace("norm_k", "k_norm")
            )
            official[f"model.diffusion_model.{branch.replace('_connector', '_embeddings_connector')}.{rest}"] = value
    bundle = {**official, "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": torch.zeros(4, 4)}
    safetensors.torch.save_file(bundle, tmp_path / "bundle.safetensors")
    config = Main_Diffusers_LTX2_Config.model_construct(
        path=str(tmp_path),
        components={
            "text_projection": "bundle.safetensors",
            "video_connector": "bundle.safetensors",
            "audio_connector": "bundle.safetensors",
        },
    )
    _cpu(monkeypatch)
    model = _folder_loader()._load_model(config, SubModelType.Connectors)
    torch.testing.assert_close(model.video_connector.learnable_registers, reference.video_connector.learnable_registers)


# --- Gemma-4 text tower --------------------------------------------------------------------------


def _tiny_gemma4_config():
    from transformers import Gemma4TextConfig

    return Gemma4TextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=128,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=128,
        tie_word_embeddings=True,
    )


@pytest.fixture(params=["model.", "model.language_model."])
def gemma_folder(request, tmp_path: Path) -> SimpleNamespace:
    """The tower under either of the two prefixes the releases use, beside the tensors that are not
    the tower's (a vision tower, the tied head, a with-proj projection) and one int8 layer."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    torch.manual_seed(0)
    config = _tiny_gemma4_config()
    model = Gemma4TextModel(config)
    config_dict = {
        **config.to_dict(),
        "architectures": ["Gemma4UnifiedForCausalLM"],
        "model_type": "gemma4_unified_text",
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))
    prefix = request.param
    sd = {f"{prefix}{k}": v.detach().clone() for k, v in model.state_dict().items()}
    key = f"{prefix}layers.0.mlp.down_proj.weight"
    dense = sd[key]
    scale = dense.abs().amax(dim=1, keepdim=True) / 127.0
    sd[key] = torch.round(dense / scale).clamp(-127, 127).to(torch.int8)
    sd[f"{prefix}layers.0.mlp.down_proj.weight_scale"] = scale.float()
    sd[f"{prefix}layers.0.mlp.down_proj.comfy_quant"] = comfy_quant_marker(
        {"format": "int8_tensorwise", "convrot": False}
    )
    sd["hf_asset__video_aggregate_embed.weight"] = torch.zeros(4, 4)
    sd["model.vision_tower.patch_embedding.weight"] = torch.zeros(4, 4)
    sd["lm_head.weight"] = torch.zeros(128, 32)
    safetensors.torch.save_file(sd, tmp_path / "te.safetensors")
    return SimpleNamespace(root=tmp_path, reference=model)


def test_the_text_tower_loads_from_the_file_with_its_int8_layer_packed(monkeypatch, gemma_folder) -> None:
    _cpu(monkeypatch)
    loader = object.__new__(LTX2Gemma4EncoderModel)
    reserved: list[int] = []
    loader._ram_cache = SimpleNamespace(make_room=reserved.append)
    loader._logger = SimpleNamespace(info=lambda *_a, **_k: None)
    config = Gemma4Encoder_Gemma4Encoder_LTX2_Config.model_construct(
        path=str(gemma_folder.root), subfolder="", weight_file="te.safetensors"
    )

    model = loader._load_model(config, SubModelType.TextEncoder)

    assert isinstance(model.layers[0].mlp.down_proj, Int8ConvrotLinear)
    assert not any(p.is_meta for p in model.parameters())
    assert reserved, "cache room is reserved for the tower"
    ids = torch.tensor([[2, 5, 9, 1]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
        ref = gemma_folder.reference(input_ids=ids, output_hidden_states=True)
    assert len(out.hidden_states) == 3, "every layer's hidden state, plus the embedding output"
    torch.testing.assert_close(out.hidden_states[-1], ref.hidden_states[-1], atol=0.2, rtol=0.05)


def test_the_tokenizer_comes_from_the_same_folder(monkeypatch, tmp_path) -> None:
    calls: list[Path] = []
    import transformers

    monkeypatch.setattr(
        transformers, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda p, **_k: calls.append(p) or "tok")
    )
    loader = object.__new__(LTX2Gemma4EncoderModel)
    config = Gemma4Encoder_Gemma4Encoder_LTX2_Config.model_construct(
        path=str(tmp_path), subfolder="gemma4-12b-ltx-v1", weight_file="te.safetensors"
    )
    assert loader._load_model(config, SubModelType.Tokenizer) == "tok"
    assert calls == [tmp_path / "gemma4-12b-ltx-v1"]


@pytest.fixture
def unregistered_tokenizer_folder(tmp_path: Path) -> Path:
    """A folder that trips both of the heuristics the released Gemma-4 encoder trips.

    Real transformers, no weights: an unregistered ``model_type`` so ``AutoConfig`` falls back to the
    base config, no ``transformers_version`` so the Mistral-regex heuristic cannot short-circuit, and
    a vocabulary over 100k with a pre-tokenizer, which is what gates that heuristic at all. Written
    here rather than mocked so the warnings are the ones transformers actually emits: a reworded
    message or a renamed kwarg has to fail a test instead of silently returning the noise.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    root = tmp_path / "gemma4-12b-ltx-v1"
    root.mkdir()
    tokenizer = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(100_002)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "t0"})
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "gemma4_unified_text", "architectures": ["Gemma4UnifiedForCausalLM"]})
    )
    return root


@pytest.fixture
def transformers_warnings():
    """Every warning transformers emits, with its verbosity pinned so the capture cannot go empty."""
    import transformers

    logger = logging.getLogger("transformers")
    captured: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                captured.append(record.getMessage())

    handler = Capture()
    previous = transformers.utils.logging.get_verbosity()
    transformers.utils.logging.set_verbosity_warning()
    logger.addHandler(handler)
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        transformers.utils.logging.set_verbosity(previous)


def test_the_release_s_two_tokenizer_warnings_are_both_answered(unregistered_tokenizer_folder, transformers_warnings):
    """One is noise about a config that is discarded; the other is advice that would mis-tokenize
    every prompt if taken. Loading the same folder directly is the baseline: both fire there, so an
    empty capture after the loader means they were answered, not that they never happen."""
    from transformers import AutoTokenizer

    direct = AutoTokenizer.from_pretrained(unregistered_tokenizer_folder, local_files_only=True)
    baseline = list(transformers_warnings)
    assert sum("to instantiate a model of type" in message for message in baseline) == 1
    assert sum("fix_mistral_regex=True" in message for message in baseline) == 1

    transformers_warnings.clear()
    loader = object.__new__(LTX2Gemma4EncoderModel)
    config = Gemma4Encoder_Gemma4Encoder_LTX2_Config.model_construct(
        path=str(unregistered_tokenizer_folder.parent), subfolder="gemma4-12b-ltx-v1", weight_file="te.safetensors"
    )
    tokenizer = loader._load_model(config, SubModelType.Tokenizer)

    assert transformers_warnings == []
    # Declining the advice is not the same as taking it: the pre-tokenizer is the released one.
    assert str(tokenizer._tokenizer.pre_tokenizer) == str(direct._tokenizer.pre_tokenizer)


def test_a_mismatch_naming_another_model_type_survives_the_tokenizer_load(
    unregistered_tokenizer_folder, transformers_warnings
):
    """The probe matches a folder on its ``architectures`` alone, so a repack can declare anything.
    The message naming what it declares is the only signal at load time that it is not the release,
    and dropping the whole family of mismatch messages would eat it."""
    logger = logging.getLogger(ltx2._CONFIG_LOGGER_NAME)
    other = "You are using a model of type `mystery_arch_v9` to instantiate a model of type ``."

    with ltx2._quiet_unregistered_model_type():
        logger.warning(other)
    assert transformers_warnings == [other]

    transformers_warnings.clear()
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(unregistered_tokenizer_folder, local_files_only=True)
    assert any("to instantiate a model of type" in message for message in transformers_warnings), (
        "the release's own mismatch reaches the log outside a tokenizer load"
    )


def test_the_encoder_sizes_its_weight_file_one_directory_down(tmp_path) -> None:
    nested = tmp_path / "gemma4-12b-ltx-v1"
    nested.mkdir()
    (nested / "te.safetensors").write_bytes(b"\0" * 5000)
    (nested / "tokenizer.json").write_bytes(b"\0" * 40)
    config = Gemma4Encoder_Gemma4Encoder_LTX2_Config.model_construct(
        path=str(tmp_path), subfolder="gemma4-12b-ltx-v1", weight_file="te.safetensors"
    )
    loader = object.__new__(LTX2Gemma4EncoderModel)
    assert loader.get_size_fs(config, tmp_path, SubModelType.TextEncoder) == 5000
    assert loader.get_size_fs(config, tmp_path, SubModelType.Tokenizer) == 40


def test_an_nvfp4_layer_missing_its_global_scale_is_refused_before_the_cache_is_evicted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The degraded half-state, at this seam rather than at the detector.

    A packed uint8 weight with a block-scale grid and no `weight_scale_2` is what a guard keyed on
    `weight_scale_2` -- the key the decode keys on -- lets straight through. `_find_nvfp4_layers`
    refuses it and `test_nvfp4.py` pins that; what only a seam can answer is whether this loader
    reaches the detector before it asks the cache for room. Here a key conversion stands between the
    pop and the reservation, and nothing but their order keeps them that way round.
    """
    sd = _tiny_transformer_checkpoint()
    target = _official_name(f"{NVFP4_LAYER}.weight")[: -len(".weight")]
    rows, packed = sd[f"{target}.weight"].shape[0], sd[f"{target}.weight"].shape[1] // 2
    sd[f"{target}.weight"] = torch.zeros(rows, packed, dtype=torch.uint8)
    sd[f"{target}.weight_scale"] = torch.zeros(rows, packed // 8).to(torch.float8_e4m3fn)
    _tiny_geometry(monkeypatch)
    run = prepare(_transformer_seam(), monkeypatch)

    with pytest.raises(ValueError, match="with a weight_scale but no weight_scale_2"):
        run.load(_checkpoint_config(tmp_path, sd))

    assert run.reserved == []
