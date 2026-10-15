"""How LTX-2 releases are recognised on disk: the per-file role classifier, the folder and
single-file configs, the Gemma-4 encoder folder, and the key maps onto the diffusers classes.

The files here are real (tiny) safetensors written to a temp directory and probed through
``ModelOnDisk`` / the config factory, so what is tested is the same header-only read the installer
does -- including that no other config claims these folders first.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.ltx2 import checkpoint_layout as layout
from invokeai.backend.ltx2 import component_configs as cc
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.gemma4_encoder import Gemma4Encoder_Gemma4Encoder_LTX2_Config
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_LTX2_Config, Main_Diffusers_LTX2_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, LTX2VariantType, ModelFormat, ModelType

T = "model.diffusion_model."


def _probe(path: Path, overrides: dict | None = None):
    """Classify through the factory, which supplies the record fields (key, hash, name, ...) that a
    config's ``from_model_on_disk`` expects in its override dict."""
    return ModelConfigFactory.from_model_on_disk(ModelOnDisk(path), overrides or {}, allow_unknown=False)


def _rejection(path: Path, config_class: type) -> Exception:
    detail = _probe(path).details[config_class.__name__]
    assert isinstance(detail, Exception), f"{config_class.__name__} matched: {detail}"
    return detail


def _transformer_tensors(*, video_ff_bias: bool = False, keyframe: bool = True) -> dict[str, torch.Tensor]:
    sd = {
        f"{T}{layout.TRANSFORMER_FINGERPRINT_KEY}": torch.zeros(2, 2),
        f"{T}transformer_blocks.0.ff.net.0.proj.weight": torch.zeros(2, 2),
        f"{T}transformer_blocks.0.audio_ff.net.2.bias": torch.zeros(2),
        f"{T}transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2),
    }
    if video_ff_bias:
        sd[f"{T}transformer_blocks.0.ff.net.2.bias"] = torch.zeros(2)
    if keyframe:
        sd[f"{T}keyframes_abs_pos_embedding"] = torch.zeros(1, 2)
    return sd


COMPONENT_TENSORS: dict[str, dict[str, torch.Tensor]] = {
    "ltx-2.5-22b_video_vae_bf16.safetensors": {
        "encoder.conv_in.conv.weight": torch.zeros(2, 2, 3, 3, 3),
        "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight": torch.zeros(2, 2, 3, 3, 3),
        "decoder.up_blocks.8.res_blocks.0.conv1.conv.weight": torch.zeros(2, 2, 3, 3, 3),
        "per_channel_statistics.mean-of-means": torch.zeros(2),
    },
    "ltx-2.5-22b_audio_vae_bf16.safetensors": {"audio_vae.encoder.conv_in.weight": torch.zeros(2, 2, 3, 3)},
    "ltx-2.5-22b_vocoder_bf16.safetensors": {"vocoder.vocoder.conv_pre.weight": torch.zeros(2, 2, 3)},
    "ltx-2.5-22b_text_embedding_projection_bf16.safetensors": {
        "text_embedding_projection.video_aggregate_embed.weight": torch.zeros(2, 2)
    },
    "ltx-2.5-22b_video_embeddings_connector_bf16.safetensors": {
        f"{T}video_embeddings_connector.learnable_registers": torch.zeros(2, 2)
    },
    "ltx-2.5-22b_audio_embeddings_connector_bf16.safetensors": {
        f"{T}audio_embeddings_connector.learnable_registers": torch.zeros(2, 2)
    },
    "ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors": {
        "initial_conv.weight": torch.zeros(2, 2, 3, 3, 3),
        "res_blocks.0.conv1.weight": torch.zeros(2, 2, 3, 3, 3),
        "final_conv.weight": torch.zeros(2, 2, 3, 3, 3),
        "upsampler.0.weight": torch.zeros(8, 2, 3, 3),
    },
    "ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors": {
        "initial_conv.weight": torch.zeros(2, 2, 3, 3, 3),
        "res_blocks.0.conv1.weight": torch.zeros(2, 2, 3, 3, 3),
        "final_conv.weight": torch.zeros(2, 2, 3, 3, 3),
        "upsampler.0.weight": torch.zeros(4, 2, 3, 3, 3),
    },
}

EXPECTED_ROLES = {
    "ltx-2.5-22b_video_vae_bf16.safetensors": layout.ROLE_VIDEO_VAE,
    "ltx-2.5-22b_audio_vae_bf16.safetensors": layout.ROLE_AUDIO_VAE,
    "ltx-2.5-22b_vocoder_bf16.safetensors": layout.ROLE_VOCODER,
    "ltx-2.5-22b_text_embedding_projection_bf16.safetensors": layout.ROLE_TEXT_PROJECTION,
    "ltx-2.5-22b_video_embeddings_connector_bf16.safetensors": layout.ROLE_VIDEO_CONNECTOR,
    "ltx-2.5-22b_audio_embeddings_connector_bf16.safetensors": layout.ROLE_AUDIO_CONNECTOR,
    "ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors": layout.ROLE_SPATIAL_UPSAMPLER,
    "ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors": layout.ROLE_TEMPORAL_UPSAMPLER,
}


def _write_components(folder: Path, names=EXPECTED_ROLES, metadata: dict[str, str] | None = None) -> None:
    folder.mkdir(exist_ok=True)
    for name in names:
        save_file(COMPONENT_TENSORS[name], folder / name, metadata=metadata)


# --- the classifier ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_ROLES))
def test_each_official_component_file_is_classified_by_its_keys_alone(name: str) -> None:
    assert layout.classify_roles(COMPONENT_TENSORS[name]) == {EXPECTED_ROLES[name]}


def test_the_header_flags_decide_an_upsampler_when_present() -> None:
    spatial_keys = COMPONENT_TENSORS["ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors"]
    header = {"config": json.dumps({"_class_name": "LatentUpsampler", "temporal_upsample": True})}
    assert layout.classify_roles(spatial_keys, header) == {layout.ROLE_TEMPORAL_UPSAMPLER}


def test_a_transformer_is_recognised_with_or_without_its_prefix_and_a_bundle_carries_every_role() -> None:
    assert layout.classify_roles(_transformer_tensors()) == {layout.ROLE_TRANSFORMER}
    assert layout.classify_roles({k[len(T) :]: v for k, v in _transformer_tensors().items()}) == {
        layout.ROLE_TRANSFORMER
    }
    bundle = {
        **_transformer_tensors(),
        **{f"vae.{k}": v for k, v in COMPONENT_TENSORS["ltx-2.5-22b_video_vae_bf16.safetensors"].items()},
    }
    assert layout.classify_roles(bundle) == {layout.ROLE_TRANSFORMER, layout.ROLE_VIDEO_VAE}


def test_a_diffusion_decoder_vae_is_not_mistaken_for_the_conv_vae() -> None:
    keys = {"encoder.conv_in.conv.weight": 0, "decoder.diff_blocks.0.x": 0}
    assert layout.classify_roles(keys) == {layout.ROLE_DIFFUSION_VIDEO_VAE}


def test_unrelated_files_yield_no_role() -> None:
    assert layout.classify_roles({"double_blocks.0.img_attn.qkv.weight": 0}) == set()


def test_versions_reduce_to_a_generation() -> None:
    assert layout.generation_from_version("2.5.0") == "2.5"
    assert layout.generation_from_version("2.3") == "2.3"
    assert layout.generation_from_version("dev") is None
    assert layout.generation_from_version(None) is None


# --- folder config -------------------------------------------------------------------------------


def test_a_component_folder_records_every_role_and_is_components_only(tmp_path: Path) -> None:
    _write_components(tmp_path)
    config = _probe(tmp_path).config
    assert isinstance(config, Main_Diffusers_LTX2_Config)
    assert config.components == {role: name for name, role in EXPECTED_ROLES.items()}
    assert config.components_only is True
    assert config.variant is LTX2VariantType.Dev
    assert config.generation == "2.5"


def test_a_folder_with_a_distilled_transformer_is_a_full_install_of_that_variant(tmp_path: Path) -> None:
    _write_components(tmp_path)
    save_file(_transformer_tensors(), tmp_path / "ltx-2.5-22b-distilled_diffusion_model_bf16.safetensors")
    config = _probe(tmp_path).config
    assert isinstance(config, Main_Diffusers_LTX2_Config)
    assert config.components_only is False
    assert config.variant is LTX2VariantType.Distilled
    assert config.components[layout.ROLE_TRANSFORMER] == "ltx-2.5-22b-distilled_diffusion_model_bf16.safetensors"


def test_the_factory_hands_a_component_folder_to_the_ltx2_config_and_nothing_else(tmp_path: Path) -> None:
    _write_components(tmp_path)
    result = _probe(tmp_path)
    assert isinstance(result.config, Main_Diffusers_LTX2_Config)
    assert result.config.base is BaseModelType.LTX2 and result.config.format is ModelFormat.Diffusers
    assert sum(not isinstance(d, Exception) for d in result.details.values()) == 1, "no other config claims it"


def test_a_folder_without_both_vaes_is_not_an_ltx2_component_source(tmp_path: Path) -> None:
    _write_components(tmp_path, names=[n for n in EXPECTED_ROLES if "audio_vae" not in n])
    rejection = _rejection(tmp_path, Main_Diffusers_LTX2_Config)
    assert isinstance(rejection, NotAMatchError) and "video VAE + audio VAE" in str(rejection)


def test_a_diffusers_layout_folder_is_left_unclaimed(tmp_path: Path) -> None:
    _write_components(tmp_path)
    (tmp_path / "model_index.json").write_text("{}")
    rejection = _rejection(tmp_path, Main_Diffusers_LTX2_Config)
    assert isinstance(rejection, NotAMatchError) and "diffusers-layout" in str(rejection)


def test_an_earlier_generation_s_vae_is_refused_by_name(tmp_path: Path) -> None:
    old_vae = dict(COMPONENT_TENSORS["ltx-2.5-22b_video_vae_bf16.safetensors"])
    old_vae.pop("decoder.up_blocks.8.res_blocks.0.conv1.conv.weight")
    _write_components(tmp_path, names=[n for n in EXPECTED_ROLES if "video_vae" not in n])
    save_file(old_vae, tmp_path / "ltx-2.3-22b_vae.safetensors")
    rejection = _rejection(tmp_path, Main_Diffusers_LTX2_Config)
    assert "cannot tell which LTX-2 generation" in str(rejection)


def test_the_header_version_is_read_when_the_file_carries_one(tmp_path: Path) -> None:
    _write_components(
        tmp_path, names=[n for n in EXPECTED_ROLES if "video_vae" not in n], metadata={"model_version": "2.3.0"}
    )
    save_file(
        COMPONENT_TENSORS["ltx-2.5-22b_video_vae_bf16.safetensors"],
        tmp_path / "vae.safetensors",
        metadata={"model_version": "2.3.0"},
    )
    rejection = _rejection(tmp_path, Main_Diffusers_LTX2_Config)
    assert isinstance(rejection, InvalidMatchError)
    assert "LTX-2.3" in str(rejection) and "only LTX-2.5" in str(rejection)


# --- single-file transformer config ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors", LTX2VariantType.Dev),
        ("ltx-2.5-22b-distilled_diffusion_model_nvfp4.safetensors", LTX2VariantType.Distilled),
    ],
)
def test_a_transformer_file_is_dated_by_its_structure_and_named_by_its_filename(tmp_path, name, variant) -> None:
    save_file(_transformer_tensors(keyframe=False), tmp_path / name)
    result = _probe(tmp_path / name)
    assert isinstance(result.config, Main_Checkpoint_LTX2_Config)
    assert result.config.variant is variant
    assert result.config.generation == "2.5"
    assert result.config.type is ModelType.Main


def test_the_variant_override_wins_over_the_filename(tmp_path) -> None:
    path = tmp_path / "renamed.safetensors"
    save_file(_transformer_tensors(), path)
    config = _probe(path, {"variant": "ltx2_distilled"}).config
    assert isinstance(config, Main_Checkpoint_LTX2_Config)
    assert config.variant is LTX2VariantType.Distilled


def test_a_transformer_with_video_feed_forward_biases_is_an_earlier_generation(tmp_path) -> None:
    path = tmp_path / "ltx-2.3-22b-dev.safetensors"
    save_file(_transformer_tensors(video_ff_bias=True), path)
    rejection = _rejection(path, Main_Checkpoint_LTX2_Config)
    assert "cannot tell which LTX-2 generation" in str(rejection)


# --- Gemma-4 encoder folder ---------------------------------------------------------------------


def _gemma4_config(hidden_size: int = 3840, num_layers: int = 48) -> dict:
    return {
        "architectures": ["Gemma4UnifiedForCausalLM"],
        "model_type": "gemma4_unified_text",
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
    }


def _write_gemma4(folder: Path, weight_names: list[str], config: dict | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config or _gemma4_config()))
    (folder / "tokenizer.json").write_text("{}")
    for name in weight_names:
        save_file({"model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2)}, folder / name)


def test_a_root_level_gemma4_folder_is_the_encoder_not_a_generic_text_llm(tmp_path: Path) -> None:
    _write_gemma4(tmp_path, ["gemma4-12b-ltx-v1_int8_convrot.safetensors"])
    result = _probe(tmp_path)
    assert isinstance(result.config, Gemma4Encoder_Gemma4Encoder_LTX2_Config)
    assert result.config.subfolder == ""
    assert result.config.weight_file == "gemma4-12b-ltx-v1_int8_convrot.safetensors"


def test_files_one_directory_down_are_found_and_the_bf16_weight_is_preferred(tmp_path: Path) -> None:
    _write_gemma4(
        tmp_path / "gemma4-12b-ltx-v1",
        ["gemma4-12b-ltx-v1_int8_convrot.safetensors", "gemma4-12b-ltx-v1_bf16.safetensors"],
    )
    config = _probe(tmp_path).config
    assert isinstance(config, Gemma4Encoder_Gemma4Encoder_LTX2_Config)
    assert config.subfolder == "gemma4-12b-ltx-v1"
    assert config.weight_file == "gemma4-12b-ltx-v1_bf16.safetensors"


def test_a_gemma4_of_another_size_cannot_feed_the_connectors(tmp_path: Path) -> None:
    _write_gemma4(tmp_path, ["te.safetensors"], config=_gemma4_config(hidden_size=2560, num_layers=32))
    rejection = _rejection(tmp_path, Gemma4Encoder_Gemma4Encoder_LTX2_Config)
    assert "hidden_size 3840" in str(rejection)


# --- key maps onto the diffusers classes --------------------------------------------------------


def test_the_video_vae_map_extends_diffusers_by_the_fourth_decoder_stage() -> None:
    converted = cc.convert_video_vae_keys(
        {
            "vae.decoder.up_blocks.7.conv.conv.weight": 1,
            "decoder.up_blocks.8.res_blocks.3.conv2.conv.bias": 2,
            "encoder.down_blocks.8.res_blocks.0.norm1.weight": 3,
            "per_channel_statistics.std-of-means": 4,
            "per_channel_statistics.channel": 5,
        }
    )
    assert converted == {
        "decoder.up_blocks.3.upsamplers.0.conv.conv.weight": 1,
        "decoder.up_blocks.3.resnets.3.conv2.conv.bias": 2,
        "encoder.mid_block.resnets.0.norm1.weight": 3,
        "latents_std": 4,
    }


def test_the_connector_map_joins_three_files_and_drops_what_is_not_a_connector() -> None:
    converted = cc.convert_connector_keys(
        {
            f"{T}video_embeddings_connector.transformer_1d_blocks.2.attn1.q_norm.weight": 1,
            f"{T}audio_embeddings_connector.learnable_registers": 2,
            "text_embedding_projection.audio_aggregate_embed.bias": 3,
            f"{T}transformer_blocks.0.attn1.to_q.weight": 5,
        }
    )
    assert converted == {
        "video_connector.transformer_blocks.2.attn1.norm_q.weight": 1,
        "audio_connector.learnable_registers": 2,
        "audio_text_proj_in.bias": 3,
    }


def test_the_vocoder_map_reaches_both_generators() -> None:
    converted = cc.convert_vocoder_keys(
        {
            "vocoder.vocoder.conv_pre.weight": 1,
            "vocoder.bwe_generator.resblocks.1.acts1.0.downsample.lowpass.filter": 2,
            "vocoder.vocoder.ups.3.bias": 3,
            "vocoder.mel_stft.mel_basis": 4,
        }
    )
    assert converted == {
        "vocoder.conv_in.weight": 1,
        "bwe_generator.resnets.1.acts1.0.downsample.filter": 2,
        "vocoder.upsamplers.3.bias": 3,
        "mel_stft.mel_basis": 4,
    }


def test_the_transformer_finish_renames_only_the_prompt_adaln_heads() -> None:
    sd, key_map = cc.finish_transformer_keys(
        {
            "prompt_adaln_single.linear.weight": 1,
            "audio_prompt_adaln_single.linear.bias": 2,
            "time_embed.linear.weight": 3,
        }
    )
    assert sd == {"prompt_adaln.linear.weight": 1, "audio_prompt_adaln.linear.bias": 2, "time_embed.linear.weight": 3}
    assert key_map == {
        "prompt_adaln_single.linear.weight": "prompt_adaln.linear.weight",
        "audio_prompt_adaln_single.linear.bias": "audio_prompt_adaln.linear.bias",
    }


# --- folder contracts the loader relies on -------------------------------------------------------


def test_a_duplicated_role_is_resolved_to_the_first_sorted_file(tmp_path: Path) -> None:
    _write_components(tmp_path)
    save_file(
        COMPONENT_TENSORS["ltx-2.5-22b_video_vae_bf16.safetensors"], tmp_path / "aaa-second-video-vae.safetensors"
    )
    config = _probe(tmp_path).config
    assert isinstance(config, Main_Diffusers_LTX2_Config)
    assert config.components[layout.ROLE_VIDEO_VAE] == "aaa-second-video-vae.safetensors"


def _direct(path: Path, overrides: dict):
    """Fields the factory only forwards for a fixed set of keys reach a config through a direct call
    (the record editor's path); the common fields still come from the factory."""
    mod = ModelOnDisk(path)
    return Main_Diffusers_LTX2_Config.from_model_on_disk(
        mod, {**ModelConfigFactory.build_common_fields(mod), **overrides}
    )


def test_the_components_only_override_wins(tmp_path: Path) -> None:
    _write_components(tmp_path)
    assert _direct(tmp_path, {"components_only": False}).components_only is False


def test_generation_and_components_overrides_are_honoured_rather_than_duplicated(tmp_path: Path) -> None:
    _write_components(tmp_path)
    config = _direct(tmp_path, {"generation": "2.5", "components": {"video_vae": "custom.safetensors"}})
    assert config.generation == "2.5"
    assert config.components == {"video_vae": "custom.safetensors"}


def test_a_folder_with_an_older_transformer_beside_2_5_components_is_refused_naming_it(tmp_path: Path) -> None:
    """Every dating file is checked at identification, so the odd file is named here rather than at
    the strict load of whichever transformer sorted first; and refused as invalid so the folder is
    not registered as an unknown model."""
    _write_components(tmp_path)
    save_file(
        _transformer_tensors(video_ff_bias=True),
        tmp_path / "zz-ltx-2.3-22b-dev.safetensors",
        metadata={"model_version": "2.3.0"},
    )
    result = _probe(tmp_path)
    assert result.config is None
    rejection = result.details[Main_Diffusers_LTX2_Config.__name__]
    assert isinstance(rejection, InvalidMatchError)
    assert "zz-ltx-2.3-22b-dev.safetensors is an LTX-2.3 file" in str(rejection)


def test_the_header_version_outranks_the_structure(tmp_path: Path) -> None:
    path = tmp_path / "ltx-2.5-22b-dev.safetensors"
    save_file(_transformer_tensors(), path, metadata={"model_version": "2.3.0"})
    result = _probe(path)
    assert result.config is None, "an invalid match is final: nothing is registered"
    assert isinstance(result.details[Main_Checkpoint_LTX2_Config.__name__], InvalidMatchError)


def test_a_folder_whose_only_video_vae_is_the_diffusion_decoder_says_so(tmp_path: Path) -> None:
    _write_components(tmp_path, names=[n for n in EXPECTED_ROLES if "video_vae" not in n])
    save_file(
        {"encoder.conv_in.conv.weight": torch.zeros(2, 2, 3, 3, 3), "decoder.diff_blocks.0.x": torch.zeros(2)},
        tmp_path / "diffusion_vae.safetensors",
    )
    rejection = _rejection(tmp_path, Main_Diffusers_LTX2_Config)
    assert "diffusion-decoder" in str(rejection)


# --- the text-LLM boundary -----------------------------------------------------------------------


def _write_sharded_gemma4(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({**_gemma4_config(), "architectures": ["Gemma4ForCausalLM"]}))
    (folder / "tokenizer.json").write_text("{}")
    (folder / "model.safetensors.index.json").write_text("{}")
    for i in (1, 2):
        save_file({f"model.layers.{i}.x": torch.zeros(2)}, folder / f"model-0000{i}-of-00002.safetensors")


def test_a_stock_sharded_gemma4_stays_a_generic_text_llm(tmp_path: Path) -> None:
    """Google's Gemma-4-12B is the same width and depth; only the LTX single-file layout is the encoder."""
    from invokeai.backend.model_manager.configs.text_llm import TextLLM_Diffusers_Config

    _write_sharded_gemma4(tmp_path)
    result = _probe(tmp_path)
    assert isinstance(result.config, TextLLM_Diffusers_Config)
    assert "sharded" in str(result.details[Gemma4Encoder_Gemma4Encoder_LTX2_Config.__name__])


def test_an_explicit_text_llm_override_keeps_the_ltx_folder_a_text_llm(tmp_path: Path) -> None:
    from invokeai.backend.model_manager.configs.text_llm import TextLLM_Diffusers_Config

    _write_gemma4(tmp_path, ["gemma4-12b-ltx-v1_int8_convrot.safetensors"])
    result = _probe(tmp_path, {"type": "text_llm"})
    assert isinstance(result.config, TextLLM_Diffusers_Config)


# --- the key maps against the real diffusers modules --------------------------------------------


def _model_keys(model: torch.nn.Module) -> set[str]:
    return {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}


def test_the_video_vae_map_is_a_bijection_onto_the_real_decoder_and_encoder() -> None:
    """A tiny VAE with the 2.5 block structure, its keys spelled the official way (flat block
    indices, ``res_blocks``, ``per_channel_statistics``) and converted back."""
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLLTX2Video

    tiny = {
        **cc.LTX2_5_VIDEO_VAE_CONFIG,
        "latent_channels": 4,
        "block_out_channels": (8, 16, 32, 32),
        "layers_per_block": (1, 1, 1, 1, 1),
        "decoder_block_out_channels": (8, 16, 16, 32),
        "decoder_layers_per_block": (1, 1, 1, 1, 1),
    }
    with init_empty_weights():
        model = AutoencoderKLLTX2Video(**tiny)
    # diffusers nested path -> official flat index, typed out from the official layout.
    flat = {
        "encoder.down_blocks.0.downsamplers.0": "encoder.down_blocks.1",
        "encoder.down_blocks.1.downsamplers.0": "encoder.down_blocks.3",
        "encoder.down_blocks.2.downsamplers.0": "encoder.down_blocks.5",
        "encoder.down_blocks.3.downsamplers.0": "encoder.down_blocks.7",
        "encoder.down_blocks.0": "encoder.down_blocks.0",
        "encoder.down_blocks.1": "encoder.down_blocks.2",
        "encoder.down_blocks.2": "encoder.down_blocks.4",
        "encoder.down_blocks.3": "encoder.down_blocks.6",
        "encoder.mid_block": "encoder.down_blocks.8",
        "decoder.up_blocks.0.upsamplers.0": "decoder.up_blocks.1",
        "decoder.up_blocks.1.upsamplers.0": "decoder.up_blocks.3",
        "decoder.up_blocks.2.upsamplers.0": "decoder.up_blocks.5",
        "decoder.up_blocks.3.upsamplers.0": "decoder.up_blocks.7",
        "decoder.up_blocks.0": "decoder.up_blocks.2",
        "decoder.up_blocks.1": "decoder.up_blocks.4",
        "decoder.up_blocks.2": "decoder.up_blocks.6",
        "decoder.up_blocks.3": "decoder.up_blocks.8",
        "decoder.mid_block": "decoder.up_blocks.0",
        "latents_mean": "per_channel_statistics.mean-of-means",
        "latents_std": "per_channel_statistics.std-of-means",
    }
    official: dict[str, int] = {}
    for key in _model_keys(model):
        name = key
        for new, old in sorted(flat.items(), key=lambda kv: -len(kv[0])):
            if name == new or name.startswith(new + "."):
                name = old + name[len(new) :]
                break
        official[name.replace(".resnets.", ".res_blocks.")] = 0
    official["per_channel_statistics.channel"] = 0
    assert set(cc.convert_video_vae_keys(official)) == _model_keys(model)


def test_the_connector_map_is_a_bijection_onto_the_real_connectors() -> None:
    from accelerate import init_empty_weights
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    tiny = {
        **cc.LTX2_5_CONNECTORS_CONFIG,
        "caption_channels": 4,
        "text_proj_in_factor": 2,
        "video_connector_num_attention_heads": 2,
        "video_connector_attention_head_dim": 4,
        "video_connector_num_layers": 2,
        "video_connector_num_learnable_registers": 2,
        "audio_connector_num_attention_heads": 2,
        "audio_connector_attention_head_dim": 2,
        "audio_connector_num_layers": 2,
        "audio_connector_num_learnable_registers": 2,
        "video_hidden_dim": 8,
        "audio_hidden_dim": 4,
    }
    with init_empty_weights():
        model = LTX2TextConnectors(**tiny)
    official: dict[str, int] = {}
    for key in _model_keys(model):
        if key.startswith("video_text_proj_in."):
            official["text_embedding_projection.video_aggregate_embed." + key[len("video_text_proj_in.") :]] = 0
        elif key.startswith("audio_text_proj_in."):
            official["text_embedding_projection.audio_aggregate_embed." + key[len("audio_text_proj_in.") :]] = 0
        else:
            branch, rest = key.split(".", 1)
            rest = (
                rest.replace("transformer_blocks.", "transformer_1d_blocks.")
                .replace("norm_q", "q_norm")
                .replace("norm_k", "k_norm")
            )
            official[f"{T}{branch.replace('_connector', '_embeddings_connector')}.{rest}"] = 0
    assert set(cc.convert_connector_keys(official)) == _model_keys(model)


def test_the_vocoder_map_is_a_bijection_onto_the_real_vocoder() -> None:
    from accelerate import init_empty_weights
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    with init_empty_weights():
        model = LTX2VocoderWithBWE(in_channels=4, hidden_channels=8, bwe_in_channels=4, bwe_hidden_channels=8)
    stored = _model_keys(model) - {"resampler.filter"}
    official: dict[str, int] = {}
    for key in stored:
        name = key
        for new, old in (
            (".conv_in.", ".conv_pre."),
            (".conv_out.", ".conv_post."),
            (".act_out.", ".act_post."),
            (".upsamplers.", ".ups."),
            (".resnets.", ".resblocks."),
            (".downsample.filter", ".downsample.lowpass.filter"),
        ):
            name = ("." + name).replace(new, old)[1:]
        official["vocoder." + name] = 0
    assert set(cc.convert_vocoder_keys(official)) == stored
