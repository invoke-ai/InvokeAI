"""The dtype of the FLUX.1, FLUX.2, MOSS-Image, FLUX.3 and Qwen-Image-2.1 autoencoders under `precise_image_storage`.

A precise image only holds more than 8 bits if the decoder ran above bfloat16 precision, so with the setting on these
autoencoders load in float32, straight from the file's own values. With it off every dtype is what it was. Each case
loads a tiny real model, saved with float32 weights that bfloat16 cannot hold, through the loader that owns it, with
`precision: auto` on a CUDA card (float16 requested) so the setting is the only thing that moves the dtype.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers import AutoencoderKL, AutoencoderKLFlux2

from invokeai.backend.model_manager.configs.main import (
    Main_Diffusers_Flux3_Config,
    Main_Diffusers_FLUX_Config,
    Main_Diffusers_MossImage_Config,
    Main_Diffusers_QwenImage21_Config,
)
from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_Flux2_Config,
    VAE_Diffusers_Flux2_Config,
    VAE_Diffusers_FLUX_Config,
)
from invokeai.backend.model_manager.load.model_loaders import flux as flux_module
from invokeai.backend.model_manager.load.model_loaders.flux import Flux2VAEDiffusersLoader, Flux2VAELoader
from invokeai.backend.model_manager.load.model_loaders.flux3 import Flux3DiffusersModel
from invokeai.backend.model_manager.load.model_loaders.moss_image import MossImageDiffusersModel
from invokeai.backend.model_manager.load.model_loaders.qwen_image_2_1 import QwenImage21DiffusersModel
from invokeai.backend.model_manager.load.model_loaders.vae import VAELoader
from invokeai.backend.model_manager.taxonomy import SubModelType

precise_on_or_off = pytest.mark.parametrize("precise", [True, False], ids=["precise_on", "precise_off"])


def _loader(cls: type, precise: bool) -> Any:
    loader = object.__new__(cls)
    loader._app_config = SimpleNamespace(precise_image_storage=precise)
    loader._logger = logging.getLogger(__name__)
    loader._torch_device = torch.device("cpu")
    # `precision: auto` on CUDA and MPS.
    loader._torch_dtype = torch.float16
    return loader


def _expected_vae_dtype(precise: bool) -> torch.dtype:
    return torch.float32 if precise else torch.bfloat16


def _fine_values_in_float32(model: torch.nn.Module) -> None:
    """Weights a bfloat16 round trip changes, so a widened-after-rounding VAE cannot pass for a float32 one."""
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1 + 1.0)


def _assert_holds_the_files_weights(vae: torch.nn.Module, written: dict[str, torch.Tensor], precise: bool) -> None:
    loaded = vae.state_dict()
    for key, value in written.items():
        if precise:
            assert torch.equal(loaded[key], value), key
        else:
            assert torch.equal(loaded[key], value.to(torch.bfloat16)), key


def _tiny_flux2_vae() -> AutoencoderKLFlux2:
    model = AutoencoderKLFlux2(
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        mid_block_add_attention=False,
    )
    _fine_values_in_float32(model)
    return model


def _tiny_flux1_vae() -> AutoencoderKL:
    model = AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        norm_num_groups=4,
        latent_channels=16,
    )
    _fine_values_in_float32(model)
    return model


def _tiny_qwen21_vae() -> torch.nn.Module:
    from diffusers import AutoencoderKLQwenImage21

    model = AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        in_channels=4,
        out_channels=4,
        scale_factor_temporal=1,
        scale_factor_spatial=2,
    )
    _fine_values_in_float32(model)
    return model


@precise_on_or_off
def test_a_flux1_vae_folder_loads_in_float32_from_its_own_values(tmp_path: Path, precise: bool) -> None:
    written = _tiny_flux1_vae()
    written.save_pretrained(tmp_path / "vae")
    config = VAE_Diffusers_FLUX_Config.model_construct(path=str(tmp_path / "vae"), name="vae")

    vae = _loader(VAELoader, precise)._load_model(config, SubModelType.VAE)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)


@precise_on_or_off
def test_a_flux2_vae_folder_loads_in_float32_from_its_own_values(tmp_path: Path, precise: bool) -> None:
    written = _tiny_flux2_vae()
    written.save_pretrained(tmp_path)
    config = VAE_Diffusers_Flux2_Config.model_construct(path=str(tmp_path), name="flux2-vae")

    vae = _loader(Flux2VAEDiffusersLoader, precise)._load_model(config)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)


@precise_on_or_off
def test_a_flux2_vae_checkpoint_is_not_rounded_to_bfloat16_when_precise(
    monkeypatch: pytest.MonkeyPatch, precise: bool
) -> None:
    """The loader derives its geometry from the tensor shapes, so a small real network stands in for the released one."""
    model = AutoencoderKLFlux2(block_out_channels=(32, 64, 128, 128))
    _fine_values_in_float32(model)
    written = {key: value.clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(flux_module, "load_file", lambda _path: dict(written))
    config = VAE_Checkpoint_Flux2_Config.model_construct(path="flux2-vae.safetensors", name="flux2-vae")

    vae = _loader(Flux2VAELoader, precise)._load_model(config)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    # With the setting on the values are the file's; off, they pass through bfloat16 as before.
    _assert_holds_the_files_weights(vae, written, precise)


@precise_on_or_off
def test_a_moss_image_pipeline_vae_follows_the_setting(tmp_path: Path, precise: bool) -> None:
    written = _tiny_flux2_vae()
    written.save_pretrained(tmp_path / "vae")
    config = Main_Diffusers_MossImage_Config.model_construct(path=str(tmp_path), name="moss")

    vae = _loader(MossImageDiffusersModel, precise)._load_model(config, SubModelType.VAE)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)


@precise_on_or_off
def test_a_flux3_pipeline_vae_follows_the_setting_and_its_transformer_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, precise: bool
) -> None:
    from diffusers import Flux3Transformer2DModel

    from invokeai.backend.model_manager.load.model_loaders import flux3 as flux3_module

    written = _tiny_flux2_vae()
    written.save_pretrained(tmp_path / "vae")
    Flux3Transformer2DModel(
        in_channels=8,
        cond_in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=24,
        timestep_channels=16,
        axes_dims_rope=(4, 4, 4, 4),
    ).save_pretrained(tmp_path / "transformer")
    # What the transformer runs in does not depend on the setting: the device's bfloat16-safe dtype.
    monkeypatch.setattr(flux3_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        flux3_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.bfloat16)
    )
    config = Main_Diffusers_Flux3_Config.model_construct(path=str(tmp_path), name="flux3")
    loader = _loader(Flux3DiffusersModel, precise)

    vae = loader._load_model(config, SubModelType.VAE)
    transformer = loader._load_model(config, SubModelType.Transformer)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)
    # The weights diffusers keeps in float32 aside, the transformer is bfloat16 either way.
    assert torch.bfloat16 in {p.dtype for p in transformer.parameters()}


@precise_on_or_off
def test_a_qwen_image_21_pipeline_vae_follows_the_setting(tmp_path: Path, precise: bool) -> None:
    written = _tiny_qwen21_vae()
    written.save_pretrained(tmp_path / "vae")
    (tmp_path / "model_index.json").write_text(json.dumps({"vae": ["diffusers", "AutoencoderKLQwenImage21"]}))
    config = Main_Diffusers_QwenImage21_Config.model_construct(path=str(tmp_path), name="qwen")

    vae = _loader(QwenImage21DiffusersModel, precise)._load_model(config, SubModelType.VAE)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)


def test_a_configured_float32_stays_float32_with_the_setting_off(tmp_path: Path) -> None:
    _tiny_flux2_vae().save_pretrained(tmp_path)
    config = VAE_Diffusers_Flux2_Config.model_construct(path=str(tmp_path), name="flux2-vae")
    loader = _loader(Flux2VAEDiffusersLoader, precise=False)
    loader._torch_dtype = torch.float32

    vae = loader._load_model(config)

    assert {p.dtype for p in vae.parameters()} == {torch.float32}


def _pipeline_folder(root: Path, vae: torch.nn.Module, vae_class: str, transformer: torch.nn.Module) -> None:
    vae.save_pretrained(root / "vae")
    transformer.save_pretrained(root / "transformer")
    (root / "model_index.json").write_text(
        json.dumps({"vae": ["diffusers", vae_class], "transformer": ["diffusers", type(transformer).__name__]})
    )


@precise_on_or_off
def test_a_flux1_pipeline_vae_follows_the_setting_and_its_transformer_stays_bfloat16(
    tmp_path: Path, precise: bool
) -> None:
    from diffusers import FluxTransformer2DModel

    from invokeai.backend.model_manager.load.model_loaders.flux import FluxDiffusersModel

    written = _tiny_flux1_vae()
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        axes_dims_rope=[4, 4, 8],
    )
    _pipeline_folder(tmp_path, written, "AutoencoderKL", transformer)
    config = Main_Diffusers_FLUX_Config.model_construct(path=str(tmp_path), name="flux")
    loader = _loader(FluxDiffusersModel, precise)

    vae = loader._load_model(config, SubModelType.VAE)
    loaded_transformer = loader._load_model(config, SubModelType.Transformer)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)
    assert {p.dtype for p in loaded_transformer.parameters()} == {torch.bfloat16}


@precise_on_or_off
def test_a_flux2_pipeline_vae_follows_the_setting_and_its_transformer_stays_bfloat16(
    tmp_path: Path, precise: bool
) -> None:
    from diffusers import Flux2Transformer2DModel

    from invokeai.backend.model_manager.configs.main import Main_Diffusers_Flux2_Config
    from invokeai.backend.model_manager.load.model_loaders.flux import Flux2DiffusersModel

    written = _tiny_flux2_vae()
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    )
    _pipeline_folder(tmp_path, written, "AutoencoderKLFlux2", transformer)
    config = Main_Diffusers_Flux2_Config.model_construct(path=str(tmp_path), name="klein")
    loader = _loader(Flux2DiffusersModel, precise)

    vae = loader._load_model(config, SubModelType.VAE)
    loaded_transformer = loader._load_model(config, SubModelType.Transformer)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)
    assert {p.dtype for p in loaded_transformer.parameters()} == {torch.bfloat16}


@precise_on_or_off
def test_a_z_image_pipeline_vae_follows_the_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, precise: bool
) -> None:
    from invokeai.backend.model_manager.configs.main import Main_Diffusers_ZImage_Config
    from invokeai.backend.model_manager.load.model_loaders import z_image as z_image_module
    from invokeai.backend.model_manager.load.model_loaders.z_image import ZImageDiffusersModel

    written = _tiny_flux1_vae()
    written.save_pretrained(tmp_path / "vae")
    (tmp_path / "model_index.json").write_text(json.dumps({"vae": ["diffusers", "AutoencoderKL"]}))
    monkeypatch.setattr(z_image_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        z_image_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.bfloat16)
    )
    config = Main_Diffusers_ZImage_Config.model_construct(path=str(tmp_path), name="z-image")

    vae = _loader(ZImageDiffusersModel, precise)._load_model(config, SubModelType.VAE)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)


@precise_on_or_off
def test_a_standalone_qwen_image_21_vae_folder_follows_the_setting(tmp_path: Path, precise: bool) -> None:
    from invokeai.backend.model_manager.configs.vae import VAE_Diffusers_QwenImage21_Config

    written = _tiny_qwen21_vae()
    written.save_pretrained(tmp_path)
    config = VAE_Diffusers_QwenImage21_Config.model_construct(path=str(tmp_path), name="qwen21-vae")

    vae = _loader(VAELoader, precise)._load_model(config)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}
    _assert_holds_the_files_weights(vae, written.state_dict(), precise)


@precise_on_or_off
def test_a_standalone_qwen_image_21_vae_checkpoint_follows_the_setting(
    monkeypatch: pytest.MonkeyPatch, precise: bool
) -> None:
    """Off, the dtype `_wan_family_dtype` gave (bfloat16 for a float16 request). Read on the meta device at the released
    VAE's real shapes: only the dtype the loader casts the file to is in question."""
    from unittest.mock import MagicMock

    import accelerate
    from diffusers import AutoencoderKLQwenImage21

    from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_QwenImage21_Config
    from invokeai.backend.model_manager.load.model_loaders import vae as vae_module

    with accelerate.init_empty_weights():
        shapes = dict(AutoencoderKLQwenImage21().state_dict())
    monkeypatch.setattr(
        vae_module, "_read_checkpoint", lambda _path: {k: torch.empty_like(v) for k, v in shapes.items()}
    )
    loader = _loader(VAELoader, precise)
    loader._ram_cache = MagicMock()
    config = VAE_Checkpoint_QwenImage21_Config.model_construct(path="qwen21-vae.safetensors", name="qwen21-vae")

    vae = loader._load_model(config)

    assert {p.dtype for p in vae.parameters()} == {_expected_vae_dtype(precise)}


@precise_on_or_off
def test_a_z_image_pipeline_leaves_its_transformer_and_text_encoder_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, precise: bool
) -> None:
    from diffusers import ZImageTransformer2DModel
    from transformers import Qwen3Config, Qwen3Model

    from invokeai.backend.model_manager.configs.main import Main_Diffusers_ZImage_Config
    from invokeai.backend.model_manager.load.model_loaders import z_image as z_image_module
    from invokeai.backend.model_manager.load.model_loaders.z_image import ZImageDiffusersModel

    ZImageTransformer2DModel(
        dim=32,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=16,
        axes_dims=[4, 4, 8],
        axes_lens=[16, 16, 16],
    ).save_pretrained(tmp_path / "transformer")
    Qwen3Model(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )
    ).save_pretrained(tmp_path / "text_encoder")
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "text_encoder": ["transformers", "Qwen3Model"],
            }
        )
    )
    monkeypatch.setattr(z_image_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        z_image_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.bfloat16)
    )
    config = Main_Diffusers_ZImage_Config.model_construct(path=str(tmp_path), name="z-image")
    loader = _loader(ZImageDiffusersModel, precise)

    transformer = loader._load_model(config, SubModelType.Transformer)
    text_encoder = loader._load_model(config, SubModelType.TextEncoder)

    assert {p.dtype for p in transformer.parameters()} == {torch.bfloat16}
    assert {p.dtype for p in text_encoder.parameters()} == {torch.bfloat16}
