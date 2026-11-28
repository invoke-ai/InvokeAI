"""The dtype a SDNQ VAE loads in and computes in.

A FLUX autoencoder is broken in float16, which `precision: auto` picks on CUDA and MPS, so every FLUX.1 VAE path
avoids it, a standalone SDNQ one included. Other bases follow the configured dtype. Whatever the compute dtype, the
parameters the export leaves unpacked are cast to it: a group norm does not cast its input, so a norm in the dtype on
disk would meet an input of another one.
"""

import json
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.vae import (
    VAE_Diffusers_FLUX_Config,
    VAE_Diffusers_SD3_Config,
    VAE_Diffusers_SDXL_Config,
)
from invokeai.backend.model_manager.load.model_loaders.flux import FluxSDNQDiffusersModel
from invokeai.backend.model_manager.load.model_loaders.vae import VAELoader
from invokeai.backend.model_manager.taxonomy import SubModelType


def _packed_compute_dtypes(vae: torch.nn.Module) -> set[torch.dtype]:
    return {p.compute_dtype for p in vae.parameters() if hasattr(p, "compute_dtype")}


def _unpacked_float_dtypes(vae: torch.nn.Module) -> set[torch.dtype]:
    return {p.dtype for p in vae.parameters() if not hasattr(p, "compute_dtype") and p.is_floating_point()}


@pytest.fixture
def sdnq_vae_folder(tmp_path: Path) -> Path:
    """A tiny SDNQ VAE: the mid-block attention projections are packed int8 with a scale, the rest is stored in fp16."""
    vae = AutoencoderKL(
        block_out_channels=(32,),
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        norm_num_groups=8,
        latent_channels=4,
        layers_per_block=1,
    )
    vae.save_config(tmp_path)
    state_dict = {}
    for key, value in vae.state_dict().items():
        if value.ndim == 2 and key.endswith("weight"):
            state_dict[key] = torch.randint(-100, 100, value.shape, dtype=torch.int8)
            state_dict[key.removesuffix("weight") + "scale"] = torch.full((value.shape[0], 1), 0.01)
        else:
            state_dict[key] = value.to(torch.float16)
    save_file(state_dict, str(tmp_path / "diffusion_pytorch_model.safetensors"))
    (tmp_path / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq", "weights_dtype": "int8"}))
    return tmp_path


def _load(folder: Path, config_class: type, configured: torch.dtype) -> AutoencoderKL:
    loader = VAELoader.__new__(VAELoader)
    loader._torch_dtype = configured
    loader._torch_device = torch.device("cpu")
    return loader._load_model(config_class.model_construct(path=str(folder), name="vae"), SubModelType.VAE)


@pytest.mark.parametrize(
    "configured, expected",
    [
        pytest.param(torch.float16, torch.bfloat16, id="auto-on-cuda-becomes-bf16"),
        pytest.param(torch.bfloat16, torch.bfloat16, id="bf16-stays"),
        pytest.param(torch.float32, torch.float32, id="fp32-stays"),
    ],
)
def test_a_flux_sdnq_vae_never_dequantizes_to_float16(sdnq_vae_folder, configured, expected) -> None:
    vae = _load(sdnq_vae_folder, VAE_Diffusers_FLUX_Config, configured)

    assert _packed_compute_dtypes(vae) == {expected}
    assert _unpacked_float_dtypes(vae) == {expected}


@pytest.mark.parametrize(
    "config_class",
    [
        pytest.param(VAE_Diffusers_SDXL_Config, id="sdxl"),
        pytest.param(VAE_Diffusers_SD3_Config, id="sd3"),
    ],
)
def test_another_bases_sdnq_vae_follows_the_configured_dtype(sdnq_vae_folder, config_class) -> None:
    vae = _load(sdnq_vae_folder, config_class, torch.float16)

    assert _packed_compute_dtypes(vae) == {torch.float16}
    assert _unpacked_float_dtypes(vae) == {torch.float16}


def test_an_sdnq_vae_with_unpacked_tensors_stored_in_another_dtype_runs(sdnq_vae_folder) -> None:
    """Float32 compute against fp16 on disk: the group norms would otherwise get an fp32 input and fp16 parameters,
    which the CPU refuses. Float32 keeps the forward off the CPU's slow bf16 matmul."""
    vae = _load(sdnq_vae_folder, VAE_Diffusers_FLUX_Config, torch.float32)

    with torch.inference_mode():
        image = vae.decode(torch.zeros(1, 4, 8, 8), return_dict=False)[0]
        latents = vae.encode(torch.zeros(1, 3, 8, 8)).latent_dist.mode()

    assert image.dtype == latents.dtype == torch.float32


def test_a_flux_sdnq_pipelines_sdnq_vae_dequantizes_to_bf16_and_casts_the_rest(sdnq_vae_folder) -> None:
    loader = FluxSDNQDiffusersModel.__new__(FluxSDNQDiffusersModel)

    vae = loader._load_sdnq_vae(sdnq_vae_folder)

    assert _packed_compute_dtypes(vae) == {torch.bfloat16}
    assert _unpacked_float_dtypes(vae) == {torch.bfloat16}
