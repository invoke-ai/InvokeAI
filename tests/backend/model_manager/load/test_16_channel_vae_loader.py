"""Loading the standalone 16-channel VAEs that have no loader of their own: SD3 single files, and
FLUX.1 folders.

A single-file SD3 VAE cannot go through `AutoencoderKL.from_single_file`: with no config beside the
weights, diffusers infers the model from the keys, finds no pipeline around a bare VAE and builds
SD 1.5's 4-channel autoencoder. So the loader builds SD3's own and reads either layout into it.

A FLUX.1 folder is read by `from_pretrained`, but not in the dtype the generic loader would pick.
"""

import json
from pathlib import Path

import accelerate
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_SD3_Config,
    VAE_Diffusers_FLUX_Config,
    VAE_Diffusers_SD3_Config,
)
from invokeai.backend.model_manager.load.model_loaders.vae import VAELoader
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.sd3.vae import SD3_VAE_SCALING_FACTOR, SD3_VAE_SHIFT_FACTOR, get_sd3_vae_diffusers_config

# `stabilityai/stable-diffusion-3.5-{medium,large}::vae/config.json`, which agree on every field. Written
# out rather than read from the fixtures: the point is an expectation that does not move with the code.
PUBLISHED_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 1.5305,
    "shift_factor": 0.0609,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}

# Tensor names and shapes of FLUX.1's `ae.safetensors`, from the identification fixture. That is the
# LDM layout of this very network, and the one a VAE extracted from an SD3 single-file checkpoint has.
LDM_LAYOUT_LISTING = (
    Path(__file__).parents[3]
    / "model_identification"
    / "stripped_models"
    / "4f7a0a7f-c823-494d-94d4-dbbec6ea6ef6"
    / "FLUX.1-schnell_ae.safetensors"
)


def _loader() -> VAELoader:
    loader = VAELoader.__new__(VAELoader)
    loader._torch_dtype = torch.bfloat16
    loader._torch_device = torch.device("cpu")
    return loader


def _numbered(shapes: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    """Each tensor filled with its own index, so where every one of them landed can be read back."""
    return {key: torch.full(shape, float(i), dtype=torch.bfloat16) for i, (key, shape) in enumerate(shapes.items())}


def _load_sd3(path: Path) -> AutoencoderKL:
    model = _loader()._load_model(VAE_Checkpoint_SD3_Config.model_construct(path=str(path), name="sd3-vae"))
    assert isinstance(model, AutoencoderKL)
    return model


def _assert_sd3_constants_and_fully_loaded(model: AutoencoderKL) -> None:
    assert (model.config.scaling_factor, model.config.shift_factor) == (SD3_VAE_SCALING_FACTOR, SD3_VAE_SHIFT_FACTOR)
    assert not any(p.is_meta for p in model.parameters())
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}


def test_the_constructed_config_matches_what_stability_publishes() -> None:
    with accelerate.init_empty_weights():
        built = dict(AutoencoderKL(**get_sd3_vae_diffusers_config()).config)

    def normalise(value: object) -> object:
        return list(value) if isinstance(value, (list, tuple)) else value

    differs = {
        field: (normalise(built.get(field, "<absent>")), normalise(expected))
        for field, expected in PUBLISHED_CONFIG.items()
        if normalise(built.get(field, "<absent>")) != normalise(expected)
    }
    assert not differs
    # A new constructor argument in a future diffusers, with a default that changes the network, would
    # pass the comparison above by not being in it. Underscored keys are diffusers' bookkeeping.
    unexpected = sorted(field for field in set(built) - set(PUBLISHED_CONFIG) if not field.startswith("_"))
    assert not unexpected, f"diffusers grew config fields this test does not know about: {unexpected}"


def test_a_diffusers_layout_file_loads_with_sd3_constants(tmp_path: Path) -> None:
    with accelerate.init_empty_weights():
        reference = AutoencoderKL(**get_sd3_vae_diffusers_config())
    weights = _numbered({key: list(t.shape) for key, t in reference.state_dict().items()})
    save_file(weights, tmp_path / "sd3_vae.safetensors")

    model = _load_sd3(tmp_path / "sd3_vae.safetensors")

    _assert_sd3_constants_and_fully_loaded(model)
    loaded = model.state_dict()
    assert all(torch.equal(loaded[key], tensor) for key, tensor in weights.items())


def test_an_ldm_layout_file_is_converted_and_loses_no_tensor(tmp_path: Path) -> None:
    listing = json.loads(LDM_LAYOUT_LISTING.read_text())
    # Every entry but the stripper's own metadata record describes a tensor.
    weights = _numbered({key: entry["shape"] for key, entry in listing.items() if "shape" in entry})
    assert "encoder.down.0.block.0.norm1.weight" in weights, "the listing must be in the LDM layout"
    save_file(weights, tmp_path / "sd3_vae.safetensors")

    model = _load_sd3(tmp_path / "sd3_vae.safetensors")

    _assert_sd3_constants_and_fully_loaded(model)
    # Renamed and, for the attention projections, reshaped -- but every tensor arrives exactly once.
    arrived = sorted(int(t.flatten()[0].item()) for t in model.state_dict().values())
    assert arrived == list(range(len(weights)))


def test_a_file_in_neither_layout_is_refused_by_name(tmp_path: Path) -> None:
    save_file(
        {"encoder.conv_in.weight": torch.zeros(128, 3, 3, 3), "decoder.conv_in.weight": torch.zeros(512, 16, 3, 3)},
        tmp_path / "odd_sd3_vae.safetensors",
    )

    with pytest.raises(ValueError, match=r"odd_sd3_vae\.safetensors is in neither the diffusers nor the LDM layout"):
        _load_sd3(tmp_path / "odd_sd3_vae.safetensors")


def _tiny_folder(path: Path, scaling_factor: float, shift_factor: float) -> Path:
    AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        norm_num_groups=4,
        latent_channels=16,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
    ).save_pretrained(path)
    return path


@pytest.mark.parametrize(
    "config_class, constants",
    [(VAE_Diffusers_FLUX_Config, (0.3611, 0.1159)), (VAE_Diffusers_SD3_Config, (1.5305, 0.0609))],
)
def test_a_standalone_vae_folder_loads_when_a_model_loader_asks_for_it_as_its_vae(
    tmp_path: Path, config_class: type, constants: tuple[float, float]
) -> None:
    """`flux_model_loader`, `z_image_model_loader` and `sd3_model_loader` all ask for `SubModelType.VAE`,
    which the generic diffusers loader used to reject on a folder that holds nothing but the VAE."""
    folder = _tiny_folder(tmp_path / "vae", *constants)
    config = config_class.model_construct(path=str(folder), name="vae")

    model = _loader()._load_model(config, SubModelType.VAE)

    assert isinstance(model, AutoencoderKL)
    assert (model.config.scaling_factor, model.config.shift_factor) == constants


def test_a_flux_vae_folder_is_not_loaded_in_float16(tmp_path: Path) -> None:
    """float16 is what `precision: auto` picks on CUDA and MPS, and the FLUX autoencoder is broken in it.
    Every other FLUX.1 VAE path already avoids it; the folder is the one that reaches the generic loader."""
    folder = _tiny_folder(tmp_path / "vae", 0.3611, 0.1159)
    loader = _loader()
    loader._torch_dtype = torch.float16

    model = loader._load_model(
        VAE_Diffusers_FLUX_Config.model_construct(path=str(folder), name="vae"), SubModelType.VAE
    )

    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
