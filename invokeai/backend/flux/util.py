# Initially pulled from https://github.com/black-forest-labs/flux

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from invokeai.backend.flux.model import FluxParams
from invokeai.backend.model_manager.taxonomy import AnyVariant, Flux2VariantType, FluxVariantType


@dataclass
class AutoEncoderParams:
    """The BFL description of the FLUX.1 autoencoder.

    Kept after the hand-written `AutoEncoder` module was replaced by diffusers' `AutoencoderKL`,
    because it is still the single source of truth for this VAE: `get_flux_vae_diffusers_config`
    derives the diffusers keyword arguments from these fields, and the PiD nodes read
    `scale_factor`/`shift_factor` off it to undo the FLUX latent scaling.
    """

    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


# Preferred resolutions for Kontext models to avoid tiling artifacts
# These are the specific resolutions the model was trained on
PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


_flux_max_seq_lengths: dict[AnyVariant, Literal[256, 512]] = {
    FluxVariantType.Dev: 512,
    FluxVariantType.DevFill: 512,
    FluxVariantType.Schnell: 256,
    Flux2VariantType.Klein4B: 512,
    Flux2VariantType.Klein9B: 512,
}


def get_flux_max_seq_length(variant: AnyVariant):
    try:
        return _flux_max_seq_lengths[variant]
    except KeyError:
        raise ValueError(f"Unknown variant for FLUX max seq len: {variant}")


_flux_ae_params = AutoEncoderParams(
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
)


def get_flux_ae_params() -> AutoEncoderParams:
    return _flux_ae_params


# The two values the BFL parameters do not carry, taken from the published diffusers config
# (`black-forest-labs/FLUX.1-{dev,schnell}::vae/config.json`, 774 bytes, identical in both):
#
#   sample_size        seeds `AutoencoderKL.tile_sample_min_size`. `AutoEncoderParams.resolution` is
#                      256, the training crop, and is a different quantity -- using it would make the
#                      default tile four times too small. Every call site that tiles sets the size
#                      explicitly through `scoped_vae_tiling`, so this is the value used only when
#                      nobody asks.
#   use_(post_)quant_conv
#                      False. The FLUX autoencoder has no quant convolutions at all; diffusers
#                      defaults to True, which would add two 1x1 convolutions the checkpoint cannot
#                      fill.
_FLUX_VAE_SAMPLE_SIZE = 1024


def is_flux_family_vae(vae: Any) -> bool:
    """Whether this VAE encodes into the FLUX.1 latent space.

    `isinstance(vae, AutoencoderKL)` used to answer this, because InvokeAI's own port of the BFL
    autoencoder was a distinct class. It no longer is: the FLUX.1 VAE *is* an `AutoencoderKL`, and so
    are SD 1.5, SDXL, SD 3.5, CogView 4 and Z-Image's. Three of those even share the 16-channel
    latent width, so the channel count alone does not separate them either.

    What does separate them is the normalisation, and that is also exactly what a caller of this
    depends on: `pid_upscale` undoes `scale * (raw - shift)` with the constants from
    `get_flux_ae_params()`, and a VAE that used different ones hands it a latent that means something
    else. So the test is on the constants themselves -- Z-Image passes, because it is the same
    autoencoder (`_name_or_path: "flux-dev"` in its published config); SD 3.5 does not, because its
    `scaling_factor` is 1.5305.
    """
    config = getattr(vae, "config", None)
    return config is not None and is_flux_family_vae_config(config)


def is_flux_family_vae_config(config: Mapping[str, Any]) -> bool:
    """`is_flux_family_vae` on the config alone, which is what identification has of a `vae/` folder.

    Identification and the nodes share this test so that a folder installed under `flux` is always
    one `flux_vae_encode` and `pid_upscale` accept.
    """
    params = get_flux_ae_params()
    return (
        config.get("latent_channels") == params.z_channels
        and config.get("scaling_factor") == params.scale_factor
        and config.get("shift_factor") == params.shift_factor
    )


def get_flux_vae_diffusers_config() -> dict[str, Any]:
    """The FLUX.1 autoencoder as `AutoencoderKL.__init__` keyword arguments.

    Derived from `_flux_ae_params` rather than transcribed from the published `vae/config.json`, so
    the two descriptions of this autoencoder cannot drift apart -- `AutoEncoderParams` is still the
    single source of truth, and is what the PiD nodes read for `scale_factor`/`shift_factor`.

    Fields left to diffusers' own defaults (`act_fn`, `norm_num_groups`, `force_upcast`,
    `mid_block_add_attention`, `latents_mean`, `latents_std`) match the published config today;
    `tests/backend/model_manager/load/test_flux_vae_loader.py::TestTheConstructedConfig` pins the
    fully constructed config against the published values, so a changed default in a future
    diffusers breaks a test rather than the model.
    """
    params = get_flux_ae_params()
    num_blocks = len(params.ch_mult)
    return {
        "in_channels": params.in_channels,
        "out_channels": params.out_ch,
        "down_block_types": ("DownEncoderBlock2D",) * num_blocks,
        "up_block_types": ("UpDecoderBlock2D",) * num_blocks,
        "block_out_channels": tuple(params.ch * mult for mult in params.ch_mult),
        "layers_per_block": params.num_res_blocks,
        "latent_channels": params.z_channels,
        "sample_size": _FLUX_VAE_SAMPLE_SIZE,
        "scaling_factor": params.scale_factor,
        "shift_factor": params.shift_factor,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
    }


_flux_transformer_params: dict[AnyVariant, FluxParams] = {
    FluxVariantType.Dev: FluxParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    ),
    FluxVariantType.Schnell: FluxParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    ),
    FluxVariantType.DevFill: FluxParams(
        in_channels=384,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    ),
    # Flux2 Klein 4B uses Qwen3 4B text encoder with stacked embeddings from layers [9, 18, 27]
    # The context_in_dim is 3 * hidden_size of Qwen3 (3 * 2560 = 7680)
    Flux2VariantType.Klein4B: FluxParams(
        in_channels=64,
        vec_in_dim=2560,  # Qwen3-4B hidden size (used for pooled output)
        context_in_dim=7680,  # 3 layers * 2560 = 7680 for Qwen3-4B
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    ),
    # Flux2 Klein 4B Base is the undistilled foundation model. It shares the same
    # architecture as Klein 4B (distilled) and reports guidance_embeds=False in its
    # HF transformer config - classical CFG (external negative pass) is the guidance mechanism.
    Flux2VariantType.Klein4BBase: FluxParams(
        in_channels=64,
        vec_in_dim=2560,  # Qwen3-4B hidden size (used for pooled output)
        context_in_dim=7680,  # 3 layers * 2560 = 7680 for Qwen3-4B
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    ),
    # Flux2 Klein 9B uses Qwen3 8B text encoder with stacked embeddings from layers [9, 18, 27]
    # The context_in_dim is 3 * hidden_size of Qwen3 (3 * 4096 = 12288)
    Flux2VariantType.Klein9B: FluxParams(
        in_channels=64,
        vec_in_dim=4096,  # Qwen3-8B hidden size (used for pooled output)
        context_in_dim=12288,  # 3 layers * 4096 = 12288 for Qwen3-8B
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    ),
    # Flux2 Klein 9B Base is the undistilled foundation model. It shares the same
    # architecture as Klein 9B (distilled) and reports guidance_embeds=False in its
    # HF transformer config - the guidance scalar is inert for all Klein variants.
    Flux2VariantType.Klein9BBase: FluxParams(
        in_channels=64,
        vec_in_dim=4096,  # Qwen3-8B hidden size (used for pooled output)
        context_in_dim=12288,  # 3 layers * 4096 = 12288 for Qwen3-8B
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    ),
}


def get_flux_transformers_params(variant: AnyVariant):
    try:
        return _flux_transformer_params[variant]
    except KeyError:
        raise ValueError(f"Unknown variant for FLUX transformer params: {variant}")
