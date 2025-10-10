# Initially pulled from https://github.com/black-forest-labs/flux

from dataclasses import dataclass
from typing import Literal

from invokeai.backend.flux.model import FluxParams
from invokeai.backend.flux.modules.autoencoder import AutoEncoderParams
from invokeai.backend.model_manager.taxonomy import AnyVariant, FluxVariantType


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
}


def get_flux_transformers_params(variant: AnyVariant):
    try:
        return _flux_transformer_params[variant]
    except KeyError:
        raise ValueError(f"Unknown variant for FLUX transformer params: {variant}")
