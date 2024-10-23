import math
import re
from typing import Any, Dict

from invokeai.backend.sd3.sd3_mmditx import ContextEmbedderConfig, Sd3MMDiTXParams


def is_sd3_checkpoint(sd: Dict[str, Any]) -> bool:
    """Is the state dict for an SD3 checkpoint like this one?:
    https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors

    Note that this checkpoint format contains both the VAE and the MMDiTX model.

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision.
    """
    # If all of the expected keys are present, then this is very likely a SD3 checkpoint.
    expected_keys = {
        # VAE decoder and encoder keys.
        "first_stage_model.decoder.conv_in.bias",
        "first_stage_model.decoder.conv_in.weight",
        "first_stage_model.encoder.conv_in.bias",
        "first_stage_model.encoder.conv_in.weight",
        # MMDiTX keys.
        "model.diffusion_model.final_layer.linear.bias",
        "model.diffusion_model.final_layer.linear.weight",
        "model.diffusion_model.joint_blocks.0.context_block.attn.ln_k.weight",
        "model.diffusion_model.joint_blocks.0.context_block.attn.ln_q.weight",
    }

    return expected_keys.issubset(sd.keys())


def infer_sd3_mmditx_params(sd: Dict[str, Any], prefix: str = "model.diffusion_model.") -> Sd3MMDiTXParams:
    """Infer the MMDiTX model parameters from the state dict.

    This logic is based on:
    https://github.com/Stability-AI/sd3.5/blob/19bf11c4e1e37324c5aa5a61f010d4127848a09c/sd3_impls.py#L68-L88
    """
    patch_size = sd[f"{prefix}x_embedder.proj.weight"].shape[2]
    depth = sd[f"{prefix}x_embedder.proj.weight"].shape[0] // 64
    num_patches = sd[f"{prefix}pos_embed"].shape[1]
    pos_embed_max_size = round(math.sqrt(num_patches))
    adm_in_channels = sd[f"{prefix}y_embedder.mlp.0.weight"].shape[1]
    context_shape = sd[f"{prefix}context_embedder.weight"].shape
    qk_norm = "rms" if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in sd else None
    x_block_self_attn_layers = sorted(
        [
            int(key.split(".x_block.attn2.ln_k.weight")[0].split(".")[-1])
            for key in list(filter(re.compile(".*.x_block.attn2.ln_k.weight").match, sd.keys()))
        ]
    )

    context_embedder_config: ContextEmbedderConfig = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": context_shape[1],
            "out_features": context_shape[0],
        },
    }
    return Sd3MMDiTXParams(
        patch_size=patch_size,
        depth=depth,
        num_patches=num_patches,
        pos_embed_max_size=pos_embed_max_size,
        adm_in_channels=adm_in_channels,
        context_shape=context_shape,
        qk_norm=qk_norm,
        x_block_self_attn_layers=x_block_self_attn_layers,
        context_embedder_config=context_embedder_config,
    )
