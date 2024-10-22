from typing import Any, Dict


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
