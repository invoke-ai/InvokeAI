from collections.abc import Iterator

import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify


def iter_wan_vae_decode_chunks(vae: AutoencoderKLWan, latents: torch.Tensor) -> Iterator[torch.Tensor]:
    """Decode one latent frame at a time while preserving Wan causal-convolution state."""
    _, _, num_frames, height, width = latents.shape
    tile_latent_min_height = vae.tile_sample_min_height // vae.spatial_compression_ratio
    tile_latent_min_width = vae.tile_sample_min_width // vae.spatial_compression_ratio
    if vae.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
        raise ValueError("Streaming Wan VAE decode does not support spatial tiling.")

    vae.clear_cache()
    try:
        hidden_states = vae.post_quant_conv(latents)
        for frame_index in range(num_frames):
            vae._conv_idx = [0]
            decoded = vae.decoder(
                hidden_states[:, :, frame_index : frame_index + 1],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
                first_chunk=frame_index == 0,
            )
            if vae.config.patch_size is not None:
                decoded = unpatchify(decoded, patch_size=vae.config.patch_size)
            yield decoded.clamp(-1.0, 1.0)
    finally:
        vae.clear_cache()
