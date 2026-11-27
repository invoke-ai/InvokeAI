import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from invokeai.backend.wan.vae_decode import iter_wan_vae_decode_chunks


def _build_tiny_vae() -> AutoencoderKLWan:
    return AutoencoderKLWan(
        base_dim=2,
        z_dim=2,
        dim_mult=[1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[True],
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
        scale_factor_temporal=2,
        scale_factor_spatial=2,
    ).eval()


def test_iter_wan_vae_decode_chunks_matches_full_decode() -> None:
    torch.manual_seed(0)
    vae = _build_tiny_vae()
    latents = torch.randn(1, 2, 3, 4, 4)

    with torch.inference_mode():
        expected = vae.decode(latents, return_dict=False)[0]
        chunks = list(iter_wan_vae_decode_chunks(vae, latents))

    actual = torch.cat(chunks, dim=2)
    torch.testing.assert_close(actual, expected)
    assert len(chunks) == latents.shape[2]
    assert max(chunk.shape[2] for chunk in chunks) <= vae.config.scale_factor_temporal
