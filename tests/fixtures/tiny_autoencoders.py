"""TAESD decoder weights at the published shapes, in all three published key layouts.

The shapes come from the decoder modules, which `tests/backend/tiny_autoencoder/test_taesd.py` pins
against the real checkpoints' headers; that test is what makes these fixtures faithful.
"""

import torch

from invokeai.backend.tiny_autoencoder.checkpoint import TaesdLayout, build_taesd_decoder

TAESD = TaesdLayout(latent_channels=4, arch="default", image_channels=3)
"""taesd / taesdxl."""
TAESD3 = TaesdLayout(latent_channels=16, arch="default", image_channels=3)
"""taesd3 / taef1."""
TAEF2 = TaesdLayout(latent_channels=32, arch="flux_2", image_channels=3)
TAEQI2_1 = TaesdLayout(latent_channels=64, arch="f16", image_channels=4)
TAEW2_1 = TaesdLayout(latent_channels=16, arch="taehv", image_channels=3)

_TAEHV_LAST_TGROW = 19
"""The stride-2 `TGrow` after the last `MemBlock`, which the image decoder builds at stride 1."""


def native_weights(layout: TaesdLayout, seed: int = 0, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """A decoder-only checkpoint (layout A) with seeded random weights."""
    torch.manual_seed(seed)
    decoder = build_taesd_decoder(layout)
    return {key: value.to(dtype).contiguous() for key, value in decoder.state_dict().items()}


def diffusers_weights(native: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same decoder in the combined layout B: `decoder.layers.{N-1}.*`, next to an encoder."""
    combined = {}
    for key, value in native.items():
        index, rest = key.split(".", 1)
        combined[f"decoder.layers.{int(index) - 1}.{rest}"] = value
    # A token encoder: identification and loading must drop it whatever it holds.
    combined["encoder.layers.0.weight"] = torch.zeros(64, 3, 3, 3)
    combined["encoder.layers.0.bias"] = torch.zeros(64)
    return combined


def tinier_weights() -> dict[str, torch.Tensor]:
    """TAEQI2.1 "tinier" in layout B: its low-res Blocks are SuperBlocks (7x7 depthwise, 1x1, 1x1)."""
    combined = diffusers_weights(native_weights(TAEQI2_1, dtype=torch.float16))
    for index, channels in ((2, 256), (3, 256), (4, 256), (7, 128), (8, 128), (9, 128)):
        prefix = f"decoder.layers.{index}.conv."
        for key in [k for k in combined if k.startswith(prefix)]:
            del combined[key]
        combined[f"{prefix}0.weight"] = torch.zeros(channels, 1, 7, 7, dtype=torch.float16)
        combined[f"{prefix}1.weight"] = torch.zeros(channels * 2, channels, 1, 1, dtype=torch.float16)
        combined[f"{prefix}1.bias"] = torch.zeros(channels * 2, dtype=torch.float16)
        combined[f"{prefix}3.weight"] = torch.zeros(channels, channels * 2, 1, 1, dtype=torch.float16)
    return combined


def taehv_weights(
    latent_channels: int = 16, patch_size: int = 1, seed: int = 0, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """A TAEHV checkpoint in upstream's layout C: `decoder.N.*` with stride-2 `TGrow`s 13 and 19, next to an encoder.

    `TGrow` 19's weight stacks two groups of output channels, one per output frame; its first group is the negated
    last, so a test can tell by value which one a load kept. `patch_size` 2 gives taeh3/taew2_2 shapes.
    """
    torch.manual_seed(seed)
    decoder = build_taesd_decoder(TaesdLayout(latent_channels=latent_channels, arch="taehv", image_channels=3))
    combined = {f"decoder.{key}": value.to(dtype).contiguous() for key, value in decoder.state_dict().items()}
    last_timestep = combined[f"decoder.{_TAEHV_LAST_TGROW}.conv.weight"]
    combined[f"decoder.{_TAEHV_LAST_TGROW}.conv.weight"] = torch.cat([-last_timestep, last_timestep])
    out_channels = 3 * patch_size**2
    combined["decoder.22.weight"] = combined["decoder.22.weight"].repeat(patch_size**2, 1, 1, 1)[:out_channels]
    combined["decoder.22.bias"] = combined["decoder.22.bias"].repeat(patch_size**2)[:out_channels]
    # A token encoder: identification and loading must drop it whatever it holds.
    combined["encoder.0.weight"] = torch.zeros(64, out_channels, 3, 3, dtype=dtype)
    combined["encoder.0.bias"] = torch.zeros(64, dtype=dtype)
    return combined
