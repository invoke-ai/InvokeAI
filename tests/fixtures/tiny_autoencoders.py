"""TAE checkpoints at the published shapes: TAESD in both key layouts, TAEHV in its one.

The shapes come from the decoder modules, which `tests/backend/tiny_autoencoder/test_taesd.py` and
`test_taehv.py` pin against the real checkpoints' headers; those tests are what make these fixtures
faithful.
"""

import torch

from invokeai.backend.tiny_autoencoder.checkpoint import (
    TaehvLayout,
    TaesdLayout,
    build_taehv_decoder,
    build_taesd_decoder,
)

TAESD = TaesdLayout(latent_channels=4, arch="default", image_channels=3)
"""taesd / taesdxl."""
TAESD3 = TaesdLayout(latent_channels=16, arch="default", image_channels=3)
"""taesd3 / taef1."""
TAEF2 = TaesdLayout(latent_channels=32, arch="flux_2", image_channels=3)
TAEQI2_1 = TaesdLayout(latent_channels=64, arch="f16", image_channels=4)


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


def taehv_weights(layout: TaehvLayout, seed: int = 0, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """A published TAEHV checkpoint: the decoder with seeded random weights, next to an encoder."""
    torch.manual_seed(seed)
    decoder = build_taehv_decoder(layout)
    sd = {key: value.to(dtype).contiguous() for key, value in decoder.state_dict().items()}
    # A token encoder: identification and loading must drop it whatever it holds.
    sd["encoder.0.weight"] = torch.zeros(64, 3 * layout.patch_size**2, 3, 3, dtype=dtype)
    sd["encoder.0.bias"] = torch.zeros(64, dtype=dtype)
    return sd
