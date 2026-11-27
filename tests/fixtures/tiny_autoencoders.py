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
    # Identification and loading must drop the encoder whatever it holds; zeros at its real shapes.
    sd |= {key: torch.zeros(shape, dtype=dtype) for key, shape in taehv_encoder_shapes(layout).items()}
    return sd


def taehv_encoder_shapes(layout: TaehvLayout) -> dict[str, tuple[int, ...]]:
    """The 64 encoder tensors of a published TAEHV file, transcribed from the taew2_1, taew2_2, taeltx_2 and
    taeltx2_3 headers at madebyollin/taehv@62f7591: three 64-wide stages, each opened by a TPool 1x1 conv
    (`64 * stride` inputs) and a stride-2 conv, with the decoder's temporal strides in reverse."""
    shapes: dict[str, tuple[int, ...]] = {
        "encoder.0.weight": (64, 3 * layout.patch_size**2, 3, 3),
        "encoder.0.bias": (64,),
    }
    for pool, downscales in zip((2, 7, 12), reversed(layout.time_upscale), strict=True):
        shapes[f"encoder.{pool}.conv.weight"] = (64, 64 * (2 if downscales else 1), 1, 1)
        shapes[f"encoder.{pool + 1}.weight"] = (64, 64, 3, 3)
        for block in range(pool + 2, pool + 5):
            shapes |= {
                f"encoder.{block}.conv.0.weight": (64, 128, 3, 3),
                f"encoder.{block}.conv.0.bias": (64,),
                f"encoder.{block}.conv.2.weight": (64, 64, 3, 3),
                f"encoder.{block}.conv.2.bias": (64,),
                f"encoder.{block}.conv.4.weight": (64, 64, 3, 3),
                f"encoder.{block}.conv.4.bias": (64,),
            }
    shapes["encoder.17.weight"] = (layout.latent_channels, 64, 3, 3)
    shapes["encoder.17.bias"] = (layout.latent_channels,)
    return shapes
