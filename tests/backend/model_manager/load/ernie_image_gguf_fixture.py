"""A real, tiny ERNIE-Image transformer GGUF, written the way the published releases pack theirs.

unsloth's builds keep the diffusers keys unchanged, store the norms and some biases as F32, six stem
weights as BF16 -- the 4-D patch convolution among them -- and quantize every other Linear, `text_proj`
included (vantagewithai's are laid out the same way, give or take one BF16 tensor). Their
`general.architecture` is borrowed from another model -- `wan` or `flux` -- which is why the writer
here is told to use one of those.
"""

from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.quants import dequantize, quantize

# Every Linear input a multiple of 32, the Q8_0 block size. head_dim is 32 and the RoPE axes sum to it,
# as the released (32, 48, 48) sums to 4096/32.
TINY_CONFIG = {
    "hidden_size": 64,
    "num_attention_heads": 2,
    "num_layers": 1,
    "ffn_hidden_size": 128,
    "in_channels": 32,
    "out_channels": 32,
    "patch_size": 1,
    "text_in_dim": 32,
    "rope_theta": 256,
    "rope_axes_dim": (8, 12, 12),
    "eps": 1e-06,
    "qk_layernorm": True,
}

# The published builds are K-quants, whose 256-element super-blocks the 64-wide config cannot hold: every quantized
# Linear input here is a multiple of 256 (`text_in_dim` differs from `hidden_size` so `text_proj` stays a Linear).
K_QUANT_CONFIG = {
    **TINY_CONFIG,
    "hidden_size": 256,
    "ffn_hidden_size": 512,
    "text_in_dim": 512,
    "rope_axes_dim": (32, 48, 48),
}

# gguf-py can read K-quants but not write them. A super-block's bytes are all valid codes, so random ones with sane
# fp16 super-scales make a real Q4_K tensor; what it means comes from gguf-py's reference reader, not from us.
_Q4_K_BLOCK_BYTES = 144  # d (fp16), dmin (fp16), 12 bytes of 6-bit scales and mins, 128 bytes of 4-bit codes
_Q4_K_SUPER_SCALE = np.float16(1e-3).tobytes()


def _random_q4_k(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    rows, cols = shape
    blocks = rng.integers(0, 256, size=(rows, cols // 256, _Q4_K_BLOCK_BYTES), dtype=np.uint8)
    blocks[..., 0:2] = np.frombuffer(_Q4_K_SUPER_SCALE, dtype=np.uint8)
    blocks[..., 2:4] = np.frombuffer(_Q4_K_SUPER_SCALE, dtype=np.uint8)
    return blocks.reshape(rows, -1)


# The six weights every unsloth release keeps in BF16 (read from the Q2_K to Q8_0 headers).
_BF16 = (
    "x_embedder.proj.weight",
    "time_embedding.linear_1.weight",
    "time_embedding.linear_2.weight",
    "adaLN_modulation.1.weight",
    "final_norm.linear.weight",
    "final_linear.weight",
)


def write_ernie_image_gguf(
    path: Path,
    *,
    architecture: str = "wan",
    qtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0,
    seed: int = 0,
    config: dict = TINY_CONFIG,
) -> dict[str, torch.Tensor]:
    """Write a randomly initialised tiny transformer to ``path``; return what each tensor dequantizes to."""
    from diffusers import ErnieImageTransformer2DModel

    torch.manual_seed(seed)
    model = ErnieImageTransformer2DModel(**config)
    # diffusers zero-initializes the adaLN modulation and the final projection (the DiT recipe), which zeroes the
    # output whatever the other weights hold, so a forward comparison would pass for any kernel. Fill them in.
    with torch.no_grad():
        for param in model.parameters():
            if not param.any():
                param.normal_(std=0.02)
    rng = np.random.default_rng(seed)

    writer = gguf.GGUFWriter(str(path), architecture)
    meant: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        data = tensor.detach().to(torch.float32).numpy()
        if name in _BF16:
            stored_as = gguf.GGMLQuantizationType.BF16
        elif data.ndim == 2:
            stored_as = qtype
        else:
            stored_as = gguf.GGMLQuantizationType.F32
        if stored_as is gguf.GGMLQuantizationType.F32:
            writer.add_tensor(name, data)
            meant[name] = torch.from_numpy(data.copy())
        else:
            raw = (
                _random_q4_k(data.shape, rng)
                if stored_as is gguf.GGMLQuantizationType.Q4_K
                else quantize(data, stored_as)
            )
            # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
            writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
            meant[name] = torch.from_numpy(dequantize(raw, stored_as)).reshape(tensor.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant
