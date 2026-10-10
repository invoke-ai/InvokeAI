"""A real, tiny ERNIE-Image transformer GGUF, written the way the published releases pack theirs.

unsloth's builds keep the diffusers keys unchanged, store the norms and some biases as F32, six stem
weights as BF16 -- the 4-D patch convolution among them -- and quantize every other Linear, `text_proj`
included (vantagewithai's are laid out the same way, give or take one BF16 tensor). Their
`general.architecture` is borrowed from another model -- `wan` or `flux` -- which is why the writer
here is told to use one of those.
"""

from pathlib import Path

import gguf
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
) -> dict[str, torch.Tensor]:
    """Write a randomly initialised tiny transformer to ``path``; return what each tensor dequantizes to."""
    from diffusers import ErnieImageTransformer2DModel

    torch.manual_seed(seed)
    model = ErnieImageTransformer2DModel(**TINY_CONFIG)

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
        raw = data if stored_as is gguf.GGMLQuantizationType.F32 else quantize(data, stored_as)
        if stored_as is gguf.GGMLQuantizationType.F32:
            writer.add_tensor(name, raw)
            meant[name] = torch.from_numpy(raw.copy())
        else:
            # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
            writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
            meant[name] = torch.from_numpy(dequantize(raw, stored_as)).reshape(tensor.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant
