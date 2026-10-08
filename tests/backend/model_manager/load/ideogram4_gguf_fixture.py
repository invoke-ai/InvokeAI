"""A real, tiny Ideogram 4 transformer GGUF, written the way the published releases pack theirs.

Shared by the identification and the loader tests. The published GGUFs disagree on what they pack,
and the two layouts below are the two that occur:

- ``packs_block_linears`` -- molbal, leejet, stduhpf: the per-block Linears are quantized, and the
  norms, the embedding, every bias and the large stem Linears (``llm_cond_proj`` among them) are BF16.
- ``packs_every_weight`` -- rectangleworm: every weight is quantized, the RMSNorms and the two-row
  ``embed_image_indicator`` included; only the biases stay BF16.

BF16 matters as much as the quantized type: ``gguf_sd_loader`` wraps it as packed storage too, so a
fixture that wrote F32 for the unquantized tensors would take the ``TORCH_COMPATIBLE_QTYPES``
shortcut and miss the case every release ships.
"""

from pathlib import Path
from typing import Callable

import gguf
import torch
from gguf.quants import dequantize, quantize

from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer

# Every dimension a multiple of 32, the Q8_0 block size, so any weight can be quantized. head_dim is
# 32, and the MRoPE sections sum to half of it, as the real (24, 20, 20) does for head_dim 128.
TINY_CONFIG = Ideogram4Config(
    emb_dim=64,
    num_layers=1,
    num_heads=2,
    intermediate_size=128,
    adanln_dim=32,
    in_channels=32,
    llm_features_dim=64,
    mrope_section=(8, 4, 4),
)

Packs = Callable[[str, tuple[int, ...]], bool]


def packs_block_linears(name: str, shape: tuple[int, ...]) -> bool:
    return name.startswith("layers.") and len(shape) == 2


def packs_every_weight(name: str, shape: tuple[int, ...]) -> bool:
    return name.endswith(".weight")


def write_ideogram4_gguf(
    path: Path,
    packs: Packs,
    *,
    qtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Write a randomly initialised tiny transformer to ``path``, ``packs`` tensors stored as ``qtype``.

    Returns the float32 values a reader dequantizes each tensor to, i.e. what the file *means*, so a
    test can build the reference model from them rather than from the pre-quantization originals.
    No key/value metadata beyond what the writer insists on: the releases carry none.
    """
    torch.manual_seed(seed)
    model = Ideogram4Transformer(TINY_CONFIG)

    writer = gguf.GGUFWriter(str(path), "ideogram4")
    meant: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        data = tensor.detach().to(torch.float32).numpy()
        stored_as = qtype if packs(name, data.shape) else gguf.GGMLQuantizationType.BF16
        raw = quantize(data, stored_as)
        # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
        writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
        meant[name] = torch.from_numpy(dequantize(raw, stored_as)).reshape(tensor.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant
