"""Materializing a quantized embedding weight, with the host RAM it needs reserved first.

A quantized encoder has to undo the quantization of exactly one tensor: ``embed_tokens.weight``.
``nn.Embedding`` indexes its weight directly, and neither ``GGMLTensor`` nor ``SDNQTensor`` can serve
an indexed read from packed storage -- so the lookup would fail at the user's first prompt, long
after the load that could have prevented it. It is also the weight least worth keeping packed: it is
the model's entire input representation, and one dequantized copy is the price of every lookup being
a plain gather.

The framework reserves a model's *file size* before a load (``ModelLoader._load_and_cache``), which
by construction does not cover unpacking a tensor on top of it. That gap is the reason this lives in
one place: 0.78 GB for a Qwen3-VL 4B, 1.24 GB for the 8B, and several times that at the peak.

Callers are the five single-file text-encoder loaders that reach a quantized embedding: Qwen3-VL
GGUF, Qwen3 GGUF, Qwen3 SDNQ, Z-Image diffusers SDNQ and Mistral GGUF. Not an inventory of every
such step in the tree -- ``gemma2_encoder.py`` dequantizes a 1.18 GB Gemma-2 embedding from a free
function with no cache to reserve against, and ``flux.py`` does the same for T5 and CLIP. Those are
other families, still unreserved, and folding them in means threading a cache through their call
chains.
"""

from typing import Any

import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.quantization.gguf.ggml_tensor import HOST_DEQUANT_PEAK_COPIES


def materialize_quantized_embedding(embedding: torch.nn.Module, *, ram_cache: ModelCache) -> bool:
    """Replace ``embedding.weight`` with a dequantized ``Parameter``, reserving the room first.

    Does nothing and returns False when the weight is already a plain tensor, so a caller can run it
    unconditionally over a model whose quantization it does not know.
    """
    weight: Any = embedding.weight
    dequantize = getattr(weight, "get_dequantized_tensor", None)
    if dequantize is None:
        return False

    # `.shape` is the dequantized shape on both wrapper types; `Tensor.numel()` is not overridden and
    # would report the packed element count instead -- for a 4-bit quant, roughly half of what is
    # about to be allocated.
    ram_cache.make_room(HOST_DEQUANT_PEAK_COPIES * weight.shape.numel() * weight.compute_dtype.itemsize)
    embedding.weight = torch.nn.Parameter(dequantize(), requires_grad=False)
    return True
