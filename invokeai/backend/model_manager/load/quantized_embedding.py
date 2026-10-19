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

# Measured, not derived: reading the dequantizers suggests 2 (an intermediate plus the result), and
# for the 8-bit formats that is what happens -- but the k-quant, 5-bit and 4-bit kernels unpack
# through full-size temporaries that coexist with both, and BF16 is not torch-compatible here
# (`TORCH_COMPATIBLE_QTYPES` is F32/F16 only) so it shuffles rather than passing through. Peak host
# RSS over one 256 MB bf16 dequantization, sampled during the call, one format per fresh process:
#
#     Q8_0 2.01x   Q4_0 3.04x   Q4_K 3.14x   Q2_K 3.32x   BF16 4.01x   Q5_K 4.01x
#     Q6_K 4.08x   Q3_K 4.16x   Q5_1 5.37x   Q5_0 5.38x
#     SDNQ: int8 2.00x   uint4 3.99x   int5 4.64x
#
# Q5_0/Q5_1 are the outliers and the reason for the headroom: `dequantize_blocks_Q5_0` expands the
# high bits through a full-size int32 shift table, which coexists with three uint8 temporaries, the
# fp16 product and the cast. The numpy fallback branch, despite going through float32, stays at ~4x.
#
# 6 clears the worst of them. It over-reserves for the 8-bit formats, which costs at most an
# eviction the cache can undo -- where under-reserving costs the swap this reservation exists to
# prevent, so the error belongs on this side. A per-format table would be exact and would silently
# rot the first time a new quant type lands, so the constant is deliberately blunt.
_PEAK_COPIES = 6


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
    ram_cache.make_room(_PEAK_COPIES * weight.shape.numel() * weight.compute_dtype.itemsize)
    embedding.weight = torch.nn.Parameter(dequantize(), requires_grad=False)
    return True
