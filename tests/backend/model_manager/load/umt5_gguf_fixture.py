"""A real, tiny llama.cpp UMT5 encoder GGUF, written from a dense ``UMT5EncoderModel``.

Shared by the Wan GGUF encoder loader's tests and by the FLUX T5 GGUF loader's, which must refuse it.
The file is what llama.cpp's converter writes for ``google/umt5-xxl``, scaled down: ``t5encoder``
metadata, a relative attention bias in every block, Q8_0 (or F32/F16/BF16) matrices and token embedding,
F32 norms and biases, and the ``tokenizer.ggml.*`` fields -- without ``unknown_token_id``, as in the
real file.

The llama.cpp names are spelled out here rather than taken from the loader's converter, and the
reference model is the dense one with each stored matrix run through gguf-py's own numpy dequantizer,
so nothing the tests compare against comes from the code under test.
"""

from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.quants import dequantize, quantize
from transformers import UMT5Config, UMT5EncoderModel

D_MODEL = 64  # Q8_0 packs blocks of 32 along a row, so every row length is a multiple of 32.
NUM_LAYERS = 3

# (piece, score, llama.cpp token type), in id order. Ids 0-3 are UMT5's: <pad>, </s>, <s>, <unk>.
_CONTROL, _UNKNOWN, _NORMAL, _USER_DEFINED = (
    gguf.TokenType.CONTROL,
    gguf.TokenType.UNKNOWN,
    gguf.TokenType.NORMAL,
    gguf.TokenType.USER_DEFINED,
)
VOCAB: list[tuple[str, float, int]] = [
    ("<pad>", 0.0, _CONTROL),
    ("</s>", 0.0, _CONTROL),
    ("<s>", 0.0, _CONTROL),
    ("<unk>", 0.0, _UNKNOWN),
    ("▁", -2.0, _NORMAL),
    ("▁a", -3.0, _NORMAL),
    ("▁cat", -4.0, _NORMAL),
    ("▁on", -4.0, _NORMAL),
    ("▁the", -4.0, _NORMAL),
    ("▁mat", -5.0, _NORMAL),
    *[(letter, -9.0, _NORMAL) for letter in "abcdefghijklmnopqrstuvwxyz"],
    ("<extra_id_0>", -1000.0, _USER_DEFINED),
]
PIECE_IDS = {piece: index for index, (piece, _, _) in enumerate(VOCAB)}
PAD_ID, EOS_ID, UNK_ID = 0, 1, 3


def tiny_umt5_config() -> UMT5Config:
    return UMT5Config(
        vocab_size=len(VOCAB),
        d_model=D_MODEL,
        d_kv=32,
        d_ff=128,
        num_layers=NUM_LAYERS,
        num_heads=2,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=128,
        feed_forward_proj="gated-gelu",
        tie_word_embeddings=False,
    )


def _llamacpp_names(num_layers: int) -> dict[str, str]:
    names = {"shared.weight": "token_embd.weight", "encoder.final_layer_norm.weight": "enc.output_norm.weight"}
    for block in range(num_layers):
        attention = f"encoder.block.{block}.layer.0"
        feed_forward = f"encoder.block.{block}.layer.1"
        names.update(
            {
                f"{attention}.SelfAttention.q.weight": f"enc.blk.{block}.attn_q.weight",
                f"{attention}.SelfAttention.k.weight": f"enc.blk.{block}.attn_k.weight",
                f"{attention}.SelfAttention.v.weight": f"enc.blk.{block}.attn_v.weight",
                f"{attention}.SelfAttention.o.weight": f"enc.blk.{block}.attn_o.weight",
                f"{attention}.SelfAttention.relative_attention_bias.weight": f"enc.blk.{block}.attn_rel_b.weight",
                f"{attention}.layer_norm.weight": f"enc.blk.{block}.attn_norm.weight",
                # The gated (activated) branch is llama.cpp's `ffn_gate`, the linear one its `ffn_up`.
                f"{feed_forward}.DenseReluDense.wi_0.weight": f"enc.blk.{block}.ffn_gate.weight",
                f"{feed_forward}.DenseReluDense.wi_1.weight": f"enc.blk.{block}.ffn_up.weight",
                f"{feed_forward}.DenseReluDense.wo.weight": f"enc.blk.{block}.ffn_down.weight",
                f"{feed_forward}.layer_norm.weight": f"enc.blk.{block}.ffn_norm.weight",
            }
        )
    return names


def write_tiny_umt5_gguf(
    path: Path,
    *,
    matrix_type: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0,
    with_tokenizer: bool = True,
    seed: int = 0,
) -> UMT5EncoderModel:
    """Write the GGUF, matrices stored as ``matrix_type``, and return the dense float32 model it encodes.

    The relative attention biases are scaled up from their initialisation so that each block's bias
    moves the output well past numerical noise: an encoder that ran every block on block 0's bias
    has to fail a comparison, not pass it within tolerance.
    """
    torch.manual_seed(seed)
    reference = UMT5EncoderModel(tiny_umt5_config()).eval()
    with torch.no_grad():
        for block in reference.encoder.block:
            block.layer[0].SelfAttention.relative_attention_bias.weight.mul_(8.0)

    writer = gguf.GGUFWriter(str(path), "t5encoder")
    writer.add_uint32("t5encoder.block_count", NUM_LAYERS)
    writer.add_uint32("t5encoder.embedding_length", D_MODEL)
    if with_tokenizer:
        writer.add_tokenizer_model("t5")
        writer.add_tokenizer_pre("default")
        writer.add_token_list([piece for piece, _, _ in VOCAB])
        writer.add_token_scores([score for _, score, _ in VOCAB])
        writer.add_token_types([kind for _, _, kind in VOCAB])
        writer.add_add_space_prefix(True)
        writer.add_remove_extra_whitespaces(False)
        writer.add_eos_token_id(EOS_ID)
        writer.add_pad_token_id(PAD_ID)
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(True)

    state_dict = reference.state_dict()
    with torch.no_grad():
        for hf_name, gguf_name in _llamacpp_names(NUM_LAYERS).items():
            tensor = state_dict[hf_name]
            data = tensor.numpy().astype(np.float32)
            if data.ndim == 2 and "relative_attention_bias" not in hf_name:
                packed = quantize(data, matrix_type)
                writer.add_tensor(gguf_name, packed, raw_shape=packed.shape, raw_dtype=matrix_type)
                tensor.copy_(torch.from_numpy(dequantize(packed, matrix_type).reshape(data.shape)))
            else:
                writer.add_tensor(gguf_name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    # `state_dict()` aliases the parameters, so the round-tripped values are already in place; the
    # encoder's `embed_tokens` is tied to `shared` and follows.
    return reference
