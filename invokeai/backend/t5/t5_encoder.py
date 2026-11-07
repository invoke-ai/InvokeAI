"""Building a transformers ``T5EncoderModel`` from a bare state dict.

Shared by the single-file T5 loaders (GGUF and ComfyUI-style safetensors): neither format ships a
``config.json``, so the architecture is read from tensor shapes, and both need the feed-forward
workaround below because their ``wo`` weights may not be in the compute dtype.
"""

import types
from typing import Any

import torch
from transformers import T5Config

from invokeai.backend.quantization.fp8_scaled import FP8_WEIGHT_DTYPES


def infer_t5_encoder_config(sd: dict[str, Any]) -> T5Config:
    """Reconstruct a ``T5Config`` from a transformers-named T5 encoder state dict.

    This only supports the T5 v1.1 encoder family (e.g. google/t5-v1_1-xxl as used by FLUX.1 and SD 3).
    Dimensions that vary (vocab, d_model, layer count, head/ff sizes) are read from tensor shapes; the
    fixed architectural constants below (``relative_attention_max_distance``, ``layer_norm_epsilon``,
    gated-gelu activation) are the T5 v1.1 defaults and would need revisiting for other T5 variants.

    ``.shape`` on a ``GGMLTensor`` returns the dequantized (logical) shape, so quantized and unquantized
    tensors are read the same way.
    """
    num_layers = 0
    for key in sd.keys():
        if isinstance(key, str) and key.startswith("encoder.block."):
            try:
                num_layers = max(num_layers, int(key.split(".")[2]) + 1)
            except (IndexError, ValueError):
                pass

    shared = sd.get("shared.weight")
    if shared is None:
        shared = sd.get("encoder.embed_tokens.weight")
    if shared is None:
        raise ValueError("Could not find shared.weight (token embeddings) in T5 state dict")
    vocab_size, d_model = (int(x) for x in shared.shape)

    # Inner attention dim from q projection: nn.Linear(d_model, inner_dim) -> weight (inner_dim, d_model).
    q_weight = sd.get("encoder.block.0.layer.0.SelfAttention.q.weight")
    if q_weight is None:
        raise ValueError("Could not find SelfAttention.q.weight in T5 state dict")
    inner_dim = int(q_weight.shape[0])

    # Number of heads and buckets from the relative attention bias: nn.Embedding(num_buckets, num_heads).
    rel_bias = sd.get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    if rel_bias is None:
        raise ValueError("Could not find relative_attention_bias.weight in T5 state dict")
    num_buckets = int(rel_bias.shape[0])
    num_heads = int(rel_bias.shape[1])
    d_kv = inner_dim // num_heads

    # Feed-forward dim from the gated FFN: nn.Linear(d_model, d_ff) -> weight (d_ff, d_model).
    wi_0 = sd.get("encoder.block.0.layer.1.DenseReluDense.wi_0.weight")
    if wi_0 is None:
        raise ValueError("Could not find DenseReluDense.wi_0.weight in T5 state dict")
    d_ff = int(wi_0.shape[0])

    return T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        relative_attention_num_buckets=num_buckets,
        relative_attention_max_distance=128,  # T5 v1.1 default
        layer_norm_epsilon=1e-6,  # T5 v1.1 default
        feed_forward_proj="gated-gelu",
        is_gated_act=True,
        dense_act_fn="gelu_new",
        tie_word_embeddings=False,
        use_cache=False,
    )


def _casts_activations_to(weight: torch.Tensor) -> bool:
    """Whether the feed-forward may cast its activations to ``weight``'s dtype before ``wo``.

    Not to a quantized storage dtype: GGML's ``uint8`` would turn them into integers, and a ``wo`` kept in
    float8 would round them to fp8 before a layer that dequantizes its weight to the compute dtype anyway.
    """
    return weight.is_floating_point() and weight.dtype not in FP8_WEIGHT_DTYPES


def make_t5_feed_forward_safe(model: torch.nn.Module) -> None:
    """Work around a transformers T5 quirk that breaks ``wo`` weights stored in a non-compute dtype.

    ``T5DenseGatedActDense.forward`` casts its activations to ``self.wo.weight.dtype`` unless that
    dtype is ``torch.int8`` (a guard meant for bitsandbytes 8-bit quantization, see transformers
    issue #20287). GGML stores quantized weights as ``torch.uint8``, which slips past the ``int8``
    guard and causes the activations to be cast to an integer dtype, corrupting them; a ``wo`` kept in
    float8 would round them to fp8. We rebind the
    forward of each feed-forward module so the cast only happens for weights the activations may take
    the dtype of; such a ``wo`` is dequantized on-the-fly by the autocast Linear regardless.
    """

    def gated_forward(self, hidden_states):  # mirrors T5DenseGatedActDense.forward
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        if _casts_activations_to(self.wo.weight) and hidden_states.dtype != self.wo.weight.dtype:
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

    def act_forward(self, hidden_states):  # mirrors T5DenseActDense.forward
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if _casts_activations_to(self.wo.weight) and hidden_states.dtype != self.wo.weight.dtype:
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

    patched = 0
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "T5DenseGatedActDense":
            module.forward = types.MethodType(gated_forward, module)
            patched += 1
        elif cls_name == "T5DenseActDense":
            module.forward = types.MethodType(act_forward, module)
            patched += 1

    # Guard against a silent no-op: if transformers ever renames these feed-forward classes, the
    # match above would patch nothing and the dtype-cast bug would silently corrupt encoder output.
    # Fail loudly instead so the mismatch is caught at load time rather than in the generated images.
    if patched == 0:
        raise RuntimeError(
            "Failed to patch any T5 feed-forward modules (expected T5DenseGatedActDense / T5DenseActDense). "
            "The installed transformers version may have renamed these classes; the T5 encoder "
            "cannot be loaded safely without the wo-dtype workaround."
        )
