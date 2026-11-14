"""The llama.cpp T5-encoder GGUF layout (``general.architecture = t5encoder``).

Two encoder families ship in it, converted by the same llama.cpp script and named alike: T5 v1.1 XXL
for FLUX and SD3 (``city96/t5-v1_1-xxl-encoder-gguf``) and UMT5-XXL for Wan 2.2
(``city96/umt5-xxl-encoder-gguf``). What tells them apart is structural: T5 v1.1 learns one relative
attention bias, in block 0, which every block reuses; UMT5 learns one per block. UMT5 also carries a
256k multilingual vocabulary where T5 v1.1 has 32k. A UMT5 file loaded as T5 v1.1 runs without an
error, on block 0's bias everywhere and the wrong tokenizer, and conditions on garbage.
"""

import re
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

if TYPE_CHECKING:
    # Identification imports this module; transformers is only needed once a model is built.
    from transformers import PretrainedConfig, PreTrainedTokenizerFast

_ATTENTION_COMPONENTS = {
    "attn_q": "SelfAttention.q",
    "attn_k": "SelfAttention.k",
    "attn_v": "SelfAttention.v",
    "attn_o": "SelfAttention.o",
    "attn_rel_b": "SelfAttention.relative_attention_bias",
    "attn_norm": "layer_norm",
}
_FEED_FORWARD_COMPONENTS = {
    "ffn_gate": "DenseReluDense.wi_0",
    "ffn_up": "DenseReluDense.wi_1",
    "ffn_down": "DenseReluDense.wo",
    "ffn_norm": "layer_norm",
}
_TOP_LEVEL_KEYS = {
    "token_embd.weight": "shared.weight",
    "enc.output_norm.weight": "encoder.final_layer_norm.weight",
}
# Every tensor one encoder block holds in a UMT5 file. A T5 v1.1 file has `attn_rel_b` in block 0 only.
BLOCK_COMPONENTS = (*_ATTENTION_COMPONENTS, *_FEED_FORWARD_COMPONENTS)

_BLOCK_PATTERN = re.compile(r"^enc\.blk\.(\d+)\.(.+)$")
_RELATIVE_BIAS_PATTERN = re.compile(r"^enc\.blk\.(\d+)\.attn_rel_b\.weight$")


def is_llamacpp_t5_encoder_state_dict(sd: Mapping[Any, Any]) -> bool:
    """True for llama.cpp T5-encoder naming: ``enc.blk.N.*`` blocks, which no decoder-only GGUF has."""
    return any(isinstance(key, str) and key.startswith("enc.blk.") for key in sd)


def has_per_layer_relative_bias(keys: Iterable[Any]) -> bool:
    """True when a block other than 0 has its own relative attention bias: UMT5, not T5 v1.1."""
    for key in keys:
        match = _RELATIVE_BIAS_PATTERN.match(key) if isinstance(key, str) else None
        if match is not None and int(match.group(1)) > 0:
            return True
    return False


def convert_llamacpp_t5_keys(sd: Mapping[Any, Any]) -> dict[Any, Any]:
    """Rename llama.cpp T5-encoder tensors to the transformers ``T5EncoderModel``/``UMT5EncoderModel`` layout.

    - ``token_embd.weight`` -> ``shared.weight``
    - ``enc.output_norm.weight`` -> ``encoder.final_layer_norm.weight``
    - ``enc.blk.N.attn_{q,k,v,o,rel_b,norm}`` -> ``encoder.block.N.layer.0.{SelfAttention.*,layer_norm}``
    - ``enc.blk.N.ffn_{gate,up,down,norm}`` -> ``encoder.block.N.layer.1.{DenseReluDense.wi_0,wi_1,wo,layer_norm}``

    ``ffn_gate`` is the activated branch (``wi_0``) and ``ffn_up`` the linear one (``wi_1``): the two are
    the same shape, so swapping them would load cleanly and encode wrongly. Unknown keys keep their
    name, so the loader's completeness check names them instead of their being dropped here.
    """
    out: dict[Any, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            out[key] = value
            continue
        if key in _TOP_LEVEL_KEYS:
            out[_TOP_LEVEL_KEYS[key]] = value
            continue
        match = _BLOCK_PATTERN.match(key)
        if match is None:
            out[key] = value
            continue
        block, rest = match.groups()
        component = rest.split(".", 1)[0]
        if component in _ATTENTION_COMPONENTS:
            out[f"encoder.block.{block}.layer.0.{_ATTENTION_COMPONENTS[component]}.weight"] = value
        elif component in _FEED_FORWARD_COMPONENTS:
            out[f"encoder.block.{block}.layer.1.{_FEED_FORWARD_COMPONENTS[component]}.weight"] = value
        else:
            out[key] = value
    return out


ConfigT = TypeVar("ConfigT", bound="PretrainedConfig")


def infer_t5_encoder_config(sd: Mapping[Any, Any], config_class: type[ConfigT]) -> ConfigT:
    """Build a ``T5Config`` or ``UMT5Config`` from a converted (transformers-named) state dict.

    The sizes that vary come from tensor shapes; ``.shape`` on a ``GGMLTensor`` is the logical one,
    so packed and plain tensors read alike. The constants -- max relative distance 128, layer-norm
    epsilon 1e-6, gated GELU -- are shared by T5 v1.1 and UMT5 and would need revisiting for any
    other T5 variant. Both families read the bucket and head counts off block 0's bias.
    """
    num_layers = 0
    for key in sd:
        if isinstance(key, str) and key.startswith("encoder.block."):
            try:
                num_layers = max(num_layers, int(key.split(".")[2]) + 1)
            except (IndexError, ValueError):
                pass

    def shape(key: str, what: str) -> tuple[int, ...]:
        tensor = sd.get(key)
        if tensor is None:
            raise ValueError(f"Could not find {key} ({what}) in the T5 GGUF state dict")
        return tuple(int(x) for x in tensor.shape)

    vocab_size, d_model = shape("shared.weight", "token embeddings")
    # nn.Linear(d_model, inner_dim) -> weight (inner_dim, d_model).
    inner_dim = shape("encoder.block.0.layer.0.SelfAttention.q.weight", "attention query")[0]
    # nn.Embedding(num_buckets, num_heads).
    num_buckets, num_heads = shape(
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", "relative attention bias"
    )
    d_ff = shape("encoder.block.0.layer.1.DenseReluDense.wi_0.weight", "gated feed-forward")[0]

    return config_class(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=inner_dim // num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        relative_attention_num_buckets=num_buckets,
        relative_attention_max_distance=128,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu",
        is_gated_act=True,
        dense_act_fn="gelu_new",
        tie_word_embeddings=False,
        use_cache=False,
    )


def _cast_for_wo(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    weight = module.wo.weight
    if not isinstance(weight, GGMLTensor) and weight.is_floating_point() and hidden_states.dtype != weight.dtype:
        hidden_states = hidden_states.to(weight.dtype)
    return hidden_states


def _gated_forward(self, hidden_states):  # mirrors T5DenseGatedActDense.forward
    hidden_states = self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return self.wo(_cast_for_wo(self, hidden_states))


def _act_forward(self, hidden_states):  # mirrors T5DenseActDense.forward
    hidden_states = self.dropout(self.act(self.wi(hidden_states)))
    return self.wo(_cast_for_wo(self, hidden_states))


_FEED_FORWARD_FORWARDS = {
    "T5DenseGatedActDense": _gated_forward,
    "UMT5DenseGatedActDense": _gated_forward,
    "T5DenseActDense": _act_forward,
    "UMT5DenseActDense": _act_forward,
}


def make_feed_forward_gguf_safe(model: torch.nn.Module) -> None:
    """Stop transformers' T5/UMT5 feed-forward from casting its activations to a GGUF ``wo``'s storage dtype.

    Before ``wo`` the feed-forward casts the activations to ``wo.weight.dtype`` unless that is
    ``torch.int8`` (a guard for bitsandbytes, transformers issue #20287). A ``GGMLTensor`` reports
    its *storage* dtype: ``uint8`` for a quantized block, which slips past the guard and turns the
    activations into integers, and ``float16`` for an F16 file, after which the matmul fails on a
    float16 input against the weight dequantized to its compute dtype. The ``GGMLTensor`` dequantizes
    to that compute dtype itself, so the cast is dropped for it and kept for plain float weights.
    """
    patched = 0
    for module in model.modules():
        forward = _FEED_FORWARD_FORWARDS.get(module.__class__.__name__)
        if forward is not None:
            module.forward = types.MethodType(forward, module)
            patched += 1

    # A rename of these classes in transformers would make the patch a silent no-op and corrupt every
    # encoding; fail at load instead.
    if patched == 0:
        raise RuntimeError(
            "Failed to patch any T5 feed-forward modules (expected one of "
            f"{', '.join(_FEED_FORWARD_FORWARDS)}). The installed transformers version may have renamed "
            "these classes; the GGUF T5 encoder cannot be loaded safely without the wo-dtype workaround."
        )


# The tokenizer fields identification can see (`ModelOnDisk.metadata` omits arrays) and the loader reads.
TOKENIZER_SCALAR_FIELDS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.add_eos_token",
)
_TOKENIZER_ARRAY_FIELDS = ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type")


def load_t5_tokenizer_from_gguf(path: Path) -> "PreTrainedTokenizerFast":
    """Build the Unigram tokenizer a llama.cpp T5-encoder GGUF describes in its ``tokenizer.ggml.*`` fields.

    It reproduces the HuggingFace ``tokenizer.json`` transformers converts from the same SentencePiece
    model: Unigram over the GGUF's pieces and scores, runs of spaces collapsed to one, Metaspace with
    a prepended ``▁``, ``</s>`` appended, and control, unknown and user-defined pieces (``<pad>``,
    ``</s>``, ``<unk>``, the ``<extra_id_N>`` sentinels) matched whole before the model sees the text.
    transformers' own GGUF tokenizer path cannot stand in: for this layout it keeps runs of spaces,
    finds no unknown token, and raises on the first character outside the vocabulary.

    Runs of spaces are collapsed whatever ``remove_extra_whitespaces`` says: the ``tokenizer.json``
    UMT5 and Wan ship does so although their SentencePiece model does not, since transformers'
    converter adds the rule unconditionally. Raises ``ValueError`` naming a missing field.
    """
    from gguf import TokenType
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
    from tokenizers.models import Unigram
    from transformers import PreTrainedTokenizerFast

    from invokeai.backend.quantization.gguf.loaders import WrappedGGUFReader

    with WrappedGGUFReader(path) as reader:
        fields: dict[str, Any] = {}
        for name in (*TOKENIZER_SCALAR_FIELDS, *_TOKENIZER_ARRAY_FIELDS):
            field = reader.fields.get(name)
            if field is None:
                raise ValueError(f"{path.name} has no '{name}' field, so its tokenizer cannot be built")
            fields[name] = field.contents()

    if fields["tokenizer.ggml.model"] != "t5":
        raise ValueError(f"{path.name} declares a '{fields['tokenizer.ggml.model']}' tokenizer, not a T5 Unigram one")
    pieces: list[str] = fields["tokenizer.ggml.tokens"]
    scores: list[float] = fields["tokenizer.ggml.scores"]
    token_types: list[int] = fields["tokenizer.ggml.token_type"]
    if not len(pieces) == len(scores) == len(token_types):
        raise ValueError(f"{path.name}: the tokenizer's tokens, scores and token types differ in length")
    unknown_ids = [i for i, kind in enumerate(token_types) if kind == TokenType.UNKNOWN]
    if len(unknown_ids) != 1:
        raise ValueError(f"{path.name}: expected exactly one unknown token, found {len(unknown_ids)}")

    tokenizer = Tokenizer(Unigram(list(zip(pieces, scores, strict=True)), unk_id=unknown_ids[0], byte_fallback=False))
    tokenizer.normalizer = normalizers.Replace(Regex(" {2,}"), " ")
    prepend_scheme = "always" if fields["tokenizer.ggml.add_space_prefix"] else "never"
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme, split=True)
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme, split=True)
    eos_id = fields["tokenizer.ggml.eos_token_id"]
    eos = pieces[eos_id]
    if fields["tokenizer.ggml.add_eos_token"]:
        tokenizer.post_processor = processors.TemplateProcessing(
            single=["$A", eos], pair=["$A", eos, "$B", eos], special_tokens=[(eos, eos_id)]
        )
    whole = (TokenType.CONTROL, TokenType.UNKNOWN, TokenType.USER_DEFINED)
    tokenizer.add_special_tokens(
        [
            AddedToken(piece, special=True, normalized=False)
            for piece, kind in zip(pieces, token_types, strict=True)
            if kind in whole
        ]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=eos,
        pad_token=pieces[fields["tokenizer.ggml.padding_token_id"]],
        unk_token=pieces[unknown_ids[0]],
    )
