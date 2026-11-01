"""Shared key handling for llama.cpp-converted decoder GGUF files.

Every text encoder InvokeAI loads from a llama.cpp GGUF is a decoder stack under the same tensor
naming: the text-only Qwen3 (Z-Image / FLUX.2 Klein), the Qwen3-VL language tower (Krea-2 /
Ideogram 4), and Mistral Small 3 (FLUX.2 [dev]). llama.cpp names them by role -- ``attn_q``,
``ffn_gate``, ``attn_norm`` -- not by architecture, so one converter serves all three and a naming
quirk found on one path is fixed for the others.

The names are where the three stop being alike. llama.cpp's converter reorders the attention query
and key rows of Mistral, not of Qwen3, so the Mistral loader also runs
``undo_llamacpp_qk_permutation`` after renaming.

``gemma2_encoder.py``'s converter stays separate on purpose: different target prefix
(``layers.N.*``, no ``model.``), Gemma's ``1 + weight`` norm convention folded in, and it raises on
an unmapped key instead of passing it through.
"""

import re
from collections.abc import Mapping
from typing import Any

import torch

from invokeai.backend.quantization.gguf.row_ops import take_rows

# llama.cpp block component -> transformers module path within a decoder layer.
_BLOCK_COMPONENTS = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    # QK normalization: present on Qwen3 and Ministral 3, absent on architectures without it.
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
}

# llama.cpp top-level tensor -> transformers key.
_TOP_LEVEL_KEYS = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",  # absent when embeddings are tied
}

_BLOCK_PATTERN = re.compile(r"^blk\.(\d+)\.(.+)$")


def is_llamacpp_decoder_state_dict(sd: Mapping[Any, Any]) -> bool:
    """True when the state dict uses llama.cpp's ``blk.N.*`` naming rather than the transformers one.

    ComfyUI-converted GGUFs keep the transformers naming, llama.cpp-converted ones do not, and both
    turn up in the wild for a text-only Qwen3 -- which is why that loader asks before converting. The
    Qwen3-VL path has no such choice to make: its identification requires the llama.cpp
    ``token_embd.weight``, so a file that reaches its loader is llama.cpp-named by construction.

    The caller's own layout comes after: a bare ``MistralModel`` has no ``model.`` prefix and strips
    it again. This function's job is the role-to-module-path mapping, not the enclosing container.
    """
    return any(isinstance(key, str) and key.startswith("blk.") for key in sd)


def convert_llamacpp_decoder_keys(sd: Mapping[Any, Any]) -> dict[Any, Any]:
    """Convert llama.cpp decoder GGUF keys to the transformers causal-LM layout.

    ``blk.N.attn_q.weight`` -> ``model.layers.N.self_attn.q_proj.weight``, ``token_embd.weight`` ->
    ``model.embed_tokens.weight``, and so on. Unrecognized block components keep their name under the
    layer prefix, and unrecognized top-level keys pass through untouched, so an unexpected tensor
    surfaces as a load-time complaint naming the real key instead of being dropped here.
    """
    # Keys are not constrained to `str`: the state dicts this runs on are typed `str | int` across
    # the config layer, and a non-string key has to survive rather than crash the conversion.
    out: dict[Any, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            out[key] = value
            continue

        match = _BLOCK_PATTERN.match(key)
        if match:
            layer_index, rest = match.groups()
            component, _, suffix = rest.partition(".")
            mapped = _BLOCK_COMPONENTS.get(component, component)
            target = f"model.layers.{layer_index}.{mapped}"
            out[f"{target}.{suffix}" if suffix else target] = value
            continue

        out[_TOP_LEVEL_KEYS.get(key, key)] = value
    return out


_QK_PROJECTION = re.compile(r"^model\.layers\.\d+\.self_attn\.([qk])_proj\.(?:weight|bias)$")


def undo_llamacpp_qk_permutation(sd: dict[Any, Any], n_head: int, n_head_kv: int) -> None:
    """Put the query and key rows of a llama.cpp GGUF back into transformers' order, in place.

    Runs on the transformers-named dict ``convert_llamacpp_decoder_keys`` returns, over every layer's
    ``q_proj`` / ``k_proj`` weight and bias. Queries are regrouped by ``n_head``, keys by
    ``n_head_kv``. Other tensors are left alone; GGML tensors stay packed. In place so each original
    is released as its replacement lands: the gather copies, and a copy of every layer's rows at once
    would add about 0.6 GB to a Q4_K Mistral Small 3 load.

    transformers' RoPE rotates dimension ``i`` of a head with ``i + head_dim / 2``; llama.cpp's
    "normal" RoPE rotates consecutive pairs. ``LlamaModel.permute`` in llama.cpp's converter
    interleaves the two halves of each head to match, and this is its inverse. Applied wrongly,
    nothing fails: queries and keys carry the same reordering, so only RoPE pairs the wrong
    dimensions, and the first position is unaffected.

    Which converters permute (llama.cpp ``conversion/`` at 339f324,
    https://github.com/ggml-org/llama.cpp/blob/339f3242c761f16c27e5ffcde1d35dcc31933a55/conversion/llama.py):

    - ``LlamaModel`` (``undo_permute = True``), registered for ``MistralForCausalLM`` among others,
      writes architecture ``llama``. ``mistral3.py``'s Mistral Small 3 / Ministral 3 text model
      subclasses it and writes ``mistral3``. Mistral Small 3 GGUFs in the wild carry either name.
    - Mistral's consolidated checkpoints (``--mistral-format``, ``mistral.py``) skip the permute
      because they are stored in llama.cpp's order already, so their GGUFs end up the same.
    - The Qwen2 / Qwen3 / Qwen3-VL and Gemma converters never permute: llama.cpp runs those models
      with half-split RoPE (NeoX, multi-section for Qwen3-VL), the pairing transformers stores. T5
      has no RoPE. See ``llama_model_rope_type`` in ``src/llama-model.cpp``.
    """
    for key in list(sd):
        match = _QK_PROJECTION.match(key) if isinstance(key, str) else None
        if match is None:
            continue
        heads = n_head if match.group(1) == "q" else n_head_kv
        rows = int(sd[key].shape[0])
        if heads <= 0 or rows % (2 * heads):
            raise ValueError(f"{key} has {rows} rows, which do not split into {heads} heads of rotary pairs")
        index = torch.arange(rows).reshape(heads, rows // heads // 2, 2).transpose(1, 2).reshape(rows)
        sd[key] = take_rows(sd[key], index)
