# Copyright (c) 2026, The InvokeAI Development Team
"""Model loaders for the Mistral text encoder used by FLUX.2 [dev].

FLUX.2 [dev] uses BFL's 30-layer "cow-mistral3-small" distillation as its sole
text encoder. The diffusers release wraps it in the multimodal
``Mistral3ForConditionalGeneration``; standalone single-file safetensors
(Comfy-Org bf16/fp8/fp4) and GGUF redistributions (gguf-org cow variants) ship
only the text tower, which we load as an encoder-only ``MistralModel``.

ERNIE-Image encodes its prompts with a different member of the family: Ministral
3B (26 layers, hidden_size 3072, YaRN RoPE), which loads as ``Ministral3Model``.
The variant recorded at install time decides which of the two a single file
becomes — see ``MistralVariantType``.

Both single-file packagings embed the canonical Tekken tokenizer as a U8 tensor
named ``tekken_model`` (~19 MB). When ``mistral_common`` is installed we use
that embedded tokenizer directly; otherwise we fall back to fetching the
tokenizer from ``black-forest-labs/FLUX.2-dev`` via HuggingFace.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, Optional

import accelerate
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Ministral3Config,
    Ministral3Model,
    MistralCommonBackend,
    MistralConfig,
    MistralModel,
)

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.mistral_encoder import (
    MistralEncoder_Checkpoint_Config,
    MistralEncoder_Diffusers_Config,
    MistralEncoder_GGUF_Config,
)
from invokeai.backend.model_manager.load.load_default import (
    ModelLoader,
    _device_supports_fp8_storage,
    _model_declared_skip_patterns,
)
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.quantized_embedding import materialize_quantized_embedding
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    MistralVariantType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.util.llamacpp_keys import (
    convert_llamacpp_decoder_keys,
    is_llamacpp_decoder_state_dict,
)
from invokeai.backend.quantization.fp8_scaled import (
    TRANSFORMER_KEY_PREFIXES,
    attach_fp8_scales,
    cast_state_dict,
    expand_weight_scale,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    full_precision_hints_respected,
    is_scale_metadata_key,
    iter_weight_scale_pairs,
    parse_quantization_metadata,
    read_safetensors_metadata,
    reject_undecoded_mx_scale,
    should_keep_fp8_weights,
    split_fp8_scaled_layers,
    strip_layer_path_prefix,
    warn_on_unattached_scales,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.quantization.int8_convrot import (
    reject_int8_layers_a_plain_fold_cannot_decode,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.quantization.nvfp4 import (
    NVFP4Payload,
    install_nvfp4_layers,
    pop_nvfp4_layers,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.state_dict_loading import log_unexpected_keys

# Architecture constants for the 30-layer cow-mistral3-small distillation.
# Sourced from BFL's FLUX.2-dev ``text_encoder/config.json`` (text-model side of
# the Mistral3 multimodal stack) with the layer count adjusted to the cow depth.
# Hidden / head / KV / RoPE settings match upstream Mistral Small 3 because the
# cow distillation only changes depth (40 → 30), not width.
_COW_HIDDEN_SIZE = 5120
_COW_INTERMEDIATE_SIZE = 32768
_COW_NUM_HIDDEN_LAYERS = 30
_COW_NUM_ATTENTION_HEADS = 32
_COW_NUM_KV_HEADS = 8  # grouped-query attention
_COW_HEAD_DIM = 128
_COW_VOCAB_SIZE = 131072
_COW_MAX_POSITION_EMBEDDINGS = 131072
_COW_ROPE_THETA = 1000000000.0  # 1e9 — matches BFL FLUX.2-dev/text_encoder/config.json
_COW_RMS_NORM_EPS = 1e-5

# Ministral 3B, ERNIE-Image's text encoder, from the ``text_config`` of
# ``baidu/ERNIE-Image``'s ``text_encoder/config.json``. Only what the weights cannot reveal is
# pinned here. The context length is simply the released value: transformers scales YaRN by the
# explicit ``factor``, not by ``max_position_embeddings / original_max_position_embeddings``, so a
# different number would not move the sigmas -- it would only make transformers warn that the two
# disagree.
_MINISTRAL_3B_HEAD_DIM = 128
_MINISTRAL_3B_MAX_POSITION_EMBEDDINGS = 262144
_MINISTRAL_3B_RMS_NORM_EPS = 1e-5

# ERNIE-Image's released `tokenizer/tokenizer_config.json` truncates at this length. The
# mistral-common backend we build from the embedded vocab carries no limit of its own, so a long
# prompt would otherwise be encoded in full where the release would have cut it.
_MINISTRAL_3B_MAX_PROMPT_TOKENS = 2048

# HuggingFace fallback for the tokenizer when the model file doesn't embed
# tekken_model (older cow GGUFs without the embedded blob, or a diffusers folder
# without a sibling tokenizer/). We only need the BFL canonical source — upstream
# Mistral tokenizers (3.1 / 3.2) don't match BFL's chat template exactly.
_TOKENIZER_FALLBACK_SOURCE: tuple[str, str] = ("black-forest-labs/FLUX.2-dev", "tokenizer")

# The same fallback for Ministral 3B. BFL's tokenizer is the wrong vocab for it, and ERNIE-Image
# publishes its own. Only reached when the encoder file embeds no Tekken blob — the released one
# does, so this is the path for repackaged files.
_ERNIE_TOKENIZER_FALLBACK_SOURCE: tuple[str, str] = ("baidu/ERNIE-Image", "tokenizer")


class _TokenizerPolicy(NamedTuple):
    """How one encoder family's tokenizer is built, and where its fallback comes from.

    ``flux2_template`` selects the *form*: the raw-text adapter that splices FLUX.2's structural
    markers as single Tekken ids, or a plain HuggingFace tokenizer. The two are not
    interchangeable -- the adapter is no ``PreTrainedTokenizerBase`` and returns padded tensors
    rather than id lists -- so the choice is made once, here, per variant.
    """

    flux2_template: bool
    fallback_source: tuple[str, str]


# Exhaustive over `MistralVariantType` on purpose. Deriving this by exclusion (`is not Ministral3B`)
# would hand every future variant the FLUX.2 path by default, which either fails in the conditioning
# node or, on a FLUX.2-shaped one, conditions off-distribution with no error at all.
_TOKENIZER_POLICIES: dict[MistralVariantType, _TokenizerPolicy] = {
    MistralVariantType.Cow: _TokenizerPolicy(True, _TOKENIZER_FALLBACK_SOURCE),
    MistralVariantType.Mistral24B: _TokenizerPolicy(True, _TOKENIZER_FALLBACK_SOURCE),
    MistralVariantType.Ministral3B: _TokenizerPolicy(False, _ERNIE_TOKENIZER_FALLBACK_SOURCE),
}


def _tokenizer_policy(variant: MistralVariantType) -> _TokenizerPolicy:
    """The tokenizer policy for a variant, refusing loudly when a new one has none."""
    policy = _TOKENIZER_POLICIES.get(variant)
    if policy is None:
        raise NotImplementedError(
            f"No tokenizer policy for Mistral variant '{variant.value}'. Add one to "
            "_TOKENIZER_POLICIES: the FLUX.2 template adapter and a plain HuggingFace tokenizer "
            "encode differently, and defaulting to either silently degrades conditioning."
        )
    return policy


def _shape_of(tensor: Any) -> Any:
    """Tensor shape, read through GGUF's quantized wrapper when there is one."""
    return tensor.tensor_shape if isinstance(tensor, GGMLTensor) else tensor.shape


def _mistral_layer_indices(
    state_dict: dict[str, Any], packed_layers: Mapping[str, NVFP4Payload] | None = None
) -> set[int]:
    """Layer indices present in a ``model.layers.N.*`` state dict.

    Shared by both config builders so a fix to the scan cannot land in only one of them. What they
    do when it comes back empty stays with each: a deliberate policy difference, not an accident.

    ``packed_layers`` are counted as their weights would be. A loader takes nvfp4 layers out of the
    state dict to keep them packed, so scanning the dict alone sees only the layers that happened to
    stay -- and a packed encoder silently gets the fallback layer count.
    """
    indices: set[int] = set()
    paths = [*(key for key in state_dict if isinstance(key, str)), *(packed_layers or {})]
    for path in paths:
        if not path.startswith("model.layers.") or ".self_attn.q_proj" not in path:
            continue
        parts = path.split(".")
        if len(parts) > 2 and parts[2].isdigit():
            indices.add(int(parts[2]))
    return indices


def _build_mistral_config(
    state_dict: dict[str, Any],
    torch_dtype: torch.dtype,
    rope_theta: float | None = None,
    max_position_embeddings: int | None = None,
    packed_layers: Mapping[str, NVFP4Payload] | None = None,
) -> MistralConfig:
    """Build a transformers ``MistralConfig`` from a cow-mistral3-small state dict.

    Reads the bulk shapes from the state dict (vocab, hidden, heads, kv_heads,
    intermediate, layer count). ``rope_theta`` and ``max_position_embeddings`` can
    be passed explicitly when an out-of-band source is available (e.g. GGUF
    metadata); otherwise we fall back to cow defaults.

    ``packed_layers`` are the nvfp4 layers a loader took out of the state dict to keep them packed. They count
    as their weights would: read from the state dict alone, a packed encoder silently gets the cow layer count
    and head counts.
    """
    packed_layers = packed_layers or {}

    def output_rows(path: str) -> int | None:
        if path in packed_layers:
            return packed_layers[path].out_features
        weight = state_dict.get(f"{path}.weight")
        if weight is None:
            return None
        return int((weight.tensor_shape if isinstance(weight, GGMLTensor) else weight.shape)[0])

    # Vocab and hidden_size come from embed_tokens.
    embed = state_dict.get("model.embed_tokens.weight")
    if embed is None:
        raise ValueError("State dict does not contain model.embed_tokens.weight")
    embed_shape = _shape_of(embed)
    vocab_size, hidden_size = int(embed_shape[0]), int(embed_shape[1])

    layer_indices = _mistral_layer_indices(state_dict, packed_layers)
    num_hidden_layers = (max(layer_indices) + 1) if layer_indices else _COW_NUM_HIDDEN_LAYERS

    # Derive head counts from the first layer's attention projections.
    q_rows = output_rows("model.layers.0.self_attn.q_proj")
    k_rows = output_rows("model.layers.0.self_attn.k_proj")
    gate_rows = output_rows("model.layers.0.mlp.gate_proj")
    head_dim = _COW_HEAD_DIM
    if q_rows is not None and k_rows is not None and gate_rows is not None:
        num_attention_heads = q_rows // head_dim
        num_key_value_heads = k_rows // head_dim
        intermediate_size = gate_rows
    else:
        num_attention_heads = _COW_NUM_ATTENTION_HEADS
        num_key_value_heads = _COW_NUM_KV_HEADS
        intermediate_size = _COW_INTERMEDIATE_SIZE

    return MistralConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings or _COW_MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=_COW_RMS_NORM_EPS,
        tie_word_embeddings=False,
        rope_theta=rope_theta or _COW_ROPE_THETA,
        attention_bias=False,
        attention_dropout=0.0,
        dtype=torch_dtype,
    )


def _build_ministral3_config(
    state_dict: dict[str, Any],
    torch_dtype: torch.dtype,
    packed_layers: Mapping[str, NVFP4Payload] | None = None,
) -> Ministral3Config:
    """Build a ``Ministral3Config`` for ERNIE-Image's encoder from its state dict.

    The geometry is read from the weights. The RoPE settings are left to ``Ministral3Config``'s
    own defaults, which reproduce the released ``text_encoder/config.json`` exactly -- YaRN,
    theta 1e6, factor 16, original context 16384, ``llama_4_scaling_beta`` 0.1. Restating them
    here would be a second copy, free to drift from the implementation that consumes it; a test
    pins the defaults against the released values instead.

    Unlike ``_build_mistral_config`` this refuses to substitute nominal geometry for shapes it
    cannot find: that builder also serves GGUFs, whose projections it may legitimately miss,
    whereas here a wrong guess yields a model that loads cleanly and encodes garbage. Tensors are
    plain ``torch.Tensor``s -- this path never sees GGUF's quantized wrapper.
    """
    required = (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    )
    packed_layers = packed_layers or {}
    missing = [key for key in required if key not in state_dict]
    if missing:
        # A caller pops nvfp4 layers out of the state dict to keep them packed, so a geometry key
        # that is missing *because it is packed* is a different situation from one that was never
        # there. No packed Ministral build exists yet; say which of the two this is rather than
        # naming a key the file does have.
        packed = [key for key in missing if key.removesuffix(".weight") in packed_layers]
        if packed:
            raise ValueError(
                f"Ministral 3B state dict holds {', '.join(packed)} nvfp4-packed. This builder reads "
                "its geometry from dense weights, so a packed Ministral build is not supported."
            )
        raise ValueError(f"Ministral 3B state dict is missing {', '.join(missing)}")
    embed, q_proj, k_proj, gate_proj = (state_dict[key] for key in required)
    # Only the layer *count* has to account for weights a loader took out to keep packed; the
    # geometry above is required dense, per the refusal just made.
    layer_indices = _mistral_layer_indices(state_dict, packed_layers)

    # hidden_size is 3072 while the 32 heads are 128 wide, so head_dim cannot be inferred from the
    # width the way it can for the Mistral Small 3 encoders.
    head_dim = _MINISTRAL_3B_HEAD_DIM
    return Ministral3Config(
        vocab_size=int(_shape_of(embed)[0]),
        hidden_size=int(_shape_of(embed)[1]),
        intermediate_size=int(_shape_of(gate_proj)[0]),
        num_hidden_layers=max(layer_indices) + 1,
        num_attention_heads=int(_shape_of(q_proj)[0]) // head_dim,
        num_key_value_heads=int(_shape_of(k_proj)[0]) // head_dim,
        head_dim=head_dim,
        max_position_embeddings=_MINISTRAL_3B_MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=_MINISTRAL_3B_RMS_NORM_EPS,
        # The released config ties them. The file ships no `lm_head` and the encoder-only model
        # builds none, so this records the released intent rather than changing what loads.
        tie_word_embeddings=True,
        attention_dropout=0.0,
        dtype=torch_dtype,
    )


def _decode_gguf_field(field: Any) -> Any | None:
    """Decode a single GGUFReader field to a Python scalar, or ``None``."""
    if field is None:
        return None
    try:
        # GGUFReader exposes scalar fields under `.contents()` in recent gguf releases.
        # Fall back to parts decoding for older versions.
        if hasattr(field, "contents"):
            return field.contents()
    except Exception:
        pass
    import struct

    try:
        if field.types[0].name in ("FLOAT32",):
            return struct.unpack("<f", bytes(field.parts[-1]))[0]
        if field.types[0].name in ("FLOAT64",):
            return struct.unpack("<d", bytes(field.parts[-1]))[0]
        if field.types[0].name in ("UINT32", "UINT64", "INT32", "INT64", "UINT16"):
            return int.from_bytes(bytes(field.parts[-1]), "little")
    except Exception:
        return None
    return None


def _read_gguf_metadata_values(path: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    """Read several named metadata fields from a GGUF header in a single reader pass.

    Returns an empty dict if the file/header can't be read — callers must treat the
    result as best-effort and fall back to defaults. Reading multiple keys with one
    ``GGUFReader`` avoids re-parsing the (potentially large) header once per key.
    """
    try:
        import gguf

        reader = gguf.GGUFReader(path)
    except Exception:
        return {}
    return {key: _decode_gguf_field(reader.fields.get(key)) for key in keys}


def _read_gguf_metadata_value(path: Path, key: str) -> Any | None:
    """Read a single named field from a GGUF file's metadata header, or ``None``."""
    return _read_gguf_metadata_values(path, (key,)).get(key)


def _read_gguf_metadata_float(path: Path, key: str) -> float | None:
    value = _read_gguf_metadata_value(path, key)
    return float(value) if isinstance(value, (int, float)) else None


def _read_gguf_metadata_int(path: Path, key: str) -> int | None:
    value = _read_gguf_metadata_value(path, key)
    return int(value) if isinstance(value, (int, float)) else None


# Wrapper prefixes this loader strips from the state dict. Kept as a named constant because the
# `_quantization_metadata` header names its layers *before* the strip, so the hints have to be
# re-keyed with exactly the same list -- see `_load_text_encoder`. Restating it there would be a
# second copy free to drift.
MISTRAL_KEY_PREFIXES = ("text_encoder.", "language_model.")


def _strip_known_prefixes(sd: dict[str, Any]) -> dict[str, Any]:
    """Strip wrapper prefixes used by some FLUX.2 single-file redistributions.

    Comfy-Org and similar packagers sometimes prefix Mistral keys with
    ``text_encoder.`` or ``language_model.`` (the latter coming from the
    multimodal Mistral3 stack). We normalize everything to plain ``model.*``.
    """
    out: dict[str, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            out[key] = value
            continue
        new_key = key
        for prefix in MISTRAL_KEY_PREFIXES:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        out[new_key] = value
    return out


def _bare_mistral_path(path: str) -> str:
    """A CausalLM layer path as bare ``MistralModel`` names it (see ``_convert_for_bare_mistral_model``)."""
    return path.removeprefix("model.")


def _convert_for_bare_mistral_model(sd: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a `model.*` causal-LM state dict for direct loading into ``MistralModel``.

    Transformers' ``MistralForCausalLM`` exposes its decoder under ``model.`` and adds
    an ``lm_head``; bare ``MistralModel`` has the decoder modules at the top level
    (``embed_tokens``, ``layers``, ``norm``) and no LM head. Our state dicts come from
    GGUF / safetensors that target the CausalLM layout, so we strip the prefix and
    drop the LM head before calling ``MistralModel.load_state_dict``.
    """
    out: dict[str, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            out[key] = value
        elif not key.startswith("lm_head."):
            out[_bare_mistral_path(key)] = value
    return out


def _reinit_inv_freq(model: torch.nn.Module, config: Any, dtype: torch.dtype) -> None:
    """Re-initialize any RoPE ``inv_freq`` buffers still on the meta device.

    ``inv_freq`` is derived from config rather than stored in the checkpoint, so a
    meta buffer here must be recomputed before ``_materialize_remaining_meta_tensors``
    zero-fills it. NB: transformers 5.x moved ``rope_theta`` into the
    ``rope_parameters``/``rope_scaling`` dict, so reading ``config.rope_theta``
    directly raises ``AttributeError`` on the pinned version — fall back through both
    (matching the z_image loaders).
    """
    rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    rope_theta = rope_params.get("rope_theta") or getattr(config, "rope_theta", 1000000.0)
    for name, buffer in list(model.named_buffers()):
        if not (buffer.is_meta and name.endswith("inv_freq")):
            continue
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
        parent.register_buffer(parts[-1], inv_freq.to(dtype), persistent=False)


def _materialize_remaining_meta_tensors(model: torch.nn.Module, dtype: torch.dtype, logger) -> None:
    """Replace any parameters/buffers still on the meta device after load_state_dict.

    A meta tensor in the final model triggers ``Cannot copy out of meta tensor`` when
    the model cache moves the weights to the compute device. We can't recover the
    actual values for missing weights, but we can at least give the model a real
    tensor — norms get ones, everything else gets zeros — so the load completes and
    obvious errors are easier to debug than a low-level move failure.
    """
    materialized: list[str] = []
    for name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        # Any RMSNorm weight must init to ones, not zeros. Use the same broad substring
        # test as the checkpoint loader's missing-norm loop so both code paths agree —
        # e.g. `layers.N.input_layernorm.weight` is a norm. (For MistralModel the only
        # params containing "norm" are the layernorms and the final norm, so there are no
        # false positives.) The narrower `split('.')`/`endswith('_norm.weight')` test used
        # here previously missed `input_layernorm.weight`, zero-filling it on the GGUF path
        # where the missing-norm loop doesn't run.
        is_norm = "norm" in name
        new_tensor = torch.ones(param.shape, dtype=dtype) if is_norm else torch.zeros(param.shape, dtype=dtype)
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, torch.nn.Parameter(new_tensor, requires_grad=False))
        materialized.append(name)
    for name, buffer in list(model.named_buffers()):
        if not buffer.is_meta:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent.register_buffer(attr, torch.zeros(buffer.shape, dtype=dtype), persistent=False)
        materialized.append(f"{name} (buffer)")
    if materialized:
        logger.warning(
            f"Mistral encoder: materialized {len(materialized)} meta tensor(s) with default values "
            f"(this usually means a key was missing from the checkpoint). First 5: {materialized[:5]}"
        )


def _strip_final_norm_for_cow(model: torch.nn.Module, variant: MistralVariantType, logger: Any) -> None:
    """Replace ``model.norm`` with ``Identity`` for the 30-layer cow distillation.

    ComfyUI's reference implementation (``Mistral3_24BModel`` with ``num_layers=30``)
    sets ``final_norm=False``, so the hidden state at extraction index 30 is the
    raw output of layer 29 — NOT the final-RMSNorm'd version. Transformers'
    ``MistralModel`` always builds a final ``model.norm`` and applies it to
    ``hidden_states[-1]`` when ``output_hidden_states=True``, which produces
    off-distribution embeddings for the cow weights. Swap the norm out for an
    identity here so our extraction matches Comfy / BFL.

    The 40-layer Mistral Small 3 variant keeps the final norm. The decision keys on
    the persisted ``config.variant`` (single source of truth) rather than re-deriving
    it from the loaded layer count, so the config and loader can never disagree.
    """
    if variant is not MistralVariantType.Cow:
        return
    if not hasattr(model, "norm"):
        return
    model.norm = torch.nn.Identity()
    logger.info("Replaced model.norm with Identity for 30-layer cow Mistral (final_norm=False).")


def _warn_if_40_layer_mistral(variant: MistralVariantType, logger: Any) -> None:
    """Warn when a 40-layer Mistral Small 3 is loaded as a FLUX.2 [dev] text encoder.

    Architecturally, BFL's canonical ``black-forest-labs/FLUX.2-dev/text_encoder``
    (40-layer, fine-tuned by BFL) and upstream ``mistralai/Mistral-Small-3.x``
    GGUFs / safetensors (40-layer, base weights) are indistinguishable. In
    practice only the BFL bundle produces clean output — upstream Mistral 3.1/3.2
    at any quantization level gives visibly degraded prompt adherence because
    the joint attention was not trained against those weights.

    We accept both at probe time and emit this warning at load time so users who
    install a non-BFL 40-layer Mistral see the issue called out in the log
    instead of just getting weird images. Keys on ``config.variant`` (single source
    of truth), consistent with ``_strip_final_norm_for_cow``.
    """
    if variant is not MistralVariantType.Mistral24B:
        return
    logger.warning(
        "Loaded a 40-layer Mistral Small 3 text encoder. "
        "If this is NOT BFL's canonical FLUX.2-dev/text_encoder, expect degraded "
        "prompt adherence — upstream Mistral 3.1 / 3.2 weights (GGUFs from "
        "unsloth, gguf-org, etc.) are not what FLUX.2's joint attention was "
        "trained against. Recommended encoders: Comfy-Org bf16/fp8/fp4 or "
        "gguf-org cow-mistral3-small quants (all 30-layer cow distillation)."
    )


def _drop_quantization_metadata(sd: dict[str, Any], logger, target_dtype: torch.dtype | None = None) -> dict[str, Any]:
    """Dequantize Comfy-Org-style scaled FP8 weights and drop their metadata keys.

    nvfp4 layers must be taken out with ``pop_nvfp4_layers`` before this runs (see ``_load_text_encoder``): their
    block scales pair with a ``.weight`` just like an fp8 scale, and the fold below would stretch them over the
    packed weight.

    Comfy-Org's Mistral FLUX.2 redistributions store quantized weights alongside
    ``*.weight_scale`` (and occasionally ``*.input_scale``) tensors. We apply the
    scale in-place and remove the metadata so transformers can load the result.

    Dequantization runs in fp32 for numerical accuracy, but each result is cast
    back down to ``target_dtype`` immediately (when provided) so the transient peak
    is a single fp32 weight at a time rather than the whole dict held at fp32. For a
    24B fp8 encoder that difference is tens of GB — enough to OOM machines that can
    otherwise load the model.
    """
    dequantized = 0
    for weight_key, scale_key in list(iter_weight_scale_pairs(sd)):
        # The one scheme no pass upstream of this fold catches. `extract_fp8_scaled_layers` decodes
        # an MX grid, but it only runs on the branch that *keeps* fp8; this is the other one, taken
        # wherever fp8 cannot be held at all. Without this the exponent bytes are folded as linear
        # multipliers -- around 120-135 -- at the right shape and dtype, with nothing logged.
        reject_undecoded_mx_scale(weight_key[: -len(".weight")], sd[scale_key])
        weight = sd[weight_key].float()
        # `expand_weight_scale` rather than a local broadcast: a per-output-channel scale is 1-D of
        # length `out`, and `(out, in) * (out,)` aligns on the *last* axis, so it scales input
        # channels instead of output channels -- wrong on a square weight, a shape error otherwise.
        scale = expand_weight_scale(weight, sd[scale_key].float(), weight_key)
        result = weight * scale
        sd[weight_key] = result.to(target_dtype) if target_dtype is not None else result
        dequantized += 1
    if dequantized:
        logger.info(f"Dequantized {dequantized} Comfy-Org-style quantized weights")

    # `is_scale_metadata_key` covers both spellings of the weight and input scales; `.scale` stays
    # here because it is this producer's own spelling and is not a scaled-fp8 key.
    drop_keys = [
        k
        for k in sd.keys()
        if isinstance(k, str) and (is_scale_metadata_key(k) or k.endswith(".scale") or k.startswith("scaled_fp8"))
    ]
    for k in drop_keys:
        del sd[k]
    return sd


class _TekkenRawTextAdapter:
    """Expose a HuggingFace-tokenizer-like ``__call__`` over a ``mistral_common``
    Tekkenizer.

    FLUX.2 [dev]'s reference encoder pipeline (matching ComfyUI's
    ``Mistral3Tokenizer`` + ``Flux2Tokenizer``) feeds a pre-formatted raw string
    — ``[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]{prompt}[/INST]`` — straight into
    the BPE encoder rather than going through ``apply_chat_template``. The
    Tekken special tokens (``[SYSTEM_PROMPT]``, ``[/SYSTEM_PROMPT]``, ``[INST]``,
    ``[/INST]``) are part of the vocab so the encode call produces the right
    token IDs without any chat-template indirection.

    Padding defaults to **left** to match Comfy's ``pad_left=True`` — this keeps
    the meaningful tokens at the right edge of the sequence, where the
    transformer's joint attention was trained to consume them.
    """

    # Default special tokens for Mistral Small 3 Tekken vocab.
    _BOS_ID = 1  # <s>
    _PAD_ID = 11  # <pad>

    # FLUX.2 [dev]'s template structural markers. These are Tekken *special
    # tokens* (single ids), but mistral_common's raw ``Tekkenizer.encode`` runs
    # with ``SpecialTokenPolicy.IGNORE``, so it BPE-encodes them as literal text
    # (e.g. ``[SYSTEM_PROMPT]`` → ``['[','SY','STEM','_PRO','MP','T',']']``). We
    # resolve their ids up front and splice them in during ``_encode`` so the
    # sequence matches the reference PixtralProcessor byte-for-byte.
    _SPECIAL_MARKERS = ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[INST]", "[/INST]")

    def __init__(self, mistral_tokenizer: Any):
        self._tok = mistral_tokenizer
        self.pad_token_id = self._PAD_ID
        self._inner = getattr(getattr(mistral_tokenizer, "instruct_tokenizer", None), "tokenizer", None)
        self._special_ids = self._resolve_special_ids()

    def _resolve_special_ids(self) -> dict[str, int]:
        """Map each FLUX.2 structural marker to its Tekken special-token id.

        Returns an empty dict if the inner tokenizer doesn't expose a special
        vocab, in which case ``_encode`` falls back to the plain raw encode.
        """
        inner = self._inner
        if inner is None:
            return {}
        rev = getattr(inner, "_special_tokens_reverse_vocab", None)
        out: dict[str, int] = {}
        for marker in self._SPECIAL_MARKERS:
            mid: Any = None
            if isinstance(rev, dict):
                mid = rev.get(marker)
            if mid is None:
                try:
                    tok = inner.get_special_token(marker)
                    mid = getattr(tok, "id", tok)
                except Exception:
                    mid = None
            if isinstance(mid, int):
                out[marker] = mid
        return out

    def _encode(self, text: str) -> list[int]:
        """Encode the FLUX.2 template, emitting structural markers as their Tekken
        special-token ids (not literal BPE) so the ids match ComfyUI / the BFL
        PixtralProcessor. Adds BOS, no EOS.

        ``mistral_common`` exposes the BPE under
        ``MistralTokenizer.instruct_tokenizer.tokenizer`` (the inner Tekkenizer).
        """
        inner = self._inner
        if inner is None or not hasattr(inner, "encode"):
            # Older mistral-common releases expose .encode on the top-level wrapper.
            return list(self._tok.encode(text, add_bos=True, add_eos=False))
        if not self._special_ids:
            # No special vocab available — raw encode (markers become literal BPE).
            return list(inner.encode(text, bos=True, eos=False))

        # Split on the markers (longest-first so `[/SYSTEM_PROMPT]` wins over
        # `[SYSTEM_PROMPT]`), splice special ids, BPE-encode the plain segments.
        markers = sorted(self._special_ids, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(m) for m in markers) + ")"
        ids: list[int] = [self._BOS_ID]
        for part in re.split(pattern, text):
            if not part:
                continue
            special = self._special_ids.get(part)
            if special is not None:
                ids.append(special)
            else:
                ids.extend(inner.encode(part, bos=False, eos=False))
        return ids

    def __call__(
        self,
        text: str,
        *,
        padding: str | bool = "max_length",
        padding_side: str = "left",
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise NotImplementedError(f"_TekkenRawTextAdapter only supports return_tensors='pt' (got {return_tensors})")

        tokens = self._encode(text)
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        attention = [1] * len(tokens)

        if padding == "max_length":
            pad_needed = max_length - len(tokens)
            if pad_needed > 0:
                pad_tokens = [self.pad_token_id] * pad_needed
                pad_attn = [0] * pad_needed
                if padding_side == "left":
                    tokens = pad_tokens + tokens
                    attention = pad_attn + attention
                else:
                    tokens = tokens + pad_tokens
                    attention = attention + pad_attn

        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }


def _extract_tekken_bytes(model_path: Path) -> Optional[bytes]:
    """Return the bytes of the embedded ``tekken_model`` blob if the file has one.

    Both Comfy-Org's safetensors and gguf-org's cow GGUFs ship the canonical
    Tekken JSON inside a tensor named ``tekken_model``, but in incompatible
    layouts:

    - **Comfy safetensors**: U8 tensor, raw bytes, ``shape=(N,)`` — direct read.
    - **gguf-org cow GGUFs**: F16 tensor with one half-float per original byte
      (so the float values are 0..255 cast to fp16, and ``shape=(N,)``). We
      recover by casting each fp16 back to ``uint8``.

    Returns ``None`` if the file isn't a recognized container, doesn't embed
    the blob, or reading fails.
    """
    suffix = model_path.suffix.lower()
    try:
        if suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(model_path), framework="pt") as f:
                if "tekken_model" in f.keys():
                    return f.get_tensor("tekken_model").cpu().numpy().tobytes()
        elif suffix == ".gguf":
            import gguf
            import numpy as np

            reader = gguf.GGUFReader(str(model_path))
            for tensor in reader.tensors:
                if tensor.name != "tekken_model":
                    continue
                data = tensor.data
                if data.dtype == np.uint8:
                    return data.tobytes()
                # cow GGUFs (and friends) store one byte per fp16 value.
                return np.clip(np.rint(data.astype(np.float32)), 0, 255).astype(np.uint8).tobytes()
    except Exception:
        return None
    return None


def _try_load_embedded_tekken(model_path: Path, logger: Any, *, flux2_template: bool) -> Optional[AnyModel]:
    """Extract the embedded Tekken tokenizer in the form this encoder family needs.

    FLUX.2's encoders get the raw-text adapter, which splices the template's structural markers
    as single Tekken ids. An encoder with no chat template gets transformers' own mistral-common
    backend instead: the same ids, but a real ``PreTrainedTokenizerBase`` returning plain id
    lists, which is what the conditioning nodes accept.

    Returns ``None`` (so callers fall through to HF) if:
    - the file isn't a single-file container, or
    - no ``tekken_model`` blob is embedded, or
    - ``mistral_common`` isn't installed, or
    - the blob can't be parsed.
    """
    if not model_path.is_file():
        return None

    tekken_bytes = _extract_tekken_bytes(model_path)
    if tekken_bytes is None:
        return None

    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError:
        logger.info(
            "Found embedded Tekken tokenizer in %s but mistral_common is not installed. "
            "Run `pip install mistral-common` (or `uv add mistral-common`) to skip the "
            "HuggingFace tokenizer fetch.",
            model_path.name,
        )
        return None

    import shutil
    import tempfile

    # `MistralCommonBackend.from_pretrained` reads a *directory*, so stage the blob under the
    # name it looks for. Both routes parse the file before returning, so the copy does not have
    # to outlive this call.
    tmp_dir = Path(tempfile.mkdtemp(prefix="invokeai-tekken-"))
    try:
        (tmp_dir / _TEKKEN_FILENAME).write_bytes(tekken_bytes)
        if not flux2_template:
            return _tekken_backend_from_dir(tmp_dir, f"the blob embedded in {model_path.name}", logger)
        mistral_tok = MistralTokenizer.from_file(str(tmp_dir / _TEKKEN_FILENAME))
    except Exception as e:
        logger.warning(
            f"Failed to load embedded Tekken tokenizer from {model_path.name}: {type(e).__name__}: {e}. "
            "Falling back to the HuggingFace tokenizer."
        )
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Loaded embedded Tekken tokenizer from {model_path.name}")
    return _TekkenRawTextAdapter(mistral_tok)


# Filename of the standalone Tekken vocab shipped by official mistralai downloads and by the
# BFL-style standalone-encoder layouts these loaders accept.
_TEKKEN_FILENAME = "tekken.json"


def _tekken_adapter_from_file(tekken_path: Path, description: str, logger: Any) -> Optional[AnyModel]:
    """Load a Tekken vocab file via ``mistral_common`` and wrap it. ``None`` on any failure."""
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError:
        logger.info(
            f"Found a Tekken tokenizer in {description} but mistral_common is not installed. "
            "Run `pip install mistral-common` (or `uv add mistral-common`) to use it."
        )
        return None

    try:
        mistral_tok = MistralTokenizer.from_file(str(tekken_path))
    except Exception as e:  # noqa: BLE001 - any parse failure just falls through to the next rung
        logger.warning(
            f"Failed to load the Tekken tokenizer at {tekken_path}: {type(e).__name__}: {e}. "
            "Falling through to the next tokenizer source."
        )
        return None

    logger.info(f"Loaded Tekken tokenizer from {description}: {tekken_path.name}")
    return _TekkenRawTextAdapter(mistral_tok)


def _tekken_backend_from_dir(path: Path, description: str, logger: Any) -> Optional[AnyModel]:
    """Load a directory's ``tekken.json`` as transformers' mistral-common backend.

    This is the non-FLUX.2 route. Without a chat template there are no structural markers to
    splice, so the backend encodes correctly on its own -- checked id-for-id against
    ERNIE-Image's released ``tokenizer.json`` -- while being a real ``PreTrainedTokenizerBase``
    that returns id lists, which is what the conditioning nodes require. ``None`` on any failure,
    so the caller keeps falling through the ladder.
    """
    try:
        tokenizer = MistralCommonBackend.from_pretrained(
            path, local_files_only=True, model_max_length=_MINISTRAL_3B_MAX_PROMPT_TOKENS
        )
    except Exception as e:  # noqa: BLE001 - a probe rung must never kill the load
        logger.warning(
            f"Failed to load the Tekken tokenizer from {description}: {type(e).__name__}: {e}. "
            "Falling through to the next tokenizer source."
        )
        return None

    logger.info(f"Loaded Tekken tokenizer from {description} as {type(tokenizer).__name__}")
    return tokenizer


def _try_load_tekken_from_dir(path: Path, description: str, logger: Any, *, flux2_template: bool) -> Optional[AnyModel]:
    """Load a directory's standalone ``tekken.json`` in the form this encoder family needs.

    This rung must run *before* the `AutoProcessor` / `AutoTokenizer` probe. A directory holding
    `tekken.json` next to a `config.json` (`model_type: "mistral3"`) — the layout of an official
    mistralai download — does not fail there: it resolves to a mistral-common-backed tokenizer,
    which BPE-encodes `[SYSTEM_PROMPT]` / `[INST]` as literal text instead of splicing them as
    single Tekken special ids. The encode "works" and conditioning is silently off-distribution,
    so never let that object represent this layout when we can read the vocab ourselves.
    """
    tekken_path = path / _TEKKEN_FILENAME
    if not tekken_path.is_file():
        return None
    if not flux2_template:
        return _tekken_backend_from_dir(path, description, logger)
    return _tekken_adapter_from_file(tekken_path, description, logger)


def _normalize_tokenizer(obj: AnyModel, description: str, logger: Any, *, flux2_template: bool) -> Optional[AnyModel]:
    """Return a tokenizer safe to use for FLUX.2 conditioning, or ``None`` to keep falling.

    Only applies to the encoders that consume FLUX.2's template. For the others a
    mistral-common-backed tokenizer is exactly what we want and is returned untouched.

    `AutoTokenizer` resolves a Tekken-carrying directory to a mistral-common-backed tokenizer
    (`MistralCommonBackend` on transformers 5.x, `MistralCommonTokenizer` on 4.5x). Its
    ``__call__`` runs the raw BPE, so the template's structural markers become literal text
    rather than their single Tekken ids — a silent corruption, not a load failure. It does hold
    the underlying ``MistralTokenizer``, so re-wrap that in the same adapter the embedded-Tekken
    rung uses; only when it cannot be reached do we discard the result and keep falling.
    """
    if not flux2_template:
        return obj

    if not any(cls.__module__.endswith("tokenization_mistral_common") for cls in type(obj).__mro__):
        return obj

    inner = getattr(obj, "tokenizer", None)
    if inner is None:
        logger.warning(
            f"{type(obj).__name__} from {description} is mistral-common-backed but exposes no underlying "
            "tokenizer; it would encode the FLUX.2 template markers as literal text. Ignoring it."
        )
        return None

    logger.info(f"Re-wrapping the mistral-common tokenizer from {description} for raw FLUX.2 template encoding")
    return _TekkenRawTextAdapter(inner)


# Loader classes to try, in order, whenever we probe a local directory for a tokenizer.
#
# `AutoProcessor` alone is not enough: transformers resolves a `config.json` with
# `model_type: "mistral3"` — precisely the BFL-style standalone-encoder layout — to a *multimodal*
# processor and raises `OSError: Can't load image processor ...` for the missing
# `preprocessor_config.json`, before it ever looks at the tokenizer files sitting right there.
# `AutoTokenizer` loads the same directory fine, so try both.
_TOKENIZER_LOADER_CLASSES = (AutoProcessor, AutoTokenizer)

# The failures we *expect* from a probe rung, logged quietly. `KeyError` is in the tuple because
# `AutoTokenizer` raises `KeyError: 'special_tokens'` on a directory carrying a `tekken.json` it
# cannot parse. This tuple now only selects the log level — the probes catch broadly, see
# `_log_tokenizer_probe_failure`.
_TOKENIZER_LOAD_ERRORS = (OSError, EnvironmentError, ValueError, KeyError)


def _log_tokenizer_probe_failure(loader_cls: Any, description: str, exc: Exception, logger: Any) -> None:
    """Log a probe miss, loudly when it was not one of the expected failures.

    The broad catch at the call sites is required, not defensive padding:
    `AutoTokenizer.from_pretrained` on a directory whose `tokenizer_config.json` names a
    `tokenizer_class` the installed transformers does not know resolves that class to ``None``
    and raises ``AttributeError: 'NoneType' object has no attribute 'from_pretrained'`` —
    transformers' `tokenizer_class_from_name` has no ``None`` guard. That is exactly what
    `MistralCommonTokenizer.save_pretrained` writes, and such an exception escaping a *probe*
    would kill the whole load while a working fallback rung remains.
    """
    message = f"{loader_cls.__name__} could not load a tokenizer from {description}: {type(exc).__name__}: {exc}"
    if isinstance(exc, _TOKENIZER_LOAD_ERRORS):
        logger.debug(message)
    else:
        logger.warning(message)


def _try_load_tokenizer_from_dir(path: Path, description: str, logger: Any, *, flux2_template: bool) -> AnyModel | None:
    """Try both loader classes against a local directory. Returns None if neither works."""
    for loader_cls in _TOKENIZER_LOADER_CLASSES:
        try:
            obj = loader_cls.from_pretrained(path, local_files_only=True)
        except Exception as e:  # noqa: BLE001 - a probe rung must never kill the load
            _log_tokenizer_probe_failure(loader_cls, description, e, logger)
            continue
        normalized = _normalize_tokenizer(obj, description, logger, flux2_template=flux2_template)
        if normalized is None:
            continue
        logger.info(f"Loaded Mistral tokenizer from {description}: {type(normalized).__name__}")
        return normalized
    return None


def _load_tokenizer_from_hf(logger: Any, *, policy: _TokenizerPolicy) -> AnyModel:
    """Download / load the canonical tokenizer for this encoder family from HuggingFace."""
    source, subfolder = policy.fallback_source
    attempts: list[str] = []
    for local_only in (True, False):
        for loader_cls in _TOKENIZER_LOADER_CLASSES:
            description = f"{source}:{subfolder} (local_only={local_only})"
            try:
                obj = loader_cls.from_pretrained(source, subfolder=subfolder, local_files_only=local_only)
            except Exception as e:  # noqa: BLE001 - keep trying the remaining rungs, then raise below
                _log_tokenizer_probe_failure(loader_cls, description, e, logger)
                attempts.append(f"{loader_cls.__name__}(local_only={local_only}): {type(e).__name__}")
                continue
            normalized = _normalize_tokenizer(obj, description, logger, flux2_template=policy.flux2_template)
            if normalized is None:
                attempts.append(f"{loader_cls.__name__}(local_only={local_only}): unusable {type(obj).__name__}")
                continue
            logger.info(f"Loaded Mistral processor/tokenizer: {type(normalized).__name__} from {description}")
            return normalized

    raise RuntimeError(
        f"Could not load the Mistral tokenizer from {source}:{subfolder}. "
        "Workarounds: (1) install an encoder that embeds the Tekken tokenizer "
        "(Comfy-Org safetensors or gguf-org cow GGUFs) and `pip install mistral-common`, "
        "(2) run once with internet access to populate the HF cache, or "
        f"(3) pre-cache the tokenizer: `huggingface-cli download {source} --include '{subfolder}/*'`. "
        f"Tried: {'; '.join(attempts)}"
    )


def _load_tokenizer_for_model(model_path: Path, logger: Any, variant: MistralVariantType) -> AnyModel:
    """Load a tokenizer matching the given Mistral encoder model path.

    The variant decides the *form*, not just the source. FLUX.2's encoders consume a
    pre-formatted template whose structural markers have to be spliced in as single Tekken ids,
    which is what ``_TekkenRawTextAdapter`` exists for. Ministral 3B (ERNIE-Image) has no chat
    template, so it takes a plain HuggingFace tokenizer over the same vocab instead.

    Strategy (first hit wins):

    1. **Embedded Tekken** — Comfy-Org safetensors and gguf-org cow GGUFs ship
       the canonical Tekken JSON as a ``tekken_model`` U8 tensor; we extract it
       and wrap it via ``mistral_common``.
    2. **Standalone ``tekken.json``** — official mistralai downloads ship the Tekken
       vocab as a plain file. It is read here, ahead of the transformers probes,
       because those *succeed* on such a directory and hand back a mistral-common
       tokenizer that mis-encodes the template markers (see ``_try_load_tekken_from_dir``).
    3. **Sibling ``tokenizer/`` folder** — diffusers-style HuggingFace layouts.
    4. **Root-directory processor / tokenizer files** — standalone downloads that
       ship them alongside the encoder weights at the folder root. BFL-style
       ``model_type: "mistral3"`` layouts resolve only via ``AutoTokenizer``.
    5. **HuggingFace fallback** — fetches the canonical tokenizer for the family, BFL's for
       FLUX.2 and ERNIE-Image's for Ministral 3B.
    """
    policy = _tokenizer_policy(variant)
    flux2_template = policy.flux2_template

    # 1. Single-file with embedded Tekken
    embedded = _try_load_embedded_tekken(model_path, logger, flux2_template=flux2_template)
    if embedded is not None:
        return embedded

    if model_path.is_dir():
        tokenizer_dir = model_path / "tokenizer"

        # 2. A standalone tekken.json anywhere we would otherwise probe with transformers.
        for probe_dir, description in ((tokenizer_dir, "sibling tokenizer/"), (model_path, "model root")):
            if probe_dir.is_dir():
                tekken = _try_load_tekken_from_dir(probe_dir, description, logger, flux2_template=flux2_template)
                if tekken is not None:
                    return tekken

        # 3. Diffusers folder with sibling tokenizer/
        if tokenizer_dir.exists():
            obj = _try_load_tokenizer_from_dir(
                tokenizer_dir, "sibling tokenizer/", logger, flux2_template=flux2_template
            )
            if obj is not None:
                return obj
        # Some diffusers folders ship the encoder weights as text_encoder/*.safetensors
        # which may embed Tekken — probe each in turn.
        text_encoder_dir = model_path / "text_encoder"
        if text_encoder_dir.is_dir():
            for st in sorted(text_encoder_dir.glob("*.safetensors")):
                embedded = _try_load_embedded_tekken(st, logger, flux2_template=flux2_template)
                if embedded is not None:
                    return embedded
        # 4. Processor / tokenizer files alongside the encoder weights at the folder root.
        obj = _try_load_tokenizer_from_dir(model_path, "model root", logger, flux2_template=flux2_template)
        if obj is not None:
            return obj

    # 5. HF fallback
    return _load_tokenizer_from_hf(logger, policy=policy)


@ModelLoaderRegistry.register(
    base=BaseModelType.Any,
    type=ModelType.MistralEncoder,
    format=ModelFormat.MistralEncoder,
)
class MistralEncoderDiffusersLoader(ModelLoader):
    """Load a Mistral text encoder from a HuggingFace folder layout.

    Handles both the full FLUX.2-dev pipeline layout (with sibling ``tokenizer/``)
    and a standalone download where ``text_encoder/`` files live at the root.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, MistralEncoder_Diffusers_Config):
            raise ValueError("Only MistralEncoder_Diffusers_Config models are supported here.")

        model_path = Path(config.path)
        text_encoder_path = model_path / "text_encoder"

        # Standalone download: text_encoder files at the root.
        if not text_encoder_path.exists() and (model_path / "config.json").exists():
            text_encoder_path = model_path

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        match submodel_type:
            case SubModelType.Tokenizer:
                logger = InvokeAILogger.get_logger("MistralEncoderProcessor")
                # Let the multi-strategy loader own the full ladder: embedded Tekken,
                # sibling tokenizer/, root-level processor files, then the HF fallback.
                return _load_tokenizer_for_model(model_path, logger, config.variant)
            case SubModelType.TextEncoder:
                # Lazy import: transformers may load `Mistral3ForConditionalGeneration`
                # only when the diffusers/transformers version supports it.
                from transformers import AutoModel

                model = AutoModel.from_pretrained(
                    text_encoder_path,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                # `MistralModel.norm` is always built by transformers, but the
                # 30-layer cow distillation was trained against the post-layer-29
                # state *without* the final norm — swap it for Identity to match
                # ComfyUI's reference implementation. ``Mistral3ForConditionalGeneration``
                # nests the LM under ``.language_model``; handle both layouts.
                inner = getattr(model, "language_model", None) or model
                logger = InvokeAILogger.get_logger("MistralEncoderDiffusersLoader")
                _strip_final_norm_for_cow(inner, config.variant, logger)
                _warn_if_40_layer_mistral(config.variant, logger)
                # The BFL `text_encoder` checkpoint maps to `Mistral3Model`, which ships a
                # `vision_tower` + `multi_modal_projector` (~0.8GB of real weights). The
                # invocation only ever runs `.language_model`, so drop the vision path to
                # keep it out of the RAM cache and every cache->VRAM transfer. The
                # checkpoint/GGUF loaders already build a bare `MistralModel`.
                for unused in ("vision_tower", "multi_modal_projector"):
                    if getattr(model, unused, None) is not None:
                        setattr(model, unused, None)
                return model

        raise ValueError(
            "Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(
    base=BaseModelType.Any,
    type=ModelType.MistralEncoder,
    format=ModelFormat.Checkpoint,
)
class MistralEncoderCheckpointLoader(ModelLoader):
    """Load a Mistral encoder from a single safetensors file (text-only)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, MistralEncoder_Checkpoint_Config):
            raise ValueError("Only MistralEncoder_Checkpoint_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)
            case SubModelType.Tokenizer:
                logger = InvokeAILogger.get_logger("MistralEncoderProcessor")
                return _load_tokenizer_for_model(Path(config.path), logger, config.variant)

        raise ValueError(
            "Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: MistralEncoder_Checkpoint_Config) -> AnyModel:
        from safetensors.torch import load_file

        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        model_path = Path(config.path)
        sd = load_file(model_path)
        sd = _strip_known_prefixes(sd)

        # Comfy-Org's Ministral 3B file ships the full multimodal stack: a Pixtral vision tower and
        # its projector, 222 tensors and ~0.9 GB. Only the language tower encodes prompts and the
        # encoder-only model has no slot for the rest, so drop them before the dequantize/cast pass
        # pays to convert weights that would then be reported as unexpected keys.
        vision_keys = [
            key for key in sd if isinstance(key, str) and key.startswith(("vision_tower.", "multi_modal_projector."))
        ]
        for key in vision_keys:
            del sd[key]
        if vision_keys:
            logger.info(f"Mistral encoder: dropped {len(vision_keys)} vision-tower tensor(s); prompts use the LM only")

        # The header names layers before this loader strips its own wrapper prefixes on top of the generic ones,
        # so the names need both lists. With only the generic tuple, a `language_model.`-prefixed redistribution
        # keeps its names while the sd keys lose the prefix: every `full_precision_matrix_mult` is silently
        # ignored, and nvfp4 layers only the header names are refused as unnamed.
        header_hints = strip_layer_path_prefix(
            parse_quantization_metadata(read_safetensors_metadata(model_path, logger)),
            prefixes=(*MISTRAL_KEY_PREFIXES, *TRANSFORMER_KEY_PREFIXES),
        )

        # Comfy's fp4_mixed build keeps most projections in nvfp4 beside scaled fp8. Take those out before
        # anything below reads the side channel: the keep-fp8 branch pops every `weight_scale` and discards the
        # ones whose weight is not float8, nvfp4's block scales included, and the dequantizing branch and the cast
        # would widen the packed codes. `install_nvfp4_layers` puts them back, packed.
        nvfp4_payloads = pop_nvfp4_layers(sd, header_layers=header_hints)

        # These redistributions are ComfyUI 'scaled fp8': an fp8 weight plus a `weight_scale`.
        # Folding the scale doubles the encoder -- 16.8 GiB on disk becomes 32.3 GiB in bf16, which
        # does not fit on a 24 GB card even on its own. Keeping the weights quantized is therefore
        # not just a speed question here, it decides whether the model loads at all.
        #
        # Both key rewrites in this loader (`_strip_known_prefixes` above and
        # `_convert_for_bare_mistral_model` below) are plain prefix operations, so a sibling
        # `.weight_scale` travels with its weight automatically -- no fused projections to split.
        # Storage keeps them too, gated on the device alone: `_should_use_fp8` excludes text encoders
        # by design (fp8 rounding costs text quality), so there is no per-model setting to read here.
        # Folding is the fallback only where fp8 cannot be held at all, and folding is what hurts
        # here: 16.8 GiB on disk becomes 32.3 GiB resident. Note what that makes this -- on CUDA the
        # device check is always true, so the choice is unconditional and has no user setting behind
        # it. This loader never calls the layerwise cast (text encoders are excluded there), so the
        # kept weights reach `CustomLinear` with their scales intact.
        # Before either branch, because both consume what the check reads: `extract_fp8_scaled_layers`
        # pops every scale key -- discarding the ones whose weight is not float8, int8's included --
        # and deletes the markers with them. On CUDA that branch is always the one taken, so a check
        # placed after it would never run on the device almost everyone loads on.
        reject_int8_layers_a_plain_fold_cannot_decode(sd, "This Mistral encoder checkpoint")

        keep_fp8 = should_keep_fp8_weights(target_device) or _device_supports_fp8_storage(target_device, logger)
        fp8_layers: dict[str, Any] = {}
        if keep_fp8:
            layer_hints = {**extract_comfy_quant_hints(sd), **header_hints}
            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)

        # Ministral 3B is a different architecture, not a smaller Mistral Small 3: YaRN RoPE and a
        # position-dependent attention scale that only `Ministral3Model` applies. The variant was
        # decided from the geometry at install time and is the single source of truth here.
        #
        # Both builders are handed the nvfp4 layers taken out above: read from the state dict alone,
        # a packed encoder would be configured from the layers that happen to be left in it.
        encoder_config: MistralConfig | Ministral3Config
        model_class: type[MistralModel] | type[Ministral3Model]
        if config.variant is MistralVariantType.Ministral3B:
            encoder_config = _build_ministral3_config(sd, torch_dtype=model_dtype, packed_layers=nvfp4_payloads)
            model_class = Ministral3Model
        else:
            encoder_config = _build_mistral_config(sd, torch_dtype=model_dtype, packed_layers=nvfp4_payloads)
            model_class = MistralModel
        logger.info(
            f"Mistral encoder config (checkpoint): variant={config.variant.value}, "
            f"layers={encoder_config.num_hidden_layers}, hidden={encoder_config.hidden_size}, "
            f"heads={encoder_config.num_attention_heads}, kv_heads={encoder_config.num_key_value_heads}, "
            f"intermediate={encoder_config.intermediate_size}"
        )

        # Built before the reservation, which depends on its modules: they decide which fp8 weights and which
        # nvfp4 layers stay quantized.
        with accelerate.init_empty_weights():
            model = model_class(encoder_config)
        skip_patterns = _model_declared_skip_patterns(model)

        # Adapt CausalLM-prefixed keys for bare MistralModel -- a rename, nothing is widened -- which also drops the
        # LM head, the single largest tensor (vocab x hidden), before anything below pays to convert it. The
        # recovered scales and the packed layers are keyed on the same CausalLM paths and need the same strip, or
        # nothing resolves: every fp8 weight stays quantized but unscaled, and every packed layer names no module.
        # Only the text tower has modules in the bare model, so packed layers outside it -- the LM head, a vision
        # tower a multimodal export bundles -- are dropped, as their dense weights are dropped by the non-strict
        # load.
        sd = _convert_for_bare_mistral_model(sd)
        nvfp4_payloads = {
            _bare_mistral_path(path): payload for path, payload in nvfp4_payloads.items() if path.startswith("model.")
        }
        fp8_layers = {_bare_mistral_path(path): layer for path, layer in fp8_layers.items()}

        # Decided once, here, and handed to both the prediction and the cast. Reading `bool(fp8_layers)`
        # at each call site is not the same expression twice: `split_fp8_scaled_layers` rebinds the name
        # below, so the cast would see the *post-split* mapping. An encoder whose scaled layers are all
        # block-wise (or all skip-patterned) empties it, the cast would flip to `keep_fp8=False` and widen
        # every *raw* fp8 weight too -- each charged one byte per element in the reservation above.
        keep_raw_fp8 = bool(fp8_layers)

        # One reservation, before the dequantizing branch or the split widens a single weight -- `make_room` makes
        # that much room rather than adding to an earlier one. The state dict is sized by the predicate the split and
        # the cast below decide with, the nvfp4 layers as they will be held.
        reserve_for_load(
            self._ram_cache.make_room,
            sd,
            model_dtype,
            keep_fp8=keep_raw_fp8,
            model=model,
            skip_patterns=skip_patterns,
            fp8_layers=fp8_layers,
            nvfp4_payloads=nvfp4_payloads,
        )

        if not fp8_layers:
            # Dequantize straight to the compute dtype (per-tensor peak, not whole-dict fp32).
            sd = _drop_quantization_metadata(sd, logger, target_dtype=model_dtype)
        else:
            # `extract_fp8_scaled_layers` removes the keys it interprets, but this producer also
            # emits `.scale` and `scaled_fp8*`, which it does not recognize. On the dequantizing
            # path `_drop_quantization_metadata` takes those; here nothing did, so they reached
            # `load_state_dict` and padded the "ignored N unexpected keys" line.
            for k in [k for k in sd if isinstance(k, str) and (k.endswith(".scale") or k.startswith("scaled_fp8"))]:
                del sd[k]

        # Layers the cast would dequantize anyway are folded here, scale applied, so the cast never
        # strips a scale that can no longer be put back.
        fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)
        cast_state_dict(sd, model_dtype, keep_fp8=keep_raw_fp8, model=model, skip_patterns=skip_patterns)

        if nvfp4_payloads:
            packed = install_nvfp4_layers(model, sd, nvfp4_payloads, model_dtype, skip_patterns)
            logger.info(f"Mistral encoder: kept {packed} of {len(nvfp4_payloads)} nvfp4 layer(s) packed.")

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        log_unexpected_keys("Mistral encoder checkpoint", unexpected)
        if missing:
            # Re-initialize any RMSNorm weights that may have been pruned during repackaging.
            for name in missing:
                if name.endswith(".weight") and "norm" in name:
                    try:
                        parent_name, attr = name.rsplit(".", 1)
                        parent = model.get_submodule(parent_name)
                        param = getattr(parent, attr)
                        if param.is_meta:
                            setattr(
                                parent,
                                attr,
                                torch.nn.Parameter(torch.ones(param.shape, dtype=model_dtype), requires_grad=False),
                            )
                    except (AttributeError, ValueError):
                        continue

        # Re-init any remaining meta buffers (e.g. RoPE inv_freq is computed from config).
        _reinit_inv_freq(model, encoder_config, model_dtype)

        _materialize_remaining_meta_tensors(model, model_dtype, logger)
        _strip_final_norm_for_cow(model, config.variant, logger)
        _warn_if_40_layer_mistral(config.variant, logger)

        if fp8_layers:
            attached = attach_fp8_scales(model, fp8_layers)
            logger.info(
                f"Mistral encoder: kept {attached} layer(s) in fp8 (scaled fp8 checkpoint, kept for {self._fp8_kept_reason()})"
            )
            warn_on_unattached_scales(logger, "Mistral encoder", attached, fp8_layers)
            marked = sum(1 for layer in fp8_layers.values() if layer.full_precision_matmul)
            if marked and full_precision_hints_respected():
                logger.info(
                    f"Mistral encoder: {marked} of {len(fp8_layers)} layer(s) are marked "
                    "full_precision_matrix_mult and will dequantize per forward."
                )

        return model


@ModelLoaderRegistry.register(
    base=BaseModelType.Any,
    type=ModelType.MistralEncoder,
    format=ModelFormat.GGUFQuantized,
)
class MistralEncoderGGUFLoader(ModelLoader):
    """Load a GGUF-quantized Mistral encoder (text-only)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, MistralEncoder_GGUF_Config):
            raise ValueError("Only MistralEncoder_GGUF_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_gguf(config)
            case SubModelType.Tokenizer:
                logger = InvokeAILogger.get_logger("MistralEncoderProcessor")
                return _load_tokenizer_for_model(Path(config.path), logger, config.variant)

        raise ValueError(
            "Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_gguf(self, config: MistralEncoder_GGUF_Config) -> AnyModel:
        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = gguf_sd_loader(Path(config.path), compute_dtype=compute_dtype)

        # Read RoPE / context hyperparameters from the GGUF metadata before key
        # conversion strips them. Mistral GGUFs use the llama.* prefix because
        # they share llama.cpp's architecture family. Falling back silently is OK:
        # `_build_mistral_config` defaults to Mistral Small 3.1 values when the
        # override is None.
        gguf_meta = _read_gguf_metadata_values(Path(config.path), ("llama.rope.freq_base", "llama.context_length"))
        rope_raw = gguf_meta.get("llama.rope.freq_base")
        rope_theta = float(rope_raw) if isinstance(rope_raw, (int, float)) else None
        ctx_raw = gguf_meta.get("llama.context_length")
        max_pos = int(ctx_raw) if isinstance(ctx_raw, (int, float)) else None
        if rope_theta is not None:
            logger.info(f"GGUF metadata: rope_theta={rope_theta}, max_position={max_pos}")

        # llama.cpp stores layers as `blk.N.*`. Normalize to transformers' `model.layers.N.*` if needed;
        # `_strip_known_prefixes` and `_convert_for_bare_mistral_model` below adapt that to MistralModel.
        if is_llamacpp_decoder_state_dict(sd):
            logger.info("Detected llama.cpp GGUF format, converting keys to transformers format")
            sd = convert_llamacpp_decoder_keys(sd)

        sd = _strip_known_prefixes(sd)

        mistral_config = _build_mistral_config(
            sd,
            torch_dtype=compute_dtype,
            rope_theta=rope_theta,
            max_position_embeddings=max_pos,
        )
        logger.info(
            f"Mistral encoder config (GGUF): layers={mistral_config.num_hidden_layers}, "
            f"hidden={mistral_config.hidden_size}, heads={mistral_config.num_attention_heads}, "
            f"kv_heads={mistral_config.num_key_value_heads}, intermediate={mistral_config.intermediate_size}"
        )

        # Adapt CausalLM-prefixed keys for bare MistralModel.
        sd = _convert_for_bare_mistral_model(sd)

        with accelerate.init_empty_weights():
            model = MistralModel(mistral_config)

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        log_unexpected_keys("Mistral GGUF encoder", unexpected)
        if missing:
            logger.debug(
                f"Mistral encoder (GGUF): {len(missing)} keys missing from state dict (first 5: {missing[:5]})"
            )

        materialize_quantized_embedding(model.embed_tokens, ram_cache=self._ram_cache)

        _reinit_inv_freq(model, mistral_config, compute_dtype)

        _materialize_remaining_meta_tensors(model, compute_dtype, logger)
        _strip_final_norm_for_cow(model, config.variant, logger)
        _warn_if_40_layer_mistral(config.variant, logger)

        return model
