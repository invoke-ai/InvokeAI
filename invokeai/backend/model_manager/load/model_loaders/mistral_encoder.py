# Copyright (c) 2026, The InvokeAI Development Team
"""Model loaders for the Mistral text encoder used by FLUX.2 [dev].

FLUX.2 [dev] uses BFL's 30-layer "cow-mistral3-small" distillation as its sole
text encoder. The diffusers release wraps it in the multimodal
``Mistral3ForConditionalGeneration``; standalone single-file safetensors
(Comfy-Org bf16/fp8/fp4) and GGUF redistributions (gguf-org cow variants) ship
only the text tower, which we load as an encoder-only ``MistralModel``.

Both single-file packagings embed the canonical Tekken tokenizer as a U8 tensor
named ``tekken_model`` (~19 MB). When ``mistral_common`` is installed we use
that embedded tokenizer directly; otherwise we fall back to fetching the
tokenizer from ``black-forest-labs/FLUX.2-dev`` via HuggingFace.
"""

from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from transformers import AutoProcessor, AutoTokenizer, MistralConfig, MistralModel

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.mistral_encoder import (
    MistralEncoder_Checkpoint_Config,
    MistralEncoder_Diffusers_Config,
    MistralEncoder_GGUF_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

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

# HuggingFace fallback for the tokenizer when the model file doesn't embed
# tekken_model (older cow GGUFs without the embedded blob, or a diffusers folder
# without a sibling tokenizer/). We only need the BFL canonical source — upstream
# Mistral tokenizers (3.1 / 3.2) don't match BFL's chat template exactly.
_TOKENIZER_FALLBACK_SOURCE: tuple[str, str] = ("black-forest-labs/FLUX.2-dev", "tokenizer")


def _build_mistral_config(
    state_dict: dict[str, Any],
    torch_dtype: torch.dtype,
    rope_theta: float | None = None,
    max_position_embeddings: int | None = None,
) -> MistralConfig:
    """Build a transformers ``MistralConfig`` from a cow-mistral3-small state dict.

    Reads the bulk shapes from the state dict (vocab, hidden, heads, kv_heads,
    intermediate, layer count). ``rope_theta`` and ``max_position_embeddings`` can
    be passed explicitly when an out-of-band source is available (e.g. GGUF
    metadata); otherwise we fall back to cow defaults.
    """
    # Vocab and hidden_size come from embed_tokens.
    embed_key = "model.embed_tokens.weight" if "model.embed_tokens.weight" in state_dict else None
    if embed_key is None:
        raise ValueError("State dict does not contain model.embed_tokens.weight")
    embed = state_dict[embed_key]
    embed_shape = embed.tensor_shape if isinstance(embed, GGMLTensor) else embed.shape
    vocab_size, hidden_size = int(embed_shape[0]), int(embed_shape[1])

    # Count layers by scanning self_attn.q_proj keys.
    layer_indices: set[int] = set()
    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
        if key.startswith("model.layers.") and ".self_attn.q_proj.weight" in key:
            try:
                layer_indices.add(int(key.split(".")[2]))
            except (ValueError, IndexError):
                pass
    num_hidden_layers = (max(layer_indices) + 1) if layer_indices else _COW_NUM_HIDDEN_LAYERS

    # Derive head counts from the first layer's attention projections.
    q_proj = state_dict.get("model.layers.0.self_attn.q_proj.weight")
    k_proj = state_dict.get("model.layers.0.self_attn.k_proj.weight")
    gate_proj = state_dict.get("model.layers.0.mlp.gate_proj.weight")
    head_dim = _COW_HEAD_DIM
    if q_proj is not None and k_proj is not None and gate_proj is not None:
        q_shape = q_proj.tensor_shape if isinstance(q_proj, GGMLTensor) else q_proj.shape
        k_shape = k_proj.tensor_shape if isinstance(k_proj, GGMLTensor) else k_proj.shape
        gate_shape = gate_proj.tensor_shape if isinstance(gate_proj, GGMLTensor) else gate_proj.shape
        num_attention_heads = int(q_shape[0]) // head_dim
        num_key_value_heads = int(k_shape[0]) // head_dim
        intermediate_size = int(gate_shape[0])
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
        torch_dtype=torch_dtype,
    )


def _read_gguf_metadata_value(path: Path, key: str) -> Any | None:
    """Read a single named field from a GGUF file's metadata header.

    Returns ``None`` if the key is missing or the file/header can't be read —
    callers must treat the return as best-effort and fall back to defaults.
    """
    try:
        import gguf

        reader = gguf.GGUFReader(path)
    except Exception:
        return None
    field = reader.fields.get(key)
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


def _read_gguf_metadata_float(path: Path, key: str) -> float | None:
    value = _read_gguf_metadata_value(path, key)
    return float(value) if isinstance(value, (int, float)) else None


def _read_gguf_metadata_int(path: Path, key: str) -> int | None:
    value = _read_gguf_metadata_value(path, key)
    return int(value) if isinstance(value, (int, float)) else None


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
        for prefix in ("text_encoder.", "language_model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        out[new_key] = value
    return out


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
            continue
        if key.startswith("lm_head."):
            continue
        if key.startswith("model."):
            out[key[len("model.") :]] = value
        else:
            out[key] = value
    return out


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
        is_norm = "norm" in name.split(".") or name.endswith("_norm.weight")
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


def _drop_quantization_metadata(sd: dict[str, Any], logger) -> dict[str, Any]:
    """Dequantize Comfy-Org-style FP8/FP4 weights and drop their metadata keys.

    Comfy-Org's Mistral FLUX.2 redistributions store quantized weights alongside
    ``*.weight_scale`` (and occasionally ``*.input_scale``) tensors. We apply the
    scale in-place and remove the metadata so transformers can load the result.
    """
    weight_scale_keys = [k for k in sd.keys() if isinstance(k, str) and k.endswith(".weight_scale")]
    dequantized = 0
    for scale_key in weight_scale_keys:
        weight_key = scale_key[: -len(".weight_scale")] + ".weight"
        if weight_key not in sd:
            continue
        weight = sd[weight_key].float()
        scale = sd[scale_key].float()
        if scale.shape != weight.shape and scale.numel() > 1:
            for dim in range(len(weight.shape)):
                if dim < len(scale.shape) and scale.shape[dim] != weight.shape[dim]:
                    block = weight.shape[dim] // scale.shape[dim]
                    if block > 1:
                        scale = scale.repeat_interleave(block, dim=dim)
        sd[weight_key] = weight * scale
        dequantized += 1
    if dequantized:
        logger.info(f"Dequantized {dequantized} Comfy-Org-style quantized weights")

    drop_suffixes = (".weight_scale", ".input_scale", ".scale")
    drop_keys = [
        k
        for k in sd.keys()
        if isinstance(k, str) and (k.endswith(drop_suffixes) or "comfy_quant" in k or k.startswith("scaled_fp8"))
    ]
    for k in drop_keys:
        del sd[k]
    return sd


def _flatten_message_content(content: Any) -> str:
    """Reduce HF chat-template content (str or [{type:"text", text:"..."}]) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


class _TekkenChatTemplateAdapter:
    """Expose HuggingFace's ``apply_chat_template`` surface backed by
    ``mistral_common.MistralTokenizer``.

    The FLUX.2 [dev] invocation only calls ``apply_chat_template(messages,
    tokenize=True, return_tensors='pt', padding='max_length', max_length=N)``,
    so only that surface is implemented.
    """

    def __init__(self, mistral_tokenizer: Any):
        self._tok = mistral_tokenizer
        # Mistral Small 3's <pad> id (token 11 in the Tekken vocab).
        self.pad_token_id = 11

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = True,
        return_dict: bool = True,
        return_tensors: str = "pt",
        add_generation_prompt: bool = False,
        padding: str | bool = "max_length",
        truncation: bool = True,
        max_length: int = 512,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if not tokenize or return_tensors != "pt":
            raise NotImplementedError(
                "_TekkenChatTemplateAdapter only supports tokenize=True / return_tensors='pt' "
                f"(got tokenize={tokenize}, return_tensors={return_tensors})"
            )

        from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        msgs: list[Any] = []
        for msg in messages:
            role = msg.get("role")
            content = _flatten_message_content(msg.get("content"))
            if role == "system":
                msgs.append(SystemMessage(content=content))
            elif role == "user":
                msgs.append(UserMessage(content=content))

        encoded = self._tok.encode_chat_completion(ChatCompletionRequest(messages=msgs))
        tokens: list[int] = list(encoded.tokens)

        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        attention: list[int] = [1] * len(tokens)

        if padding == "max_length":
            pad_needed = max_length - len(tokens)
            if pad_needed > 0:
                tokens.extend([self.pad_token_id] * pad_needed)
                attention.extend([0] * pad_needed)

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


def _try_load_embedded_tekken(model_path: Path, logger: Any) -> Optional[AnyModel]:
    """Extract the embedded Tekken tokenizer and wrap it in the HF-compatible adapter.

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

    import os
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="invokeai-tekken-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(tekken_bytes)
        mistral_tok = MistralTokenizer.from_file(tmp_path)
    except Exception as e:
        logger.warning(
            f"Failed to load embedded Tekken tokenizer from {model_path.name}: {type(e).__name__}: {e}. "
            "Falling back to the HuggingFace BFL tokenizer."
        )
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    logger.info(f"Loaded embedded Tekken tokenizer from {model_path.name}")
    return _TekkenChatTemplateAdapter(mistral_tok)


def _load_tokenizer_from_hf(logger: Any) -> AnyModel:
    """Download / load the BFL canonical FLUX.2 tokenizer from HuggingFace."""
    source, subfolder = _TOKENIZER_FALLBACK_SOURCE
    attempts: list[str] = []
    for local_only in (True, False):
        for loader_cls in (AutoProcessor, AutoTokenizer):
            try:
                obj = loader_cls.from_pretrained(source, subfolder=subfolder, local_files_only=local_only)
                logger.info(
                    f"Loaded Mistral processor/tokenizer: {type(obj).__name__} from "
                    f"{source}:{subfolder} (local_only={local_only})"
                )
                return obj
            except (OSError, EnvironmentError, ValueError) as e:
                attempts.append(f"{loader_cls.__name__}(local_only={local_only}): {type(e).__name__}")

    raise RuntimeError(
        f"Could not load FLUX.2 Mistral tokenizer from {source}:{subfolder}. "
        "Workarounds: (1) install a Mistral encoder that embeds the Tekken tokenizer "
        "(Comfy-Org safetensors or gguf-org cow GGUFs) and `pip install mistral-common`, "
        "(2) run once with internet access to populate the HF cache, or "
        "(3) pre-cache the tokenizer: "
        "`huggingface-cli download black-forest-labs/FLUX.2-dev --include 'tokenizer/*'`. "
        f"Tried: {'; '.join(attempts)}"
    )


def _load_tokenizer_for_model(model_path: Path, logger: Any) -> AnyModel:
    """Load a tokenizer matching the given Mistral encoder model path.

    Strategy (first hit wins):

    1. **Embedded Tekken** — Comfy-Org safetensors and gguf-org cow GGUFs ship
       the canonical Tekken JSON as a ``tekken_model`` U8 tensor; we extract it
       and wrap it via ``mistral_common``.
    2. **Sibling ``tokenizer/`` folder** — diffusers-style HuggingFace layouts.
    3. **BFL HuggingFace fallback** — fetches the canonical tokenizer from
       ``black-forest-labs/FLUX.2-dev/tokenizer``.
    """
    # 1. Single-file with embedded Tekken
    embedded = _try_load_embedded_tekken(model_path, logger)
    if embedded is not None:
        return embedded

    # 2. Diffusers folder with sibling tokenizer/
    if model_path.is_dir():
        tokenizer_dir = model_path / "tokenizer"
        if tokenizer_dir.exists():
            try:
                obj = AutoProcessor.from_pretrained(tokenizer_dir, local_files_only=True)
                logger.info(f"Loaded Mistral tokenizer from sibling tokenizer/: {type(obj).__name__}")
                return obj
            except (OSError, EnvironmentError, ValueError):
                pass
        # Some diffusers folders ship the encoder weights as text_encoder/*.safetensors
        # which may embed Tekken — probe each in turn.
        text_encoder_dir = model_path / "text_encoder"
        if text_encoder_dir.is_dir():
            for st in sorted(text_encoder_dir.glob("*.safetensors")):
                embedded = _try_load_embedded_tekken(st, logger)
                if embedded is not None:
                    return embedded

    # 3. HF fallback
    return _load_tokenizer_from_hf(logger)


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
        tokenizer_path = model_path / "tokenizer"

        # Standalone download: text_encoder files at the root.
        if not text_encoder_path.exists() and (model_path / "config.json").exists():
            text_encoder_path = model_path
        if not tokenizer_path.exists():
            # If tokenizer was not co-downloaded, fall back to root (some standalone
            # downloads include processor files alongside the encoder weights).
            tokenizer_path = model_path

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        match submodel_type:
            case SubModelType.Tokenizer:
                logger = InvokeAILogger.get_logger("MistralEncoderProcessor")
                # Try the sibling tokenizer/ first when the diffusers folder ships one,
                # else fall through to the multi-strategy loader (embedded Tekken / HF).
                if tokenizer_path.exists() and tokenizer_path != model_path:
                    try:
                        return AutoProcessor.from_pretrained(tokenizer_path, local_files_only=True)
                    except (OSError, EnvironmentError):
                        pass
                return _load_tokenizer_for_model(model_path, logger)
            case SubModelType.TextEncoder:
                # Lazy import: transformers may load `Mistral3ForConditionalGeneration`
                # only when the diffusers/transformers version supports it.
                from transformers import AutoModel

                return AutoModel.from_pretrained(
                    text_encoder_path,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )

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
                return _load_tokenizer_for_model(Path(config.path), logger)

        raise ValueError(
            "Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: MistralEncoder_Checkpoint_Config) -> AnyModel:
        from safetensors.torch import load_file

        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(Path(config.path))
        sd = _strip_known_prefixes(sd)
        sd = _drop_quantization_metadata(sd, logger)

        mistral_config = _build_mistral_config(sd, torch_dtype=model_dtype)
        logger.info(
            f"Mistral encoder config (checkpoint): layers={mistral_config.num_hidden_layers}, "
            f"hidden={mistral_config.hidden_size}, heads={mistral_config.num_attention_heads}, "
            f"kv_heads={mistral_config.num_key_value_heads}, intermediate={mistral_config.intermediate_size}"
        )

        # Cast tensors to compute dtype before loading.
        for k in list(sd.keys()):
            sd[k] = sd[k].to(model_dtype)

        # Adapt CausalLM-prefixed keys for bare MistralModel.
        sd = _convert_for_bare_mistral_model(sd)

        with accelerate.init_empty_weights():
            model = MistralModel(mistral_config)

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if unexpected:
            logger.debug(f"Mistral encoder: ignored {len(unexpected)} unexpected keys")
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
        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta and name.endswith("inv_freq"):
                parts = name.rsplit(".", 1)
                parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
                inv_freq = 1.0 / (
                    mistral_config.rope_theta
                    ** (torch.arange(0, mistral_config.head_dim, 2, dtype=torch.float32) / mistral_config.head_dim)
                )
                parent.register_buffer(parts[-1], inv_freq.to(model_dtype), persistent=False)

        _materialize_remaining_meta_tensors(model, model_dtype, logger)

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
                return _load_tokenizer_for_model(Path(config.path), logger)

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
        rope_theta = _read_gguf_metadata_float(Path(config.path), "llama.rope.freq_base")
        max_pos = _read_gguf_metadata_int(Path(config.path), "llama.context_length")
        if rope_theta is not None:
            logger.info(f"GGUF metadata: rope_theta={rope_theta}, max_position={max_pos}")

        # llama.cpp stores layers as `blk.N.*`. Normalize to transformers' `model.layers.N.*` if needed.
        is_llamacpp = any(isinstance(k, str) and k.startswith("blk.") for k in sd.keys())
        if is_llamacpp:
            logger.info("Detected llama.cpp GGUF format, converting keys to transformers format")
            sd = _convert_llamacpp_mistral_to_pytorch(sd)

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
        if unexpected:
            logger.debug(f"Mistral encoder (GGUF): ignored {len(unexpected)} unexpected keys")
        if missing:
            logger.debug(
                f"Mistral encoder (GGUF): {len(missing)} keys missing from state dict (first 5: {missing[:5]})"
            )

        # Embedding lookups require an indexable tensor — dequantize the GGMLTensor for embed_tokens.
        embed_weight = model.embed_tokens.weight
        if isinstance(embed_weight, GGMLTensor):
            model.embed_tokens.weight = torch.nn.Parameter(embed_weight.get_dequantized_tensor(), requires_grad=False)

        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta and name.endswith("inv_freq"):
                parts = name.rsplit(".", 1)
                parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
                inv_freq = 1.0 / (
                    mistral_config.rope_theta
                    ** (torch.arange(0, mistral_config.head_dim, 2, dtype=torch.float32) / mistral_config.head_dim)
                )
                parent.register_buffer(parts[-1], inv_freq.to(compute_dtype), persistent=False)

        _materialize_remaining_meta_tensors(model, compute_dtype, logger)

        return model


def _convert_llamacpp_mistral_to_pytorch(sd: dict[str, Any]) -> dict[str, Any]:
    """Rename llama.cpp Mistral keys to the transformers layout."""
    key_map = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    out: dict[str, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            out[key] = value
            continue
        if key in key_map:
            out[key_map[key]] = value
            continue
        # Per-layer keys: `blk.N.<thing>` -> `model.layers.N.<thing>`
        if key.startswith("blk."):
            parts = key.split(".", 2)  # ["blk", "<N>", "<rest>"]
            if len(parts) == 3:
                rest = parts[2]
                # Order matters: q_norm/k_norm must be checked BEFORE attn_q/attn_k
                # so we don't rewrite "attn_q_norm" -> "self_attn.q_proj_norm".
                rest = rest.replace("attn_q_norm.", "self_attn.q_norm.")
                rest = rest.replace("attn_k_norm.", "self_attn.k_norm.")
                rest = rest.replace("attn_q.", "self_attn.q_proj.")
                rest = rest.replace("attn_k.", "self_attn.k_proj.")
                rest = rest.replace("attn_v.", "self_attn.v_proj.")
                rest = rest.replace("attn_output.", "self_attn.o_proj.")
                rest = rest.replace("attn_norm.", "input_layernorm.")
                rest = rest.replace("ffn_norm.", "post_attention_layernorm.")
                rest = rest.replace("ffn_gate.", "mlp.gate_proj.")
                rest = rest.replace("ffn_up.", "mlp.up_proj.")
                rest = rest.replace("ffn_down.", "mlp.down_proj.")
                out[f"model.layers.{parts[1]}.{rest}"] = value
                continue
        out[key] = value
    return out
