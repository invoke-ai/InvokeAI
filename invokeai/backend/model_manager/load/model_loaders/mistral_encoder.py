# Copyright (c) 2026, The InvokeAI Development Team
"""Model loaders for the Mistral text encoder used by FLUX.2 [dev].

FLUX.2 [dev] uses Mistral Small 3.1 (24B) as its sole text encoder. The diffusers
release ships it as the multimodal `Mistral3ForConditionalGeneration`; standalone
single-file safetensors and GGUF redistributions typically contain only the text
tower, which we load as an encoder-only `MistralModel`.
"""

from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from transformers import AutoProcessor, MistralConfig, MistralModel

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

# Architecture constants for Mistral Small 3.1 (used by FLUX.2 [dev]).
# Sourced from the FLUX.2-dev `text_encoder/config.json` (text-model side of the
# Mistral3 multimodal stack). Layers/heads/head_dim are needed when reconstructing
# the model from a state dict (single-file or GGUF) because the architecture is
# not embedded in those files.
_MISTRAL_SMALL_3_1_HIDDEN_SIZE = 5120
_MISTRAL_SMALL_3_1_INTERMEDIATE_SIZE = 32768
_MISTRAL_SMALL_3_1_NUM_HIDDEN_LAYERS = 40
_MISTRAL_SMALL_3_1_NUM_ATTENTION_HEADS = 32
_MISTRAL_SMALL_3_1_NUM_KV_HEADS = 8  # grouped-query attention
_MISTRAL_SMALL_3_1_HEAD_DIM = 128
_MISTRAL_SMALL_3_1_VOCAB_SIZE = 131072
_MISTRAL_SMALL_3_1_MAX_POSITION_EMBEDDINGS = 131072
_MISTRAL_SMALL_3_1_ROPE_THETA = 1000000.0
_MISTRAL_SMALL_3_1_RMS_NORM_EPS = 1e-5

# Default tokenizer / processor source. The official Mistral repo requires
# accepting a license; FLUX.2-dev embeds the same processor under `tokenizer/`
# and is the canonical companion for image-generation use.
_DEFAULT_PROCESSOR_SOURCE = "black-forest-labs/FLUX.2-dev"
_DEFAULT_PROCESSOR_SUBFOLDER = "tokenizer"


def _build_mistral_config(
    state_dict: dict[str, Any],
    torch_dtype: torch.dtype,
) -> MistralConfig:
    """Build a transformers ``MistralConfig`` from a Mistral Small 3.1 state dict.

    Reads the bulk shapes from the state dict (vocab, hidden, heads, kv_heads,
    intermediate, layer count) so we can also handle non-Small-3.1 Mistrals that
    happen to be wired through this loader.
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
    num_hidden_layers = (max(layer_indices) + 1) if layer_indices else _MISTRAL_SMALL_3_1_NUM_HIDDEN_LAYERS

    # Derive head counts from the first layer's attention projections.
    q_proj = state_dict.get("model.layers.0.self_attn.q_proj.weight")
    k_proj = state_dict.get("model.layers.0.self_attn.k_proj.weight")
    gate_proj = state_dict.get("model.layers.0.mlp.gate_proj.weight")
    head_dim = _MISTRAL_SMALL_3_1_HEAD_DIM
    if q_proj is not None and k_proj is not None and gate_proj is not None:
        q_shape = q_proj.tensor_shape if isinstance(q_proj, GGMLTensor) else q_proj.shape
        k_shape = k_proj.tensor_shape if isinstance(k_proj, GGMLTensor) else k_proj.shape
        gate_shape = gate_proj.tensor_shape if isinstance(gate_proj, GGMLTensor) else gate_proj.shape
        num_attention_heads = int(q_shape[0]) // head_dim
        num_key_value_heads = int(k_shape[0]) // head_dim
        intermediate_size = int(gate_shape[0])
    else:
        num_attention_heads = _MISTRAL_SMALL_3_1_NUM_ATTENTION_HEADS
        num_key_value_heads = _MISTRAL_SMALL_3_1_NUM_KV_HEADS
        intermediate_size = _MISTRAL_SMALL_3_1_INTERMEDIATE_SIZE

    return MistralConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=_MISTRAL_SMALL_3_1_MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=_MISTRAL_SMALL_3_1_RMS_NORM_EPS,
        tie_word_embeddings=False,
        rope_theta=_MISTRAL_SMALL_3_1_ROPE_THETA,
        attention_bias=False,
        attention_dropout=0.0,
        torch_dtype=torch_dtype,
    )


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


def _load_processor_with_offline_fallback() -> AnyModel:
    """Load the FLUX.2 Mistral processor (tokenizer + chat template) from cache, else HF."""
    try:
        return AutoProcessor.from_pretrained(
            _DEFAULT_PROCESSOR_SOURCE,
            subfolder=_DEFAULT_PROCESSOR_SUBFOLDER,
            local_files_only=True,
        )
    except (OSError, EnvironmentError):
        return AutoProcessor.from_pretrained(
            _DEFAULT_PROCESSOR_SOURCE,
            subfolder=_DEFAULT_PROCESSOR_SUBFOLDER,
        )


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
                try:
                    return AutoProcessor.from_pretrained(tokenizer_path, local_files_only=True)
                except (OSError, EnvironmentError):
                    # Fall back to the canonical FLUX.2-dev tokenizer subfolder on HF.
                    return _load_processor_with_offline_fallback()
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
                return _load_processor_with_offline_fallback()

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
                return _load_processor_with_offline_fallback()

        raise ValueError(
            "Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_gguf(self, config: MistralEncoder_GGUF_Config) -> AnyModel:
        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = gguf_sd_loader(Path(config.path), compute_dtype=compute_dtype)

        # llama.cpp stores layers as `blk.N.*`. Normalize to transformers' `model.layers.N.*` if needed.
        is_llamacpp = any(isinstance(k, str) and k.startswith("blk.") for k in sd.keys())
        if is_llamacpp:
            logger.info("Detected llama.cpp GGUF format, converting keys to transformers format")
            sd = _convert_llamacpp_mistral_to_pytorch(sd)

        sd = _strip_known_prefixes(sd)

        mistral_config = _build_mistral_config(sd, torch_dtype=compute_dtype)
        logger.info(
            f"Mistral encoder config (GGUF): layers={mistral_config.num_hidden_layers}, "
            f"hidden={mistral_config.hidden_size}, heads={mistral_config.num_attention_heads}, "
            f"kv_heads={mistral_config.num_key_value_heads}, intermediate={mistral_config.intermediate_size}"
        )

        with accelerate.init_empty_weights():
            model = MistralModel(mistral_config)

        model.load_state_dict(sd, strict=False, assign=True)

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
