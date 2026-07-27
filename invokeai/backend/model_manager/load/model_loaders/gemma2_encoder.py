"""Loader for the Gemma-2 text encoder used by PiD.

PiD only consumes the decoder block of the causal LM (see
`pid/_src/models/pixeldit_model.py::_load_text_encoder`:
`AutoModelForCausalLM.from_pretrained(...).get_decoder()`), so this loader
returns the decoder sub-module for the `TextEncoder` submodel and the
tokenizer for the `Tokenizer` submodel.
"""

import re
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.gemma2_encoder import (
    Gemma2Encoder_Gemma2Encoder_Config,
    Gemma2Encoder_GGUF_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.util.devices import TorchDevice

# llama.cpp GGUF tensor-name component -> Gemma2Model (decoder-only) component. PiD consumes only the
# decoder stack, so we target Gemma2Model directly: no `model.` prefix and no lm_head. Gemma-2 has no
# q/k norms (unlike Qwen3) but has pre/post attention and feed-forward RMSNorms.
_GEMMA_GGUF_KEY_MAP = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
}
_GEMMA_BLK_PATTERN = re.compile(r"^blk\.(\d+)\.(.+)$")


def _convert_gemma_llamacpp_to_pytorch(sd: dict[str, Any]) -> dict[str, Any]:
    """Map a llama.cpp Gemma-2 GGUF state dict to Gemma2Model (decoder-only) parameter names.

    Raises ValueError on any tensor key that has no mapping, so a wrong/contaminated checkpoint fails
    loudly here rather than silently dropping weights.
    """
    out: dict[str, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            out[key] = value
            continue
        m = _GEMMA_BLK_PATTERN.match(key)
        if m:
            idx, rest = m.group(1), m.group(2)
            component, _, suffix = rest.partition(".")
            mapped = _GEMMA_GGUF_KEY_MAP.get(component)
            if mapped is None:
                raise ValueError(f"Unmapped Gemma-2 GGUF tensor key component '{component}' (from '{key}')")
            out[f"layers.{idx}.{mapped}" + (f".{suffix}" if suffix else "")] = value
        elif key == "token_embd.weight":
            out["embed_tokens.weight"] = value
        elif key == "output_norm.weight":
            out["norm.weight"] = value
        else:
            raise ValueError(f"Unmapped Gemma-2 GGUF tensor key '{key}'")
    return out


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Gemma2Encoder, format=ModelFormat.Gemma2Encoder)
class Gemma2EncoderLoader(ModelLoader):
    """Loads a Gemma-2 causal LM directory and exposes its decoder + tokenizer."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Gemma2Encoder_Gemma2Encoder_Config):
            raise ValueError("Only Gemma2Encoder_Gemma2Encoder_Config models are supported here.")

        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.Tokenizer:
                return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            case SubModelType.TextEncoder:
                target_device = TorchDevice.choose_torch_device()
                model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                # PiD only ever uses the decoder block — the transformer stack
                # without the LM head. Upstream calls `.get_decoder()`, but
                # transformers 4.56 returns None for Gemma2, so we reach for
                # `.model` (the underlying Gemma2Model) directly and let the
                # rest of `causal_lm` (lm_head etc.) be garbage-collected.
                inner = getattr(causal_lm, "get_decoder", lambda: None)() or causal_lm.model
                inner.eval()
                inner.requires_grad_(False)
                return inner

        raise ValueError(
            f"Unsupported submodel type for Gemma2 encoder: {submodel_type!r}. Expected Tokenizer or TextEncoder."
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Gemma2Encoder, format=ModelFormat.GGUFQuantized)
class Gemma2EncoderGGUFLoader(ModelLoader):
    """Loads a single-file GGUF Gemma-2-2b encoder and exposes its decoder + tokenizer.

    Unlike a naive `from_pretrained(gguf_file=...)` (which dequantizes every weight into RAM/VRAM at load,
    giving no memory saving over the unquantized model), this keeps the large 2D projection weights as
    InvokeAI ``GGMLTensor`` — the model cache's custom linear handling dequantizes them on demand. Only the
    embedding and the RMSNorm weights are materialized eagerly. The tokenizer is still read from the GGUF
    metadata. Mirrors the Qwen3 GGUF encoder loader in ``z_image.py``.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Gemma2Encoder_GGUF_Config):
            raise ValueError("Only Gemma2Encoder_GGUF_Config models are supported here.")

        gguf_path = Path(config.path)

        match submodel_type:
            case SubModelType.Tokenizer:
                # The tokenizer is parsed from the GGUF metadata; no model tensors are loaded here.
                return AutoTokenizer.from_pretrained(gguf_path.parent, gguf_file=gguf_path.name, local_files_only=True)
            case SubModelType.TextEncoder:
                compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
                return load_gemma2_model_from_gguf(gguf_path, compute_dtype)

        raise ValueError(
            f"Unsupported submodel type for Gemma2 encoder: {submodel_type!r}. Expected Tokenizer or TextEncoder."
        )


def load_gemma2_model_from_gguf(gguf_path: Path, compute_dtype: "torch.dtype") -> AnyModel:
    """Build a Gemma2Model from a single-file llama.cpp GGUF, keeping the 2D projection weights quantized.

    The large 2D projection weights stay as InvokeAI ``GGMLTensor`` (dequantized on demand by the model
    cache's custom linear handling). Only the embedding and the RMSNorm weights are materialized eagerly.
    """
    import accelerate
    from transformers import Gemma2Config, Gemma2Model
    from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

    from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
    from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader

    # Read the Gemma-2 config from the GGUF metadata (avoids re-deriving Gemma-2 defaults), then load the
    # quantized storage as GGMLTensor wrappers.
    gemma_config = Gemma2Config(**load_gguf_checkpoint(str(gguf_path), return_tensors=False)["config"])
    sd = _convert_gemma_llamacpp_to_pytorch(gguf_sd_loader(gguf_path, compute_dtype=compute_dtype))

    with accelerate.init_empty_weights():
        model = Gemma2Model(gemma_config)

    _missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading Gemma-2 GGUF encoder: {unexpected[:10]}")

    # Materialize the weights that cannot remain quantized:
    #  - the token embedding, because nn.Embedding needs indexed access, and
    #  - every RMSNorm weight (1D). Gemma2RMSNorm does `self.weight.float()` and, critically, llama.cpp
    #    folds +1 into the stored norm weight while Gemma2RMSNorm re-adds it at runtime (`1 + weight`), so
    #    we subtract 1 to match transformers' Gemma2TensorProcessor. The large 2D projection weights stay
    #    GGMLTensor and are dequantized on demand by the model cache.
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if not isinstance(param, GGMLTensor):
                continue
            if isinstance(module, torch.nn.Embedding):
                setattr(module, name, torch.nn.Parameter(param.get_dequantized_tensor(), requires_grad=False))
            elif param.ndim == 1:
                setattr(module, name, torch.nn.Parameter(param.get_dequantized_tensor() - 1.0, requires_grad=False))

    # Re-materialize meta buffers not present in the GGUF (the rotary embedding's inv_freq).
    for name, buf in list(model.named_buffers()):
        if buf.is_meta and name.endswith("inv_freq"):
            head_dim = gemma_config.head_dim
            base = float(getattr(gemma_config, "rope_theta", 10000.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            parent.register_buffer(name.rsplit(".", 1)[-1], inv_freq.to(compute_dtype), persistent=False)

    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_params:
        raise RuntimeError(
            f"Gemma-2 GGUF encoder has parameters left on the meta device after loading: {meta_params[:10]}"
        )

    model.eval()
    return model
