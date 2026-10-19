"""Loader for single-file Qwen3.5 text encoders."""

from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from torch import nn

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config, qwen3_5_key_prefix
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.fp8_scaled import cast_state_dict, reject_quantized_side_channel
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import log_unexpected_keys, reject_incomplete_load


class _ZeroMLP(nn.Module):
    """Stands in for an MLP sub-block the checkpoint does not ship, so the layer adds nothing there."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _language_model_state_dict(sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    """The decoder's own tensors, keyed as `Qwen35Encoder` names them.

    Everything else is dropped: a multimodal export's vision tower, the LM head, the final norm and,
    in Anima-3.8B's `qwen35_4b.safetensors`, the 2560->1024 projection head stored under `norm.*` --
    none of which an encoder that returns intermediate hidden states runs.
    """
    return {
        key[len(prefix) :]: tensor
        for key, tensor in sd.items()
        if key.startswith(prefix) and key[len(prefix) :].startswith(("embed_tokens.", "layers."))
    }


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen35Encoder, format=ModelFormat.Checkpoint)
class Qwen35EncoderCheckpointLoader(ModelLoader):
    """Loads a single-file Qwen3.5 text encoder and the vendored Qwen3.5 tokenizer."""

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Qwen35Encoder_Checkpoint_Config):
            raise ValueError("Only Qwen35Encoder_Checkpoint_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)
            case SubModelType.Tokenizer:
                from invokeai.backend.qwen3_5.qwen3_5_encoder import load_bundled_qwen3_5_tokenizer

                # Single-file checkpoints ship no tokenizer.
                return load_bundled_qwen3_5_tokenizer()

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: Qwen35Encoder_Checkpoint_Config) -> AnyModel:
        from safetensors.torch import load_file

        from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder, qwen3_5_4b_text_config

        model_path = Path(config.path)
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(model_path)
        # Anima-3.8B's encoder ships raw fp8 projections (no scale), which the cast below handles
        # exactly. A scaled, int8 or nvfp4 build would load *wrong* here rather than fail, so refuse it
        # until one exists to support.
        reject_quantized_side_channel(sd, f"Qwen3.5 encoder checkpoint {model_path.name}")
        prefix = qwen3_5_key_prefix(sd)
        if prefix is None:
            raise ValueError(f"{model_path.name} is not a Qwen3.5 language model checkpoint.")
        sd = _language_model_state_dict(sd, prefix)

        text_config = qwen3_5_4b_text_config()
        with accelerate.init_empty_weights():
            model = Qwen35Encoder(text_config)

        # Anima-3.8B's encoder ships its last layer without the MLP sub-block (and its norm), because
        # nothing it was used for reads past that layer's attention. Give such a layer an MLP that adds
        # nothing, so the layer computes exactly what the file encodes instead of loading half-empty.
        for index, layer in enumerate(model.layers):
            if not any(key.startswith(f"layers.{index}.mlp.") for key in sd):
                layer.mlp = _ZeroMLP()
                layer.post_attention_layernorm = nn.Identity()
                self._logger.info(f"Qwen3.5 encoder: layer {index} ships without an MLP; running it attention-only.")

        keep_fp8 = self._keep_fp8_weights(config, SubModelType.TextEncoder)
        skip_patterns = _model_declared_skip_patterns(model)
        reserve_for_load(
            self._ram_cache.make_room,
            sd,
            model_dtype,
            keep_fp8=keep_fp8,
            model=model,
            skip_patterns=skip_patterns,
            fp8_layers={},
            nvfp4_payloads={},
        )
        kept = cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)

        result = model.load_state_dict(sd, strict=False, assign=True)
        log_unexpected_keys("Qwen3.5 encoder checkpoint", result.unexpected_keys)
        reject_incomplete_load(model, what=f"Qwen3.5 encoder checkpoint {model_path.name}")
        sd.clear()

        if kept:
            self._logger.info(f"Qwen3.5 encoder: kept {kept} raw fp8 weight(s) quantized ({self._fp8_kept_reason()}).")
        model.eval()
        return self._apply_fp8_layerwise_casting(model, config, SubModelType.TextEncoder)
