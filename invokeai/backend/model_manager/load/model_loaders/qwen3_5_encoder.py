"""Loading of Qwen3.5 text encoders.

`load_qwen3_5_text_encoder` builds the encoder from the language-model tensors of one safetensors
file or of a model's shards; the single-file loader below and a model folder's loader both call it.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import accelerate
import torch
from safetensors import safe_open
from torch import nn

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.qwen3_5_encoder import (
    QWEN3_5_FINAL_NORM_VARIANTS,
    Qwen35Encoder_Checkpoint_Config,
    qwen3_5_key_prefix,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.fp8_scaled import cast_state_dict, reject_quantized_side_channel
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import log_unexpected_keys, reject_incomplete_load

if TYPE_CHECKING:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder


class _ZeroMLP(nn.Module):
    """Stands in for an MLP sub-block the checkpoint does not ship, so the layer adds nothing there."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _encoder_names(keys: Iterable[str], prefix: str, *, final_norm: bool) -> dict[str, str]:
    """The checkpoint keys `Qwen35Encoder` holds, mapped to its names for them.

    Everything else stays on disk: a multimodal export's vision tower, the LM head, the MTP head and,
    unless the final norm is read, whatever sits under `norm.*` -- in Anima-3.8B's
    `qwen35_4b.safetensors`, a 2560->1024 projection head that no encoder of it runs.
    """
    names: dict[str, str] = {}
    for key in keys:
        name = key[len(prefix) :]
        if key.startswith(prefix) and (
            name.startswith(("embed_tokens.", "layers.")) or (final_norm and name == "norm.weight")
        ):
            names[key] = name
    return names


def load_qwen3_5_text_encoder(
    loader: ModelLoader,
    files: Sequence[Path],
    text_config: "Qwen3_5TextConfig",
    *,
    final_norm: bool,
    keep_fp8: bool,
    what: str,
) -> "Qwen35Encoder":
    """Build a `Qwen35Encoder` from the language-model tensors of a safetensors file or of a model's shards.

    Nothing else is read, so a multimodal export's vision tower never reaches RAM.

    Args:
        loader: The calling loader; the load reserves room in its RAM cache.
        files: The safetensors file, or every shard of the model.
        text_config: The architecture to build.
        final_norm: Build and load the final norm, for a caller that reads the last hidden state. The
            checkpoint must then ship `norm.weight` and every layer whole.
        keep_fp8: Whether a raw-fp8 checkpoint's weights stay packed rather than widened: the calling
            loader's `_keep_fp8_weights` answer, asked by the caller because `test_fp8_capability.py`
            can only see the fp8 helpers reached through a loader's own methods. For an encoder that
            means the fp8 matmul: text encoders never take FP8 Storage's layerwise cast
            (`_FP8_EXCLUDED_SUBMODEL_TYPES`).
        what: Names the checkpoint in errors and log lines.
    """
    from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder

    keys_by_file: dict[Path, list[str]] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys_by_file[path] = list(f.keys())
    all_keys = dict.fromkeys(key for keys in keys_by_file.values() for key in keys)
    # Anima-3.8B's encoder ships raw fp8 projections (no scale), which the cast below handles
    # exactly. A scaled, int8 or nvfp4 build would load *wrong* here rather than fail, so refuse it
    # until one exists to support.
    reject_quantized_side_channel(all_keys, what)
    prefix = qwen3_5_key_prefix(all_keys)
    if prefix is None:
        raise ValueError(f"{what} is not a Qwen3.5 language model checkpoint.")
    names = _encoder_names(all_keys, prefix, final_norm=final_norm)

    sd: dict[str, torch.Tensor] = {}
    for path, keys in keys_by_file.items():
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                if key in names:
                    sd[names[key]] = f.get_tensor(key)

    with accelerate.init_empty_weights():
        model = Qwen35Encoder(text_config, final_norm=final_norm)

    if not final_norm:
        # Anima-3.8B's encoder ships its last layer without the MLP sub-block (and its norm), because
        # nothing it was used for reads past that layer's attention. Give such a layer an MLP that adds
        # nothing, so the layer computes exactly what the file encodes instead of loading half-empty.
        # A caller of the final norm runs every layer whole, so there a missing MLP stays a missing tensor.
        for index, layer in enumerate(model.layers):
            if not any(key.startswith(f"layers.{index}.mlp.") for key in sd):
                layer.mlp = _ZeroMLP()
                layer.post_attention_layernorm = nn.Identity()
                loader._logger.info(f"Qwen3.5 encoder: layer {index} ships without an MLP; running it attention-only.")

    model_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
    skip_patterns = _model_declared_skip_patterns(model)
    reserve_for_load(
        loader.ram_cache.make_room,
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
    log_unexpected_keys(what, result.unexpected_keys)
    reject_incomplete_load(model, what=what)
    sd.clear()

    if kept:
        loader._logger.info(f"Qwen3.5 encoder: kept {kept} raw fp8 weight(s) quantized ({loader._fp8_kept_reason()}).")
    return model.eval()


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

                # Single-file checkpoints ship no tokenizer. The vendored one serves the 4B and the 9B.
                return load_bundled_qwen3_5_tokenizer()

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: Qwen35Encoder_Checkpoint_Config) -> AnyModel:
        from invokeai.backend.qwen3_5.qwen3_5_encoder import qwen3_5_text_config

        model_path = Path(config.path)
        model = load_qwen3_5_text_encoder(
            self,
            [model_path],
            qwen3_5_text_config(config.variant),
            # MOSS-Image reads the 9B after its final norm. Anima-3.8B reads intermediate layers of the 4B,
            # and its checkpoint keeps a projection head under `norm.*` instead.
            final_norm=config.variant in QWEN3_5_FINAL_NORM_VARIANTS,
            keep_fp8=self._keep_fp8_weights(config, SubModelType.TextEncoder),
            what=f"Qwen3.5 encoder checkpoint {model_path.name}",
        )
        return self._apply_fp8_layerwise_casting(model, config, SubModelType.TextEncoder)
