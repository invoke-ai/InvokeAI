# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Anima model loading in InvokeAI."""

from pathlib import Path
from typing import Any, Optional

import accelerate

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_Anima_Config
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Anima_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.fp8_scaled import (
    attach_fp8_scales,
    cast_state_dict,
    dequantize_fp8_scaled,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    full_precision_hints_respected,
    parse_quantization_metadata,
    predict_cast_state_dict_size,
    read_safetensors_metadata,
    should_keep_fp8_weights,
    split_fp8_scaled_layers,
    warn_on_unattached_scales,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)


def _strip_anima_bundle_prefix(sd: dict) -> dict:
    """Strip the transformer-key prefix from an Anima single-file checkpoint.

    Handles both packaging formats:
      - Official format: keys prefixed with `net.` (e.g. `net.blocks.0...`)
      - ComfyUI bundled format: transformer keys prefixed with `model.diffusion_model.`
        alongside `first_stage_model.*` (VAE) and `cond_stage_model.*` (text encoder).

    Only keys under the detected prefix are kept; unrelated keys from bundled
    checkpoints (VAE, text encoder) are dropped. If no known prefix is present, the
    state dict is returned unchanged.
    """
    prefix_to_strip = None
    for prefix in ["model.diffusion_model.", "net."]:
        if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
            prefix_to_strip = prefix
            break

    if prefix_to_strip is None:
        return sd

    stripped_sd: dict = {}
    for key, value in sd.items():
        if isinstance(key, str) and key.startswith(prefix_to_strip):
            stripped_sd[key[len(prefix_to_strip) :]] = value
        # Skip non-transformer keys from bundled checkpoints (VAE, text encoder)
    return stripped_sd


# Checkpoint tensors that are not part of the transformer's in-memory state. Suffixes match
# derived buffers that the model regenerates at runtime (registered as non-persistent or
# recomputed locally); prefixes match metadata that export tools serialize alongside the
# weights (e.g. sampling schedules). Extend these tuples as new checkpoint variants surface.
_NON_MODEL_KEY_SUFFIXES = (
    ".inv_freq",
    "pos_embedder.dim_spatial_range",
    "pos_embedder.dim_temporal_range",
    "pos_embedder.seq",
)
_NON_MODEL_KEY_PREFIXES = ("model_sampling.",)


def _strip_anima_prefix_from_layer_paths(layer_names: Any) -> dict[str, str]:
    """Map checkpoint-scheme layer names to the paths the module tree actually uses.

    `_quantization_metadata` names its layers in the checkpoint's own scheme -- `net.`-prefixed on
    every Anima redistribution measured -- while the scales are read after `_strip_anima_bundle_prefix`
    has run. Without this the per-layer flags, `full_precision_matrix_mult` above all, match nothing
    and are silently ignored. Rather than restating the prefix list, the names are pushed through
    the real strip function and read back.
    """
    stripped = _strip_anima_bundle_prefix({f"{name}.weight": None for name in layer_names})
    return {name: key[: -len(".weight")] for name, key in zip(layer_names, stripped.keys(), strict=True)}


def _filter_non_model_keys(sd: dict) -> dict:
    """Drop checkpoint keys that don't belong to the transformer module's state dict."""
    return {
        k: v
        for k, v in sd.items()
        if not (k.endswith(_NON_MODEL_KEY_SUFFIXES) or k.startswith(_NON_MODEL_KEY_PREFIXES))
    }


@ModelLoaderRegistry.register(base=BaseModelType.Anima, type=ModelType.Main, format=ModelFormat.Checkpoint)
class AnimaCheckpointModel(ModelLoader):
    """Class to load Anima transformer models from single-file checkpoints.

    The Anima checkpoint contains both the MiniTrainDIT backbone and the LLM Adapter
    under a shared `net.` prefix. The loader strips this prefix and instantiates
    the AnimaTransformer model with the correct architecture parameters.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from safetensors.torch import load_file

        from invokeai.backend.anima.anima_transformer import AnimaTransformer

        if not isinstance(config, Main_Checkpoint_Anima_Config):
            raise TypeError(
                f"Expected Main_Checkpoint_Anima_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Load the state dict from safetensors
        sd = load_file(model_path)

        # Strip the transformer-key prefix (`net.` or bundled `model.diffusion_model.`).
        sd = _strip_anima_bundle_prefix(sd)

        # Drop runtime-derived buffers and exporter metadata that aren't model weights.
        sd = _filter_non_model_keys(sd)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_anima_inference_dtype(target_device)

        # ComfyUI 'scaled fp8': an fp8 weight plus a `weight_scale`. `_filter_non_model_keys` above
        # keeps those keys, and `load_state_dict` below rejects the checkpoint outright over them --
        # 500 unexpected keys on a plain scaled export, 749 on one that also ships `comfy_quant`
        # markers. Such a checkpoint therefore does not load at all today.
        #
        # Anima keeps `q_proj`/`k_proj`/`v_proj` separate and the only key rewrite is a prefix strip,
        # so a sibling scale travels with its weight and nothing has to be split.
        keep_fp8 = should_keep_fp8_weights(target_device)
        header_hints = parse_quantization_metadata(read_safetensors_metadata(model_path, logger))
        path_map = _strip_anima_prefix_from_layer_paths(list(header_hints))
        layer_hints = {
            **extract_comfy_quant_hints(sd),
            **{path_map.get(name, name): hints for name, hints in header_hints.items()},
        }
        fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)
        if fp8_layers and not keep_fp8:
            # Without the matmul, keeping them quantized would halve VRAM but dequantize on every
            # forward. Fold the scale into the weight instead.
            dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
            fp8_layers = {}

        # Create an empty AnimaTransformer with Anima's default architecture parameters
        with accelerate.init_empty_weights():
            model = AnimaTransformer(
                max_img_h=240,
                max_img_w=240,
                max_frames=1,
                in_channels=16,
                out_channels=16,
                patch_spatial=2,
                patch_temporal=1,
                concat_padding_mask=True,
                model_channels=2048,
                num_blocks=28,
                num_heads=16,
                mlp_ratio=4.0,
                crossattn_emb_channels=1024,
                pos_emb_cls="rope3d",
                # Anima reuses the Cosmos-Predict2 2B Text2Image DiT, which trains with
                # rope_scale=(t=1.0, h=4.0, w=4.0). The NTK-scaled spatial RoPE base is
                # mandatory; omitting it (theta=10000 on all axes) shifts every step's
                # velocity ~7% off and compounds into degraded images. Matches diffusers
                # CosmosTransformer3DModel rope_scale via *_extrapolation_ratio.
                rope_h_extrapolation_ratio=4.0,
                rope_w_extrapolation_ratio=4.0,
                rope_t_extrapolation_ratio=1.0,
                use_adaln_lora=True,
                adaln_lora_dim=256,
                extra_per_block_abs_pos_emb=False,
                image_model="anima",
            )

        skip_patterns = tuple(getattr(model, "_skip_layerwise_casting_patterns", None) or ())
        # Layers the cast would dequantize anyway are folded here, scale applied, so the cast never
        # strips a scale that can no longer be put back.
        fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)

        self._ram_cache.make_room(
            predict_cast_state_dict_size(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)
        )
        kept = cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)

        load_result = model.load_state_dict(sd, assign=True, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint contains {len(load_result.unexpected_keys)} unexpected keys. "
                f"This may indicate a corrupted or incompatible checkpoint. "
                f"First 5 unexpected keys: {load_result.unexpected_keys[:5]}"
            )
        if load_result.missing_keys:
            logger.warning(
                f"Checkpoint is missing {len(load_result.missing_keys)} keys "
                f"(expected for inv_freq buffers). First 5: {load_result.missing_keys[:5]}"
            )

        # Without this the `fp8_storage` toggle is shown for Anima models but does nothing. The
        # state dict was cast to a single `model_dtype` above, so the layerwise cast has one
        # unambiguous compute dtype to restore to. AnimaTransformer is a plain nn.Module, so this
        # takes the hook-based path in `_apply_fp8_to_nn_module`.
        if fp8_layers:
            attached = attach_fp8_scales(model, fp8_layers)
            logger.info(f"Anima: kept {attached} layer(s) in fp8 (scaled fp8 checkpoint, fp8_compute enabled)")
            warn_on_unattached_scales(logger, "Anima", attached, fp8_layers)
            marked = sum(1 for layer in fp8_layers.values() if layer.full_precision_matmul)
            if marked and full_precision_hints_respected():
                logger.info(
                    f"Anima: {marked} of {len(fp8_layers)} layer(s) are marked full_precision_matrix_mult "
                    "and will dequantize per forward."
                )
        elif kept:
            logger.info(f"Anima: kept {kept} raw fp8 weight(s) quantized for the fp8 tensor cores.")

        model = self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Anima, type=ModelType.ControlNet, format=ModelFormat.Checkpoint)
class AnimaControlNetLLLiteModel(ModelLoader):
    """Class to load Anima ControlNet-LLLite adapter models from safetensors checkpoints.

    LLLite adapters are standalone files holding a shared conditioning trunk
    (lllite_conditioning1) plus tiny per-Linear modules (lllite_dit_blocks_*).
    Hyperparameters are stored in the safetensors metadata (`lllite.*` keys) with
    state-dict-shape fallbacks.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from safetensors import safe_open
        from safetensors.torch import load_file

        from invokeai.backend.anima.control_net_lllite import AnimaControlNetLLLite

        if not isinstance(config, ControlNet_Checkpoint_Anima_Config):
            raise ValueError("Only ControlNet_Checkpoint_Anima_Config models are supported here.")

        # ControlNet type models don't use submodel_type - load the adapter directly
        model_path = Path(config.path)

        sd = load_file(model_path)
        with safe_open(model_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()

        model = AnimaControlNetLLLite.from_state_dict(sd, metadata)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_anima_inference_dtype(target_device)
        model.to(dtype=model_dtype)

        return model
