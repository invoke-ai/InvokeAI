# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Anima model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import accelerate

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_Anima_Config
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Anima_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
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
    read_safetensors_metadata,
    reject_quantized_side_channel,
    split_fp8_scaled_layers,
    strip_layer_path_prefix,
    warn_on_unattached_scales,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.state_dict_loading import log_unexpected_keys, reject_incomplete_load

logger = InvokeAILogger.get_logger(__name__)


#: Anima's own wrapper namespaces. `net.` is the official packaging; the ComfyUI bundle uses the
#: usual `model.diffusion_model.`. `diffusion_model.` is deliberately absent -- no Anima
#: redistribution measured uses it, and `net.` is not stripped anywhere else because it is also the
#: root module name of a PiD checkpoint.
ANIMA_KEY_PREFIXES = ("model.diffusion_model.", "net.")


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
    return CheckpointPrefix.detect(sd, ANIMA_KEY_PREFIXES).strip(sd, drop_foreign=True)


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


def _filter_non_model_keys(sd: dict) -> dict:
    """Drop checkpoint keys that don't belong to the transformer module's state dict."""
    return {
        k: v
        for k, v in sd.items()
        if not (k.endswith(_NON_MODEL_KEY_SUFFIXES) or k.startswith(_NON_MODEL_KEY_PREFIXES))
    }


# Anima's fixed transformer architecture. Kept at module level so tests can instantiate the real
# module graph (e.g. to pin `_skip_layerwise_casting_patterns` to actual dotted module paths)
# without duplicating these values.
ANIMA_TRANSFORMER_CONFIG = {
    "max_img_h": 240,
    "max_img_w": 240,
    "max_frames": 1,
    "in_channels": 16,
    "out_channels": 16,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "concat_padding_mask": True,
    "model_channels": 2048,
    "num_blocks": 28,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "crossattn_emb_channels": 1024,
    "pos_emb_cls": "rope3d",
    # Anima reuses the Cosmos-Predict2 2B Text2Image DiT, which trains with
    # rope_scale=(t=1.0, h=4.0, w=4.0). The NTK-scaled spatial RoPE base is mandatory; omitting it
    # (theta=10000 on all axes) shifts every step's velocity ~7% off and compounds into degraded
    # images. Matches diffusers CosmosTransformer3DModel rope_scale via *_extrapolation_ratio.
    "rope_h_extrapolation_ratio": 4.0,
    "rope_w_extrapolation_ratio": 4.0,
    "rope_t_extrapolation_ratio": 1.0,
    "use_adaln_lora": True,
    "adaln_lora_dim": 256,
    "extra_per_block_abs_pos_emb": False,
    "image_model": "anima",
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
        keep_fp8 = self._keep_fp8_weights(config, SubModelType.Transformer)
        header_hints = parse_quantization_metadata(read_safetensors_metadata(model_path, logger))
        # The header names layers in the checkpoint's own scheme -- `net.`-prefixed on every Anima
        # redistribution measured -- while the scales are read after `_strip_anima_bundle_prefix`
        # has run. Without this the per-layer flags, `full_precision_matrix_mult` above all, match
        # nothing and are silently ignored.
        layer_hints = {
            **extract_comfy_quant_hints(sd),
            **strip_layer_path_prefix(header_hints),
        }
        fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)

        # Create an empty AnimaTransformer with Anima's default architecture parameters
        with accelerate.init_empty_weights():
            model = AnimaTransformer(**ANIMA_TRANSFORMER_CONFIG)

        skip_patterns = _model_declared_skip_patterns(model)
        # Reserve before anything below widens a weight -- the fold and the split both do, and
        # reserving afterwards lets either peak land on a cache that was only ever sized for the
        # file. `scaled_layers` is what keeps the prediction honest where the weights are kept: the
        # split also widens layers whose scale layout `scaled_mm` cannot apply, and without the
        # mapping the prediction would charge those 1 byte/element and arrive at 2. Where they are
        # not kept the prediction charges every float at `model_dtype`, folded yet or not, so the
        # number is the same on either side of the fold -- what changes is when the room exists.
        # Building the model first costs nothing: `init_empty_weights` leaves every param on meta.
        reserve_for_load(
            self._ram_cache.make_room,
            sd,
            model_dtype,
            keep_fp8=keep_fp8,
            model=model,
            skip_patterns=skip_patterns,
            fp8_layers=fp8_layers,
            nvfp4_payloads={},
        )

        if fp8_layers and not keep_fp8:
            # Neither the matmul nor FP8 Storage asked for them, so keeping them quantized would
            # dequantize on every forward to save memory nobody wanted saved. Fold the scale in.
            dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
            fp8_layers = {}

        # Layers the cast would dequantize anyway are folded here too, scale applied, so the cast
        # never strips a scale that can no longer be put back.
        fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)
        kept = cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)

        load_result = model.load_state_dict(sd, assign=True, strict=False)
        log_unexpected_keys("Anima transformer checkpoint", load_result.unexpected_keys)
        # `missing_keys` alone cannot police completeness here: AnimaTransformer's only three buffers
        # are registered `persistent=False`, so they never appear in it (the old warning claiming
        # otherwise was misleading). Sweep for tensors the checkpoint left on the meta device instead
        # — that is the failure worth catching, and it is what the removed unexpected-key
        # `RuntimeError` was really standing in for.
        reject_incomplete_load(model, what="Anima transformer checkpoint")

        # Without this the `fp8_storage` toggle is shown for Anima models but does nothing. When
        # nothing stayed packed, the state dict was cast to a single `model_dtype` above, so the
        # layerwise cast has one unambiguous compute dtype to restore to; when something did stay
        # packed, the cast bails out on its own rather than upcast a scaled weight without applying
        # its scale. AnimaTransformer is a plain nn.Module, so this takes the hook-based path in
        # `_apply_fp8_to_nn_module`.
        if fp8_layers:
            attached = attach_fp8_scales(model, fp8_layers)
            logger.info(
                f"Anima: kept {attached} layer(s) in fp8 (scaled fp8 checkpoint, kept for {self._fp8_kept_reason()})"
            )
            warn_on_unattached_scales(logger, "Anima", attached, fp8_layers)
            marked = sum(1 for layer in fp8_layers.values() if layer.full_precision_matmul)
            if marked and full_precision_hints_respected():
                logger.info(
                    f"Anima: {marked} of {len(fp8_layers)} layer(s) are marked full_precision_matrix_mult "
                    "and will dequantize per forward."
                )
        elif kept:
            logger.info(f"Anima: kept {kept} raw fp8 weight(s) quantized ({self._fp8_kept_reason()}).")

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
        reject_quantized_side_channel(sd, f"Anima ControlNet checkpoint {model_path.name}")
        with safe_open(model_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()

        model = AnimaControlNetLLLite.from_state_dict(sd, metadata)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_anima_inference_dtype(target_device)
        model.to(dtype=model_dtype)

        return model
