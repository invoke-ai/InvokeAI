"""Class for Anima model loading in InvokeAI."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import accelerate

from invokeai.backend.anima.checkpoint_layout import (
    ANIMA_KEY_PREFIXES,
    anima_transformer_config,
    filter_non_model_keys,
    strip_anima_bundle_prefix,
)
from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_Anima_Config
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    ANIMA_V2_CONNECTOR_KEY_PREFIX,
    Main_Checkpoint_Anima_Config,
    Main_GGUF_Anima_Config,
)
from invokeai.backend.model_manager.load.fp8_capability import NotApplicable
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
    Fp8ScaledLayer,
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
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, gguf_tensor_bytes, unpack_ggml_at_load
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
    resolve_quantized_module_paths,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.state_dict_loading import (
    load_state_dict_ignoring_extras,
    log_unexpected_keys,
    reject_incomplete_load,
)

logger = InvokeAILogger.get_logger(__name__)


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
        from invokeai.backend.anima.semantic_connector import AnimaSemanticConnectorConfig

        if not isinstance(config, Main_Checkpoint_Anima_Config):
            raise TypeError(
                f"Expected Main_Checkpoint_Anima_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)
        # Releases before `Main_GGUF_Anima_Config` identified GGUF builds as checkpoints, and their records
        # still say so. Read as safetensors, such a file fails on a header error that names neither cause.
        if model_path.suffix.lower() == ".gguf":
            raise ValueError(
                f"{model_path.name} is a GGUF file recorded as an Anima checkpoint by an earlier release. "
                "Use 'Re-identify model' in the Model Manager to load it as a GGUF model."
            )

        # Load the state dict from safetensors
        sd = load_file(model_path)

        # Strip the transformer-key prefix (`net.` or bundled `model.diffusion_model.`).
        sd = strip_anima_bundle_prefix(sd)

        # Drop runtime-derived buffers and exporter metadata that aren't model weights.
        sd = filter_non_model_keys(sd)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_anima_inference_dtype(target_device)

        # Two ComfyUI side-channel formats share the `weight` + `weight_scale` layout, and a checkpoint
        # carries one or the other (Anima-2.9B ships an `int8_tensorwise` build beside its bf16 one).
        # The int8 markers come out first: the fp8 path below reads `.comfy_quant` as layer hints and
        # would take an int8 weight for an unscaled one. The rejection sits outside the branch so an
        # int8 weight whose marker is missing or unreadable is refused instead of cast as raw codes.
        int8_markers = extract_int8_convrot_markers(sd)
        reject_unmarked_int8_weights(sd, int8_markers, "Anima")

        metadata = read_safetensors_metadata(model_path, logger)
        # Anima-3.8B bundles a semantic connector whose hyperparameters live in the header. Without
        # it in the model the connector's tensors would fall to `strict=False` and the 52-block DiT
        # would generate from the native adapter alone, which its new blocks were not trained on.
        semantic_connector = None
        if any(key.startswith(ANIMA_V2_CONNECTOR_KEY_PREFIX) for key in sd):
            semantic_connector = AnimaSemanticConnectorConfig.from_metadata(metadata)

        keep_fp8 = False
        fp8_layers: dict[str, Fp8ScaledLayer] = {}
        if int8_markers:
            sd = drop_unconsumed_quantization_sidecars(sd)
            # The prefix strip is the only key rewrite, so the markers already name the model's modules.
            quantized = resolve_quantized_module_paths(int8_markers, key_map={})
        else:
            # ComfyUI 'scaled fp8': an fp8 weight plus a `weight_scale`. `filter_non_model_keys` above
            # keeps those keys, and `load_state_dict` below rejects the checkpoint outright over them --
            # 500 unexpected keys on a plain scaled export, 749 on one that also ships `comfy_quant`
            # markers. Such a checkpoint therefore does not load at all today.
            #
            # Anima keeps `q_proj`/`k_proj`/`v_proj` separate and the only key rewrite is a prefix strip,
            # so a sibling scale travels with its weight and nothing has to be split.
            keep_fp8 = self._keep_fp8_weights(config, SubModelType.Transformer)
            header_hints = parse_quantization_metadata(metadata)
            # The header names layers in the checkpoint's own scheme -- `net.`-prefixed on every Anima
            # redistribution measured -- while the scales are read after `strip_anima_bundle_prefix`
            # has run. Without this the per-layer flags, `full_precision_matrix_mult` above all, match
            # nothing and are silently ignored.
            layer_hints = {
                **extract_comfy_quant_hints(sd),
                **strip_layer_path_prefix(header_hints),
            }
            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)

        # Build at the checkpoint's own depth: 28 blocks for the official release, more for the
        # depth-expanded finetunes (Anima-2.9B: 40, Anima-3.8B: 52).
        with accelerate.init_empty_weights():
            model = AnimaTransformer(**anima_transformer_config(sd), semantic_connector=semantic_connector)
        if semantic_connector is not None:
            logger.info(f"Anima: {len(model.blocks)}-block transformer with a bundled Qwen3.5 semantic connector")

        skip_patterns = _model_declared_skip_patterns(model)
        if int8_markers:
            quantized = install_int8_convrot_layers(
                model,
                sd,
                quantized,
                model_dtype,
                architecture="Anima",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
            )
            logger.info(
                f"Anima: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                "(int8_tensorwise checkpoint, dequantized per forward)"
            )
            kept = 0
        else:
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


@ModelLoaderRegistry.register(base=BaseModelType.Anima, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class AnimaGGUFModel(ModelLoader):
    """Loads an Anima transformer from a GGUF file.

    The community builds use the checkpoints' wrappers -- bare keys (ComfyUI-GGUF), `net.`, and
    stable-diffusion.cpp's all-in-one `model.diffusion_model.`, which bundles a Qwen3 text encoder and a VAE
    beside the transformer. Those two are skipped before their data is read: the model loader node supplies
    both, as for a checkpoint. The Linear weights stay packed and dequantize per forward; what the model reads
    outside a matmul (the norms, the LLM adapter's token embedding), the modules it declares precision-sensitive
    and what is not quantized at all are unpacked here (see `unpack_ggml_at_load`).

    No `fp8_storage` declaration: FP8 Storage is refused for every quantized format by rule.
    """

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """The bytes of the transformer's tensors: an all-in-one's encoder and VAE are never read."""
        sizes = gguf_tensor_bytes(model_path)
        prefix = _transformer_prefix(sizes)
        return sum(size for name, size in sizes.items() if name.startswith(prefix.prefix))

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from invokeai.backend.anima.anima_transformer import AnimaTransformer

        if not isinstance(config, Main_GGUF_Anima_Config):
            raise TypeError(f"Expected Main_GGUF_Anima_Config, got {type(config).__name__}.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                "An Anima GGUF holds only the transformer. "
                f"Received: {submodel_type.value if submodel_type else 'None'}"
            )

        path = Path(config.path)
        compute_dtype = TorchDevice.choose_anima_inference_dtype(TorchDevice.choose_torch_device())

        # The wrapper is read off the header before any data, so that only the transformer's tensors are copied
        # out of the file: an all-in-one's encoder and VAE would cost their size in RAM to be dropped.
        prefix = _transformer_prefix(gguf_tensor_bytes(path))
        keep = (lambda name: name.startswith(prefix.prefix)) if prefix.prefix else None
        sd = filter_non_model_keys(prefix.strip(gguf_sd_loader(path, compute_dtype, keep=keep)))

        with accelerate.init_empty_weights():
            model = AnimaTransformer(**anima_transformer_config(sd))

        unpacked = unpack_ggml_at_load(sd, model, _model_declared_skip_patterns(model), self._ram_cache.make_room)
        load_state_dict_ignoring_extras(model, sd, source="Anima GGUF transformer", assign=True, allow_missing=True)
        # The model's only buffers are non-persistent, so `missing_keys` cannot see an unfilled tensor; this can.
        reject_incomplete_load(model, what="Anima GGUF transformer")

        self._logger.info(
            f"Anima: loaded a {len(model.blocks)}-block GGUF transformer, {unpacked} unquantized or non-Linear "
            "tensor(s) unpacked at load, the remaining Linear weights kept packed"
        )
        return model


def _transformer_prefix(tensor_names: Mapping[str, Any]) -> CheckpointPrefix:
    """The wrapper an Anima GGUF's transformer tensors sit under, read off the names alone."""
    return CheckpointPrefix.detect(tensor_names, ANIMA_KEY_PREFIXES)


@ModelLoaderRegistry.register(
    base=BaseModelType.Anima,
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
    fp8_storage=NotApplicable(
        "an LLLite adapter is 8-66 MB (every one in the starter catalog), so the saving is tens of "
        "megabytes bought with an upcast on each of its per-Linear modules, every forward"
    ),
)
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
