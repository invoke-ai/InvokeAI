from pathlib import Path
from typing import Optional

import accelerate
import torch

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_QwenImage_Config,
    Main_GGUF_QwenImage_Config,
)
from invokeai.backend.model_manager.configs.qwen_vl_encoder import (
    QwenVLEncoder_Checkpoint_Config,
    QwenVLEncoder_Diffusers_Config,
    QwenVLEncoder_GGUF_Config,
)
from invokeai.backend.model_manager.load.load_default import (
    ModelLoader,
    _model_declared_skip_patterns,
)
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.qwen_image_controlnet_keys import is_qwen_image_instantx_controlnet
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    QwenImageVariantType,
    SubModelType,
)
from invokeai.backend.model_manager.util.llamacpp_keys import convert_llamacpp_decoder_keys
from invokeai.backend.model_manager.util.qwen3_vl import drop_qwen3vl_visual_tower
from invokeai.backend.quantization.fp8_scaled import (
    FP8_WEIGHT_DTYPES,
    Fp8ScaledLayer,
    attach_fp8_scales,
    cast_state_dict,
    dequantize_fp8_scaled,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    load_safetensors_checkpoint,
    mxfp8_scale_decode_host_bytes,
    parse_quantization_metadata,
    read_safetensors_metadata,
    safetensors_backend,
    split_fp8_scaled_layers,
    strip_layer_path_prefix,
    warn_on_unattached_scales,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, unpack_ggml_at_load
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.quantization.nvfp4 import (
    install_nvfp4_layers,
    pop_nvfp4_layers,
    predict_nvfp4_install_host_size,
    predict_nvfp4_install_size,
)
from invokeai.backend.qwen2_5_vl.qwen2_5_vl_assets import (
    load_bundled_qwen2_5_vl_config_dict,
    load_bundled_qwen2_5_vl_tokenizer,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.ram import require_ram_headroom
from invokeai.backend.util.state_dict_loading import (
    load_state_dict_ignoring_extras,
    log_unexpected_keys,
    reject_incomplete_load,
)


def _remap_qwen_vl_checkpoint_keys(sd: dict) -> dict:
    """Remap legacy ComfyUI Qwen2.5-VL single-file keys to the transformers layout.

    ComfyUI single-file checkpoints use the legacy Qwen2.5-VL key layout
    (`visual.X`, `model.X`); transformers ≥4.50 expects `model.visual.X` and
    `model.language_model.X`. This applies the same conversion mapping that
    `Qwen2_5_VLForConditionalGeneration.from_pretrained` would, since
    `load_state_dict` does not.

    transformers ≤4.x exposed this mapping as `_checkpoint_conversion_mapping`; 5.x
    emptied and then removed it, so we carry it ourselves. The negative lookahead keeps
    already-converted keys untouched, so the remap is safe (and idempotent) for both
    legacy and new-layout single-file checkpoints.
    """
    import re

    key_mapping = {
        r"^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }

    remapped_sd: dict = {}
    for old_key, tensor in sd.items():
        new_key = old_key
        if isinstance(old_key, str):
            for pattern, replacement in key_mapping.items():
                new_key, n_replace = re.subn(pattern, replacement, new_key)
                if n_replace > 0:
                    break
        remapped_sd[new_key] = tensor
    return remapped_sd


def _is_lm_head_name(name: object) -> bool:
    """Whether a single-file key or module path belongs to the LM head, in either spelling, sidecars included."""
    return isinstance(name, str) and name.removeprefix("model.").split(".", 1)[0] == "lm_head"


def _build_qwen_image_transformer_config(sd: dict, is_edit: bool) -> dict:
    """Auto-detect Qwen Image transformer architecture parameters from the state dict.

    Works for both GGUF (GGMLTensor) and plain safetensors (torch.Tensor) state dicts.
    Mutates nothing.
    """
    from diffusers import QwenImageTransformer2DModel

    def _shape(t):
        return t.tensor_shape if isinstance(t, GGMLTensor) else t.shape

    num_layers = 0
    for key in sd.keys():
        if isinstance(key, str) and key.startswith("transformer_blocks."):
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    num_layers = max(num_layers, int(parts[1]) + 1)
                except ValueError:
                    pass

    num_attention_heads = 24
    attention_head_dim = 128
    in_channels = 64

    if "img_in.weight" in sd:
        shape = _shape(sd["img_in.weight"])
        hidden_dim = shape[0]
        in_channels = shape[1]
        num_attention_heads = hidden_dim // attention_head_dim

    joint_attention_dim = 3584
    if "txt_in.weight" in sd:
        joint_attention_dim = _shape(sd["txt_in.weight"])[1]

    model_config: dict = {
        "patch_size": 2,
        "in_channels": in_channels,
        "out_channels": 16,
        "num_layers": num_layers if num_layers > 0 else 60,
        "attention_head_dim": attention_head_dim,
        "num_attention_heads": num_attention_heads,
        "joint_attention_dim": joint_attention_dim,
        "guidance_embeds": False,
        "axes_dims_rope": (16, 56, 56),
    }

    # zero_cond_t enables dual modulation for noisy vs reference patches in edit-variant
    # models. Setting it on txt2img models produces garbage. Requires diffusers 0.37+.
    import inspect

    if is_edit and "zero_cond_t" in inspect.signature(QwenImageTransformer2DModel.__init__).parameters:
        model_config["zero_cond_t"] = True

    return model_config


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class QwenImageDiffusersModel(GenericDiffusersLoader):
    """Class to load Qwen Image Edit main models."""

    _fixed_dtype = torch.bfloat16

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for Qwen Image Edit models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        if submodel_type is SubModelType.TextEncoder:
            from transformers import Qwen2_5_VLForConditionalGeneration

            from invokeai.backend.qwen2_5_vl.headless import HeadlessQwen2_5_VLForConditionalGeneration

            if load_class is Qwen2_5_VLForConditionalGeneration:
                load_class = HeadlessQwen2_5_VLForConditionalGeneration
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # We force bfloat16 for Qwen Image Edit models.
        # Use `dtype` (newer) with fallback to `torch_dtype` (older diffusers).
        dtype = self._decision(config, submodel_type).dtype
        dtype_kwarg = {"dtype": dtype}
        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                **dtype_kwarg,
                variant=variant,
                local_files_only=True,
            )
        except TypeError:
            # Older diffusers uses torch_dtype instead of dtype
            dtype_kwarg = {"torch_dtype": dtype}
            result = load_class.from_pretrained(
                model_path,
                **dtype_kwarg,
                variant=variant,
                local_files_only=True,
            )
        except OSError as e:
            if variant and "no file named" in str(e):
                result = load_class.from_pretrained(model_path, **dtype_kwarg, local_files_only=True)
            else:
                raise e

        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
        return result


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class QwenImageGGUFCheckpointModel(ModelLoader):
    """Class to load GGUF-quantized Qwen Image Edit transformer models."""

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

    def _load_from_singlefile(self, config: AnyModelConfig) -> AnyModel:
        from diffusers import QwenImageTransformer2DModel

        if not isinstance(config, Main_GGUF_QwenImage_Config):
            raise TypeError(f"Expected Main_GGUF_QwenImage_Config, got {type(config).__name__}.")
        model_path = Path(config.path)

        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = gguf_sd_loader(model_path, compute_dtype=compute_dtype)
        sd = CheckpointPrefix.detect(sd).strip(sd)

        is_edit = getattr(config, "variant", None) == QwenImageVariantType.Edit
        model_config = _build_qwen_image_transformer_config(sd, is_edit=is_edit)

        with accelerate.init_empty_weights():
            model = QwenImageTransformer2DModel(**model_config)

        load_state_dict_ignoring_extras(
            model, sd, source="Qwen-Image transformer checkpoint", assign=True, allow_missing=True
        )
        return model


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage, type=ModelType.Main, format=ModelFormat.Checkpoint)
class QwenImageCheckpointModel(ModelLoader):
    """Loads Qwen Image transformer models from single-file safetensors checkpoints
    (e.g. ComfyUI fp8_scaled, nvfp4 or int8_tensorwise, plain bf16/fp16). nvfp4 layers stay packed and
    int8_tensorwise(+convrot) layers stay int8 (`Int8ConvrotLinear`), both dequantized per forward. Scaled
    fp8 layers keep their fp8 weight and scale when fp8 compute is available or the model's
    `default_settings.fp8_storage` is on (which then also casts the dense remainder to fp8);
    otherwise they are dequantized to bf16 at load time. A file carries int8 or scaled fp8, not both."""

    _fp8_checkpoint_submodels = frozenset({SubModelType.Transformer})

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                model = self._load_from_singlefile(config)
                return self._apply_fp8_layerwise_casting(model, config, submodel_type)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(self, config: AnyModelConfig) -> AnyModel:
        from diffusers import QwenImageTransformer2DModel

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, Main_Checkpoint_QwenImage_Config):
            raise TypeError(f"Expected Main_Checkpoint_QwenImage_Config, got {type(config).__name__}.")
        model_path = Path(config.path)
        source_tensors_mapped = safetensors_backend(model_path) == "mmap"

        keep_checkpoint_fp8, model_dtype = self._storage_format(config, SubModelType.Transformer)

        sd = load_safetensors_checkpoint(str(model_path))
        sd = CheckpointPrefix.detect(sd).strip(sd)
        # Earlier versions identified InstantX's Qwen-Image ControlNet, a trimmed copy of this transformer, as a main
        # model. Such a record still exists; the file only needs identifying again.
        if is_qwen_image_instantx_controlnet(sd):
            raise ValueError(
                f"'{config.name}' is a Qwen Image ControlNet, not a main model: an earlier version of Invoke "
                "installed it as one. Run Re-identify model on it in the Model Manager to fix the record, then pick "
                "it in a control layer."
            )

        # Comfy's nvfp4 build keeps the image stream's attention and MLP in nvfp4, beside scaled fp8. Take those layers
        # out before the fold: `_dequantize_comfyui_fp8` multiplies every `.weight_scale` into its weight, nvfp4's
        # block scales included. `install_nvfp4_layers` puts them back, packed.
        header_layers = strip_layer_path_prefix(
            parse_quantization_metadata(read_safetensors_metadata(model_path, logger))
        )
        nvfp4_payloads = pop_nvfp4_layers(sd, header_layers=header_layers)

        # ComfyUI 'int8_tensorwise' (+convrot) shares the fp8 key layout (`.weight` beside `.weight_scale`), and the
        # fp8 extraction below pops every `.weight_scale`, so the int8 layers are claimed first. Outside the branch
        # on purpose: an int8 weight whose marker is missing would otherwise be cast as raw codes and generate noise.
        int8_markers = extract_int8_convrot_markers(sd, header_layers=header_layers)
        reject_unmarked_int8_weights(sd, int8_markers, "Qwen Image", header_layers=header_layers)

        if int8_markers:
            sd = drop_unconsumed_quantization_sidecars(sd)
            fp8_layers = {}
            # Nothing scaled to keep for fp8 storage; the int8 layers make the outer `_apply_fp8_layerwise_casting`
            # decline the model, so the dense remainder stays as loaded.
            use_fp8_storage = False
            keep_fp8 = False
        else:
            # ComfyUI 'scaled fp8': an fp8 weight plus its `weight_scale`, named in the header or by a per-layer marker.
            layer_hints = {**extract_comfy_quant_hints(sd), **header_layers}
            mxfp8_decode_host_bytes = mxfp8_scale_decode_host_bytes(sd, layer_hints)
            if mxfp8_decode_host_bytes:
                require_ram_headroom(
                    self._ram_cache.make_room(0, host_bytes_needed=mxfp8_decode_host_bytes), mxfp8_decode_host_bytes
                )
            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)
            # Kept fp8 only when something uses them: the fp8 matmul, or fp8 storage the user asked of this model. For
            # storage the checkpoint's own scale is exact, where the layerwise cast of a folded weight has none.
            # Without either, a dequantize per forward would cost speed for memory nobody asked to save, so they are
            # folded.
            use_fp8_storage = bool(fp8_layers) and self._should_use_fp8(config, SubModelType.Transformer)
            keep_fp8 = bool(fp8_layers) and keep_checkpoint_fp8

        is_edit = getattr(config, "variant", None) == QwenImageVariantType.Edit
        model_config = _build_qwen_image_transformer_config(sd, is_edit=is_edit)

        # Built before the reservation, which depends on its modules: they decide which nvfp4 and fp8 layers stay
        # quantized.
        with accelerate.init_empty_weights():
            model = QwenImageTransformer2DModel(**model_config)
        skip_patterns = _model_declared_skip_patterns(model)

        if int8_markers:
            # The checkpoint's keys already match diffusers' module tree, so the markers need no renaming. The one
            # reservation also covers the nvfp4 layers held beside the int8 ones.
            quantized = install_int8_convrot_layers(
                model,
                sd,
                int8_markers,
                model_dtype,
                architecture="Qwen Image",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
                extra_reserved_bytes=predict_nvfp4_install_size(model, nvfp4_payloads, model_dtype, skip_patterns),
                extra_host_bytes=predict_nvfp4_install_host_size(model, nvfp4_payloads, model_dtype, skip_patterns),
                source_tensors_mapped=source_tensors_mapped,
            )
            logger.info(
                f"Qwen Image: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                "(int8_tensorwise checkpoint, dequantized per forward)"
            )
        else:
            # One reservation, before the fold or the split widens a single weight: `make_room` makes that much room
            # rather than adding to an earlier one.
            reserve_for_load(
                self._ram_cache.make_room,
                sd,
                model_dtype,
                keep_fp8=keep_fp8,
                model=model,
                skip_patterns=skip_patterns,
                fp8_layers=fp8_layers,
                nvfp4_payloads=nvfp4_payloads,
                source_tensors_mapped=source_tensors_mapped,
            )

            if fp8_layers and not keep_fp8:
                dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
                logger.info(f"Qwen Image: folded {len(fp8_layers)} scaled fp8 layer(s) into {model_dtype}.")
                fp8_layers = {}
            # Layers the cast would widen anyway are folded here with their scale applied, so the cast never drops one.
            fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)
            cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)

        if nvfp4_payloads:
            packed = install_nvfp4_layers(model, sd, nvfp4_payloads, model_dtype, skip_patterns)
            logger.info(f"Qwen Image: kept {packed} of {len(nvfp4_payloads)} nvfp4 layer(s) packed.")

        load_state_dict_ignoring_extras(
            model, sd, source="Qwen-Image transformer checkpoint", assign=True, allow_missing=True
        )
        if int8_markers:
            # `allow_missing` lets a partial file through; a quantized layer that lost a tensor would otherwise fail
            # on the first forward with a meta-tensor error.
            reject_incomplete_load(model, what="Qwen Image int8 transformer checkpoint")
        # `assign=True` aliases every param to its `sd` tensor: without this, the fp8 storage cast below would hold
        # each dense weight at bf16 and fp8 at once, past the reservation.
        sd.clear()

        if fp8_layers:
            attached = attach_fp8_scales(model, fp8_layers)
            warn_on_unattached_scales(logger, "Qwen Image", attached, fp8_layers)
            if not use_fp8_storage:
                logger.info(f"Qwen Image: kept {attached} scaled fp8 layer(s) fp8 for fp8 compute.")
            else:
                # The rest of the dense weights go to fp8 storage too. The scaled layers are left alone: the cast
                # hooks would upcast them without their scale. Marking the model cast keeps
                # `_apply_fp8_layerwise_casting` from doing exactly that afterwards.
                self._apply_fp8_to_nn_module(
                    model,
                    storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=model_dtype,
                    extra_skip_patterns=skip_patterns,
                    skip=lambda _name, module: getattr(module, "weight_scale", None) is not None,
                )
                logger.info(
                    f"FP8 layerwise casting enabled for {config.name} (storage=float8_e4m3fn, compute={model_dtype}); "
                    f"kept {attached} scaled fp8 layer(s) with their own scale."
                )
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.QwenVLEncoder, format=ModelFormat.QwenVLEncoder)
class QwenVLEncoderLoader(ModelLoader):
    """Loads a standalone Qwen2.5-VL encoder (text_encoder/ + tokenizer/ + processor/)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, QwenVLEncoder_Diffusers_Config):
            raise TypeError(f"Expected QwenVLEncoder_Diffusers_Config, got {type(config).__name__}.")

        from transformers import AutoTokenizer

        from invokeai.backend.qwen2_5_vl.headless import HeadlessQwen2_5_VLForConditionalGeneration

        model_path = Path(config.path)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        match submodel_type:
            case SubModelType.Tokenizer:
                tokenizer_path = model_path / "tokenizer"
                return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
            case SubModelType.TextEncoder:
                encoder_path = model_path / "text_encoder"
                return HeadlessQwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(encoder_path),
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.QwenVLEncoder, format=ModelFormat.Checkpoint)
class QwenVLEncoderCheckpointLoader(ModelLoader):
    """Loads a single-file Qwen2.5-VL encoder checkpoint (e.g. ComfyUI fp8_scaled).

    The checkpoint bundles the language model and the visual tower into one safetensors file, so
    it carries neither the architecture config nor the tokenizer. Both are vendored in
    `invokeai.backend.qwen2_5_vl`, so this encoder loads with no network access.
    """

    _fp8_checkpoint_submodels = frozenset({SubModelType.TextEncoder})
    _encoder_fp8_storage = True

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, QwenVLEncoder_Checkpoint_Config):
            raise TypeError(f"Expected QwenVLEncoder_Checkpoint_Config, got {type(config).__name__}.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return load_bundled_qwen2_5_vl_tokenizer()
            case SubModelType.TextEncoder:
                return self._load_text_encoder_from_singlefile(config)

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder_from_singlefile(self, config: QwenVLEncoder_Checkpoint_Config) -> AnyModel:
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        from invokeai.backend.qwen2_5_vl.headless import OmittedLMHead
        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        model_path = Path(config.path)
        source_tensors_mapped = safetensors_backend(model_path) == "mmap"

        device_keeps_fp8, model_dtype = self._storage_format(config, SubModelType.TextEncoder)

        sd = load_safetensors_checkpoint(str(model_path))
        header_layers = parse_quantization_metadata(read_safetensors_metadata(model_path, logger))
        # The LM head is not loaded. Its weight, quantization sidecars and header entry go first, so that nothing below
        # folds, installs or reserves room for them.
        sd = {key: value for key, value in sd.items() if not _is_lm_head_name(key)}
        header_layers = {path: hint for path, hint in header_layers.items() if not _is_lm_head_name(path)}

        # Comfy's nvfp4 build keeps the language model's projections in nvfp4, beside scaled fp8 embeddings. Take those
        # layers out before the fp8 extraction: it pops every `.weight_scale`, nvfp4's block scales included.
        # `install_nvfp4_layers` puts them back, packed.
        nvfp4_payloads = pop_nvfp4_layers(sd, header_layers=header_layers)

        # ComfyUI 'int8_tensorwise' (+convrot) shares the fp8 key layout, and the fp8 extraction below treats
        # every `.weight_scale` as an fp8 scale, which for a rotated layer would leave a weight of the right shape and
        # no relation to the one stored. The int8 layers are claimed first and stay int8. Ahead of the reservation, so a
        # load that cannot finish does not evict the cache for room it will never use. The refusal is outside the
        # branch on purpose: an int8 weight with no marker is refused in every case.
        int8_markers = extract_int8_convrot_markers(sd, header_layers=header_layers)
        reject_unmarked_int8_weights(sd, int8_markers, "Qwen2.5-VL encoder", header_layers=header_layers)
        if int8_markers:
            sd = drop_unconsumed_quantization_sidecars(sd)

        # The architecture config is vendored rather than fetched: these constants decide the
        # module tree that the weights below are folded into, and a re-upload changing one of them
        # would silently change conditioning against unchanged weights.
        qwen_config = Qwen2_5_VLConfig.from_dict(load_bundled_qwen2_5_vl_config_dict())
        qwen_config.torch_dtype = model_dtype

        # Built before the reservation, which depends on its modules: they decide which nvfp4 layers stay packed.
        with accelerate.init_empty_weights():
            model = Qwen2_5_VLForConditionalGeneration(qwen_config)
        model.lm_head = OmittedLMHead()
        skip_patterns = _model_declared_skip_patterns(model)
        # The packed and int8 layers are named in the same legacy layout as the keys.
        nvfp4_payloads = _remap_qwen_vl_checkpoint_keys(nvfp4_payloads)
        int8_markers = _remap_qwen_vl_checkpoint_keys(int8_markers)

        # ComfyUI single-file checkpoints use the legacy Qwen2.5-VL key layout
        # (`visual.X`, `model.X`); remap to the `model.visual.X` / `model.language_model.X`
        # layout transformers expects. See `_remap_qwen_vl_checkpoint_keys` for details.
        sd = _remap_qwen_vl_checkpoint_keys(sd)

        fp8_layers: dict[str, Fp8ScaledLayer] = {}

        if int8_markers:
            # The dense tensors are cast inside, and the fp32 scales are left alone: a blanket cast would ruin them.
            quantized = install_int8_convrot_layers(
                model,
                sd,
                int8_markers,
                model_dtype,
                architecture="Qwen2.5-VL encoder",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
                extra_reserved_bytes=predict_nvfp4_install_size(model, nvfp4_payloads, model_dtype, skip_patterns),
                extra_host_bytes=predict_nvfp4_install_host_size(model, nvfp4_payloads, model_dtype, skip_patterns),
                source_tensors_mapped=source_tensors_mapped,
            )
            logger.info(
                f"Qwen VL encoder: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                "(int8_tensorwise checkpoint, dequantized per forward)"
            )
        else:
            # ComfyUI 'scaled fp8': an fp8 weight plus its `weight_scale`, named in the header or by a per-layer marker.
            # Extracted after the key remap so each scale travels with its weight and the recovered paths match the
            # model's modules. Pops every scale and marker, and refuses an unparsed scale spelling beside an fp8
            # weight rather than casting it unscaled.
            layer_hints = {**extract_comfy_quant_hints(sd), **_remap_qwen_vl_checkpoint_keys(header_layers)}
            mxfp8_decode_host_bytes = mxfp8_scale_decode_host_bytes(sd, layer_hints)
            if mxfp8_decode_host_bytes:
                require_ram_headroom(
                    self._ram_cache.make_room(0, host_bytes_needed=mxfp8_decode_host_bytes), mxfp8_decode_host_bytes
                )
            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)
            # Text encoders have no FP8 Storage setting, so the device decides (`_storage_format`), as for the other
            # encoders: float8 where the device can hold it or multiply in it, folded into the compute dtype elsewhere.
            source_is_fp8 = bool(fp8_layers) or any(
                getattr(tensor, "dtype", None) in FP8_WEIGHT_DTYPES for tensor in sd.values()
            )
            keep_fp8 = source_is_fp8 and device_keeps_fp8

            # One reservation, before the fold or the split widens a single weight: `make_room` makes that much room
            # rather than adding to an earlier one.
            reserve_for_load(
                self._ram_cache.make_room,
                sd,
                model_dtype,
                keep_fp8=keep_fp8,
                model=model,
                skip_patterns=skip_patterns,
                fp8_layers=fp8_layers,
                nvfp4_payloads=nvfp4_payloads,
                source_tensors_mapped=source_tensors_mapped,
            )
            if fp8_layers and not keep_fp8:
                dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
                logger.info(f"Qwen VL encoder: folded {len(fp8_layers)} scaled fp8 layer(s) into {model_dtype}.")
                fp8_layers = {}
            # Layers the cast would widen anyway (embeddings, block-wise scales) are folded here with their scale
            # applied, so the cast never drops one.
            fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)
            cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)

        if nvfp4_payloads:
            packed = install_nvfp4_layers(model, sd, nvfp4_payloads, model_dtype, skip_patterns)
            logger.info(f"Qwen VL encoder: kept {packed} of {len(nvfp4_payloads)} nvfp4 layer(s) packed.")

        # Load weights; allow missing keys for re-initialised buffers.
        load_result = model.load_state_dict(sd, strict=False, assign=True)
        log_unexpected_keys("Qwen2.5-VL text encoder checkpoint", load_result.unexpected_keys)

        # Re-initialise any leftover meta buffers (RoPE inv_freq etc.)
        for name, buffer in list(model.named_buffers()):
            if not buffer.is_meta:
                continue
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
                buffer_name = parts[1]
            else:
                parent = model
                buffer_name = name
            # Replace meta buffer with a real (zero) tensor of the same shape; the model
            # will recompute or refill these as needed at first forward pass.
            try:
                shape = buffer.shape
                parent.register_buffer(buffer_name, torch.zeros(shape, dtype=model_dtype), persistent=False)
            except Exception:
                logger.warning(f"Could not re-initialise meta buffer {name}")

        meta_params = [name for name, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Failed to load all parameters from checkpoint. Meta tensors remain: {meta_params[:5]}")

        if fp8_layers:
            # The scales go on the modules that kept their float8 weights; `CustomLinear` applies them per forward.
            attached = attach_fp8_scales(model, fp8_layers)
            warn_on_unattached_scales(logger, "Qwen VL encoder", attached, fp8_layers)
        # Storage only, as for T5: the fp8 matmul quantizes the activations too, and one encoder pass per prompt gains
        # nothing from it while outliers in the activations cost conditioning accuracy. Raw fp8 weights included.
        kept = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.weight.dtype in FP8_WEIGHT_DTYPES:
                module._fp8_full_precision_matmul = True
                kept += 1
        if kept:
            logger.info(f"Qwen VL encoder: kept {kept} Linear weight(s) in float8, dequantized per forward")

        model.eval()
        return model


# llama.cpp's name for the LM head. Conditioning reads hidden states and never runs it, so it is not read.
_GGUF_LM_HEAD_KEY = "output.weight"


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.QwenVLEncoder, format=ModelFormat.GGUFQuantized)
class QwenVLEncoderGGUFLoader(ModelLoader):
    """Loads a llama.cpp GGUF of the Qwen2.5-VL-7B language model as a text-only Qwen-Image encoder.

    Built as the same `Qwen2_5_VLForConditionalGeneration` the checkpoint loader builds, from the vendored config,
    so the prompt node runs one class for every format. The vision tower is dropped (llama.cpp keeps its weights in
    a separate mmproj file) and the LM head is never read. Linear weights stay packed and dequantize per forward;
    what a module reads outside a matmul, and what the file stores unquantized, is unpacked at load.
    """

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        # The tokenizer is vendored, not read from the file; sized at the file, it would evict cached models for room
        # it never uses.
        if submodel_type is SubModelType.Tokenizer:
            return 0
        return super().get_size_fs(config, model_path, submodel_type)

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, QwenVLEncoder_GGUF_Config):
            raise TypeError(f"Expected QwenVLEncoder_GGUF_Config, got {type(config).__name__}.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return load_bundled_qwen2_5_vl_tokenizer()
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: QwenVLEncoder_GGUF_Config) -> AnyModel:
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        from invokeai.backend.qwen2_5_vl.headless import OmittedLMHead

        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        qwen_config = Qwen2_5_VLConfig.from_dict(load_bundled_qwen2_5_vl_config_dict())

        sd = gguf_sd_loader(Path(config.path), compute_dtype=compute_dtype, keep=lambda name: name != _GGUF_LM_HEAD_KEY)
        # llama.cpp names to the transformers 4 layout, then on to the one transformers 5 builds. llama.cpp does not
        # permute Q/K for Qwen2-VL (it does for Llama), so the projections map as they are.
        sd = _remap_qwen_vl_checkpoint_keys(convert_llamacpp_decoder_keys(sd))

        with accelerate.init_empty_weights():
            model = Qwen2_5_VLForConditionalGeneration(qwen_config)
        drop_qwen3vl_visual_tower(model.model)
        model.lm_head = OmittedLMHead()

        unpacked = unpack_ggml_at_load(sd, model, _model_declared_skip_patterns(model), self._ram_cache.make_room)

        # Strict: with the vision tower and the LM head gone, every remaining parameter comes from the file, so a
        # missing one means an incomplete file and is named here rather than left on the meta device.
        load_state_dict_ignoring_extras(model, sd, source="Qwen2.5-VL encoder GGUF", assign=True)
        sd.clear()

        self._logger.info(
            f"Qwen2.5-VL encoder '{config.name}': loaded from GGUF, {unpacked} unquantized or non-Linear tensor(s) "
            f"unpacked, Linear weights kept packed (compute={compute_dtype}); vision tower and LM head omitted."
        )
        model.eval()
        return model
