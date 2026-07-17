from pathlib import Path
from typing import Optional

import accelerate
import torch

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_QwenImage_Config,
    Main_GGUF_QwenImage_Config,
)
from invokeai.backend.model_manager.configs.qwen_vl_encoder import (
    QwenVLEncoder_Checkpoint_Config,
    QwenVLEncoder_Diffusers_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    QwenImageVariantType,
    SubModelType,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.util.devices import TorchDevice


def _strip_comfyui_prefix(sd: dict) -> dict:
    """Strip ComfyUI-style `model.diffusion_model.` / `diffusion_model.` prefixes from keys."""
    prefix_to_strip = None
    for prefix in ["model.diffusion_model.", "diffusion_model."]:
        if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
            prefix_to_strip = prefix
            break
    if prefix_to_strip is None:
        return sd
    stripped: dict = {}
    for key, value in sd.items():
        if isinstance(key, str) and key.startswith(prefix_to_strip):
            stripped[key[len(prefix_to_strip) :]] = value
        else:
            stripped[key] = value
    return stripped


def _dequantize_comfyui_fp8(sd: dict, compute_dtype: torch.dtype) -> int:
    """Dequantize ComfyUI-style fp8_scaled weights in-place. Returns count of dequantized tensors.

    Weights are dequantized directly to `compute_dtype` (typically bf16) instead of via a
    full-precision float32 intermediate. The previous float32 path materialised a complete
    4-byte/param copy of the model before a separate downcast pass, spiking peak RAM to ~2x the
    final bf16 size (~80GB for the 20B Qwen-Image transformer). Multiplying in the target dtype
    keeps the dict at the bf16 model size plus a single transient tensor. fp8 has only 3 mantissa
    bits and bf16 shares float32's exponent range, so the bf16 multiply loses no meaningful
    precision here.

    Two key naming schemes are in the wild:
      - `<path>.weight` + `<path>.weight_scale`  (FLUX, Z-Image style)
      - `<path>.weight` + `<path>.scale_weight`  (Qwen2.5-VL fp8_scaled style, also
        emits `<path>.scale_input` for activation scaling that we discard).
    """
    scale_suffixes = (".weight_scale", ".scale_weight")
    weight_scale_keys = [k for k in sd.keys() if isinstance(k, str) and k.endswith(scale_suffixes)]
    count = 0
    for scale_key in weight_scale_keys:
        for suffix in scale_suffixes:
            if scale_key.endswith(suffix):
                weight_key = scale_key[: -len(suffix)] + ".weight"
                break
        if weight_key not in sd:
            continue
        weight = sd[weight_key].to(compute_dtype)
        scale = sd[scale_key].to(compute_dtype)
        if scale.shape != weight.shape and scale.numel() > 1:
            for dim in range(len(weight.shape)):
                if dim < len(scale.shape) and scale.shape[dim] != weight.shape[dim]:
                    block_size = weight.shape[dim] // scale.shape[dim]
                    if block_size > 1:
                        scale = scale.repeat_interleave(block_size, dim=dim)
        sd[weight_key] = weight * scale
        count += 1
    return count


def _remap_qwen_vl_checkpoint_keys(sd: dict) -> dict:
    """Remap legacy ComfyUI Qwen2.5-VL single-file keys to the transformers layout.

    ComfyUI single-file checkpoints use the legacy Qwen2.5-VL key layout
    (`visual.X`, `model.X`); transformers ≥4.50 expects `model.visual.X` and
    `model.language_model.X`. This applies the same conversion mapping that
    `Qwen2_5_VLForConditionalGeneration.from_pretrained` would, since
    `load_state_dict` does not.

    transformers ≤4.x exposed this as `_checkpoint_conversion_mapping`, but 5.x
    dropped it (returns `{}`), so we fall back to the legacy mapping ourselves. The
    negative lookahead keeps already-converted keys untouched, so the remap is safe
    (and idempotent) for both legacy and new-layout single-file checkpoints.
    """
    import re

    from transformers import Qwen2_5_VLForConditionalGeneration

    key_mapping = Qwen2_5_VLForConditionalGeneration._checkpoint_conversion_mapping or {
        r"^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    if not key_mapping:
        return sd

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


def _strip_quantization_metadata(sd: dict) -> None:
    """Strip ComfyUI fp8 quantization metadata keys in-place."""
    keys_to_drop = [
        k
        for k in sd.keys()
        if isinstance(k, str)
        and (
            k.endswith(".weight_scale")
            or k.endswith(".scale_weight")
            or k.endswith(".scale_input")
            or "comfy_quant" in k
            or k == "scaled_fp8"
        )
    ]
    for k in keys_to_drop:
        del sd[k]


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
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # We force bfloat16 for Qwen Image Edit models.
        # Use `dtype` (newer) with fallback to `torch_dtype` (older diffusers).
        dtype_kwarg = {"dtype": torch.bfloat16}
        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                **dtype_kwarg,
                variant=variant,
                local_files_only=True,
            )
        except TypeError:
            # Older diffusers uses torch_dtype instead of dtype
            dtype_kwarg = {"torch_dtype": torch.bfloat16}
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
        sd = _strip_comfyui_prefix(sd)

        is_edit = getattr(config, "variant", None) == QwenImageVariantType.Edit
        model_config = _build_qwen_image_transformer_config(sd, is_edit=is_edit)

        with accelerate.init_empty_weights():
            model = QwenImageTransformer2DModel(**model_config)

        model.load_state_dict(sd, strict=False, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage, type=ModelType.Main, format=ModelFormat.Checkpoint)
class QwenImageCheckpointModel(ModelLoader):
    """Loads Qwen Image transformer models from single-file safetensors checkpoints
    (e.g. ComfyUI fp8_scaled, plain bf16/fp16). Dequantizes ComfyUI fp8 scaling to
    bf16 at load time; the `default_settings.fp8_storage` toggle then optionally
    re-casts to fp8 for VRAM savings."""

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
        from safetensors.torch import load_file

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, Main_Checkpoint_QwenImage_Config):
            raise TypeError(f"Expected Main_Checkpoint_QwenImage_Config, got {type(config).__name__}.")
        model_path = Path(config.path)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(str(model_path))
        sd = _strip_comfyui_prefix(sd)

        dequantized = _dequantize_comfyui_fp8(sd, model_dtype)
        if dequantized > 0:
            logger.info(f"Dequantized {dequantized} ComfyUI-quantized weights")
        _strip_quantization_metadata(sd)

        is_edit = getattr(config, "variant", None) == QwenImageVariantType.Edit
        model_config = _build_qwen_image_transformer_config(sd, is_edit=is_edit)

        with accelerate.init_empty_weights():
            model = QwenImageTransformer2DModel(**model_config)

        # Dequantized fp8 weights are already at model_dtype; this only casts any remaining
        # non-quantized float weights (e.g. a plain fp16/fp32 checkpoint) to the compute dtype
        # so the cache reservation below is sized from the actual post-cast tensors.
        for k in list(sd.keys()):
            if sd[k].is_floating_point():
                sd[k] = sd[k].to(model_dtype)

        new_sd_size = sum(t.nelement() * t.element_size() for t in sd.values())
        self._ram_cache.make_room(new_sd_size)

        model.load_state_dict(sd, strict=False, assign=True)
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

        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        model_path = Path(config.path)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        match submodel_type:
            case SubModelType.Tokenizer:
                tokenizer_path = model_path / "tokenizer"
                return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
            case SubModelType.TextEncoder:
                encoder_path = model_path / "text_encoder"
                return Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

    The checkpoint bundles the language model and the visual tower into one
    safetensors file. Tokenizer + processor are pulled from HuggingFace
    (`Qwen/Qwen2.5-VL-7B-Instruct`) on first use, with offline cache fallback.
    """

    DEFAULT_HF_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, QwenVLEncoder_Checkpoint_Config):
            raise TypeError(f"Expected QwenVLEncoder_Checkpoint_Config, got {type(config).__name__}.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return self._load_tokenizer_with_offline_fallback()
            case SubModelType.TextEncoder:
                return self._load_text_encoder_from_singlefile(config)

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_tokenizer_with_offline_fallback(self) -> AnyModel:
        from transformers import AutoTokenizer

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        try:
            return AutoTokenizer.from_pretrained(self.DEFAULT_HF_REPO, local_files_only=True)
        except OSError:
            logger.info(
                f"Tokenizer for single-file Qwen VL encoder not found in HuggingFace cache; "
                f"downloading from {self.DEFAULT_HF_REPO} (one-time, requires network access)."
            )
            try:
                return AutoTokenizer.from_pretrained(self.DEFAULT_HF_REPO)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to load Qwen VL tokenizer. Single-file Qwen VL encoder checkpoints do not "
                    f"include the tokenizer; it must be downloaded from HuggingFace ({self.DEFAULT_HF_REPO}) "
                    f"on first use. Either restore network access, or install the encoder in the "
                    f"diffusers folder layout (text_encoder/ + tokenizer/) instead. Original error: {e}"
                ) from e

    def _load_text_encoder_from_singlefile(self, config: QwenVLEncoder_Checkpoint_Config) -> AnyModel:
        from safetensors.torch import load_file
        from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        model_path = Path(config.path)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(str(model_path))

        # Dequantize ComfyUI-style fp8 weights, then strip the now-unused quantization
        # metadata (`scale_input` is the activation scale ComfyUI's fp8 matmul kernels
        # use at runtime — we run the encoder in bf16 after dequantization).
        dequantized_count = _dequantize_comfyui_fp8(sd, model_dtype)
        if dequantized_count > 0:
            logger.info(f"Dequantized {dequantized_count} ComfyUI-quantized weights")
        _strip_quantization_metadata(sd)

        # ComfyUI single-file checkpoints use the legacy Qwen2.5-VL key layout
        # (`visual.X`, `model.X`); remap to the `model.visual.X` / `model.language_model.X`
        # layout transformers expects. See `_remap_qwen_vl_checkpoint_keys` for details.
        sd = _remap_qwen_vl_checkpoint_keys(sd)

        # Cast to compute dtype (skip integer/index tensors)
        for k in list(sd.keys()):
            if sd[k].is_floating_point():
                sd[k] = sd[k].to(model_dtype)

        # Fetch the architecture config from HuggingFace (small, ~5KB).
        # Offline fallback: tries cache first, downloads only if missing.
        try:
            qwen_config = AutoConfig.from_pretrained(self.DEFAULT_HF_REPO, local_files_only=True)
        except OSError:
            logger.info(
                f"Architecture config for single-file Qwen VL encoder not found in HuggingFace cache; "
                f"downloading from {self.DEFAULT_HF_REPO} (one-time, ~5KB, requires network access)."
            )
            try:
                qwen_config = AutoConfig.from_pretrained(self.DEFAULT_HF_REPO)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to load Qwen VL architecture config. Single-file Qwen VL encoder checkpoints "
                    f"do not include the model config; it must be downloaded from HuggingFace "
                    f"({self.DEFAULT_HF_REPO}) on first use. Either restore network access, or install the "
                    f"encoder in the diffusers folder layout (text_encoder/config.json + tokenizer/) "
                    f"instead. Original error: {e}"
                ) from e
        qwen_config.torch_dtype = model_dtype

        new_sd_size = sum(t.nelement() * t.element_size() for t in sd.values())
        self._ram_cache.make_room(new_sd_size)

        with accelerate.init_empty_weights():
            model = Qwen2_5_VLForConditionalGeneration(qwen_config)

        # Load weights; allow missing keys for tied lm_head and re-initialised buffers.
        load_result = model.load_state_dict(sd, strict=False, assign=True)
        if load_result.unexpected_keys:
            logger.warning(
                f"{len(load_result.unexpected_keys)} unexpected keys in checkpoint, "
                f"first 5: {load_result.unexpected_keys[:5]}"
            )

        # Tie lm_head ↔ embed_tokens if config requires it and lm_head wasn't loaded
        if getattr(qwen_config, "tie_word_embeddings", False):
            try:
                if hasattr(model, "lm_head") and model.lm_head.weight.is_meta:
                    model.lm_head.weight = model.model.embed_tokens.weight
                else:
                    model.tie_weights()
            except AttributeError:
                model.tie_weights()

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

        model.eval()
        return model
