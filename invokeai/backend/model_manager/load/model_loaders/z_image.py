# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Z-Image model loading in InvokeAI."""

from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_ZImage_Config
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ZImage_Config, Main_GGUF_ZImage_Config
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
    Qwen3Encoder_GGUF_Config,
    Qwen3Encoder_Qwen3Encoder_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.util.devices import TorchDevice


def _convert_z_image_gguf_to_diffusers(sd: dict[str, Any]) -> dict[str, Any]:
    """Convert Z-Image GGUF state dict keys to diffusers format.

    The GGUF format uses original model keys that differ from diffusers:
    - qkv.weight (fused) -> to_q.weight, to_k.weight, to_v.weight (split)
    - out.weight -> to_out.0.weight
    - q_norm.weight -> norm_q.weight
    - k_norm.weight -> norm_k.weight
    - x_embedder.* -> all_x_embedder.2-1.*
    - final_layer.* -> all_final_layer.2-1.*
    - norm_final.* -> skipped (diffusers uses non-learnable LayerNorm)
    - x_pad_token, cap_pad_token: [dim] -> [1, dim] (diffusers expects batch dimension)
    """
    new_sd: dict[str, Any] = {}

    for key, value in sd.items():
        if not isinstance(key, str):
            new_sd[key] = value
            continue

        # Handle padding tokens: GGUF has shape [dim], diffusers expects [1, dim]
        if key in ("x_pad_token", "cap_pad_token"):
            if hasattr(value, "shape") and len(value.shape) == 1:
                # GGMLTensor doesn't support unsqueeze, so dequantize first if needed
                if hasattr(value, "get_dequantized_tensor"):
                    value = value.get_dequantized_tensor()
                # Use reshape instead of unsqueeze for better compatibility
                value = torch.as_tensor(value).reshape(1, -1)
            new_sd[key] = value
            continue

        # Handle x_embedder -> all_x_embedder.2-1
        if key.startswith("x_embedder."):
            suffix = key[len("x_embedder.") :]
            new_key = f"all_x_embedder.2-1.{suffix}"
            new_sd[new_key] = value
            continue

        # Handle final_layer -> all_final_layer.2-1
        if key.startswith("final_layer."):
            suffix = key[len("final_layer.") :]
            new_key = f"all_final_layer.2-1.{suffix}"
            new_sd[new_key] = value
            continue

        # Skip norm_final keys - the diffusers model uses LayerNorm with elementwise_affine=False
        # (no learnable weight/bias), but some checkpoints (e.g., FP8) include these as all-zeros
        if key.startswith("norm_final."):
            continue

        # Handle fused QKV weights - need to split
        if ".attention.qkv." in key:
            # Get the layer prefix and suffix
            prefix = key.rsplit(".attention.qkv.", 1)[0]
            suffix = key.rsplit(".attention.qkv.", 1)[1]  # "weight" or "bias"

            # Skip non-weight/bias tensors (e.g., FP8 scale_weight tensors)
            # These are quantization metadata and should not be split
            if suffix not in ("weight", "bias"):
                new_sd[key] = value
                continue

            # Split the fused QKV tensor into Q, K, V
            tensor = value
            if hasattr(tensor, "shape"):
                if tensor.shape[0] % 3 != 0:
                    raise ValueError(
                        f"Cannot split QKV tensor '{key}': first dimension ({tensor.shape[0]}) "
                        "is not divisible by 3. The model file may be corrupted or incompatible."
                    )
                dim = tensor.shape[0] // 3
                q = tensor[:dim]
                k = tensor[dim : 2 * dim]
                v = tensor[2 * dim :]

                new_sd[f"{prefix}.attention.to_q.{suffix}"] = q
                new_sd[f"{prefix}.attention.to_k.{suffix}"] = k
                new_sd[f"{prefix}.attention.to_v.{suffix}"] = v
            continue

        # Handle attention key renaming
        if ".attention." in key:
            new_key = key.replace(".q_norm.", ".norm_q.")
            new_key = new_key.replace(".k_norm.", ".norm_k.")
            new_key = new_key.replace(".attention.out.", ".attention.to_out.0.")
            new_sd[new_key] = value
            continue

        # For all other keys, just copy as-is
        new_sd[key] = value

    return new_sd


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class ZImageDiffusersModel(GenericDiffusersLoader):
    """Class to load Z-Image main models (Z-Image-Turbo, Z-Image-Base, Z-Image-Edit)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for Z-Image models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # Z-Image prefers bfloat16, but use safe dtype based on target device capabilities.
        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                torch_dtype=dtype,
                variant=variant,
            )
        except OSError as e:
            if variant and "no file named" in str(
                e
            ):  # try without the variant, just in case user's preferences changed
                result = load_class.from_pretrained(model_path, torch_dtype=dtype)
            else:
                raise e

        return result


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.Checkpoint)
class ZImageCheckpointModel(ModelLoader):
    """Class to load Z-Image transformer models from single-file checkpoints (safetensors, etc)."""

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
        from diffusers import ZImageTransformer2DModel
        from safetensors.torch import load_file

        if not isinstance(config, Main_Checkpoint_ZImage_Config):
            raise TypeError(
                f"Expected Main_Checkpoint_ZImage_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Load the state dict from safetensors/checkpoint file
        sd = load_file(model_path)

        # Some Z-Image checkpoint files have keys prefixed with "diffusion_model." or
        # "model.diffusion_model." (ComfyUI-style format). Check if we need to strip this prefix.
        prefix_to_strip = None
        for prefix in ["model.diffusion_model.", "diffusion_model."]:
            if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
                prefix_to_strip = prefix
                break

        if prefix_to_strip:
            stripped_sd = {}
            for key, value in sd.items():
                if isinstance(key, str) and key.startswith(prefix_to_strip):
                    stripped_sd[key[len(prefix_to_strip) :]] = value
                else:
                    stripped_sd[key] = value
            sd = stripped_sd

        # Check if the state dict is in original format (not diffusers format)
        # Original format has keys like "x_embedder.weight" instead of "all_x_embedder.2-1.weight"
        needs_conversion = any(k.startswith("x_embedder.") for k in sd.keys() if isinstance(k, str))

        if needs_conversion:
            # Convert from original format to diffusers format
            sd = _convert_z_image_gguf_to_diffusers(sd)

        # Create an empty model with the default Z-Image config
        # Z-Image-Turbo uses these default parameters from diffusers
        with accelerate.init_empty_weights():
            model = ZImageTransformer2DModel(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=3840,
                n_layers=30,
                n_refiner_layers=2,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
                cap_feat_dim=2560,
                rope_theta=256.0,
                t_scale=1000.0,
                axes_dims=[32, 48, 48],
                axes_lens=[1024, 512, 512],
            )

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Handle memory management and dtype conversion
        new_sd_size = sum([ten.nelement() * model_dtype.itemsize for ten in sd.values()])
        self._ram_cache.make_room(new_sd_size)

        # Filter out FP8 scale_weight and scaled_fp8 metadata keys
        # These are quantization metadata that shouldn't be loaded into the model
        keys_to_remove = [k for k in sd.keys() if k.endswith(".scale_weight") or k == "scaled_fp8"]
        for k in keys_to_remove:
            del sd[k]

        # Convert to target dtype
        for k in sd.keys():
            sd[k] = sd[k].to(model_dtype)

        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class ZImageGGUFCheckpointModel(ModelLoader):
    """Class to load GGUF-quantized Z-Image transformer models."""

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
        from diffusers import ZImageTransformer2DModel

        if not isinstance(config, Main_GGUF_ZImage_Config):
            raise TypeError(
                f"Expected Main_GGUF_ZImage_Config, got {type(config).__name__}. Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the GGUF state dict
        sd = gguf_sd_loader(model_path, compute_dtype=compute_dtype)

        # Some Z-Image GGUF models have keys prefixed with "diffusion_model." or
        # "model.diffusion_model." (ComfyUI-style format). Check if we need to strip this prefix.
        prefix_to_strip = None
        for prefix in ["model.diffusion_model.", "diffusion_model."]:
            if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
                prefix_to_strip = prefix
                break

        if prefix_to_strip:
            stripped_sd = {}
            for key, value in sd.items():
                if isinstance(key, str) and key.startswith(prefix_to_strip):
                    stripped_sd[key[len(prefix_to_strip) :]] = value
                else:
                    stripped_sd[key] = value
            sd = stripped_sd

        # Convert GGUF format keys to diffusers format
        sd = _convert_z_image_gguf_to_diffusers(sd)

        # Create an empty model with the default Z-Image config
        # Z-Image-Turbo uses these default parameters from diffusers
        with accelerate.init_empty_weights():
            model = ZImageTransformer2DModel(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=3840,
                n_layers=30,
                n_refiner_layers=2,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
                cap_feat_dim=2560,
                rope_theta=256.0,
                t_scale=1000.0,
                axes_dims=[32, 48, 48],
                axes_lens=[1024, 512, 512],
            )

        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.Qwen3Encoder)
class Qwen3EncoderLoader(ModelLoader):
    """Class to load standalone Qwen3 Encoder models for Z-Image (directory format)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_Qwen3Encoder_Config):
            raise ValueError("Only Qwen3Encoder_Qwen3Encoder_Config models are supported here.")

        model_path = Path(config.path)

        # Support both structures:
        # 1. Full model: model_root/text_encoder/ and model_root/tokenizer/
        # 2. Standalone download: model_root/ contains text_encoder files directly
        text_encoder_path = model_path / "text_encoder"
        tokenizer_path = model_path / "tokenizer"

        # Check if this is a standalone text_encoder download (no nested text_encoder folder)
        is_standalone = not text_encoder_path.exists() and (model_path / "config.json").exists()

        if is_standalone:
            text_encoder_path = model_path
            tokenizer_path = model_path  # Tokenizer files should also be in root

        match submodel_type:
            case SubModelType.Tokenizer:
                return AutoTokenizer.from_pretrained(tokenizer_path)
            case SubModelType.TextEncoder:
                # Determine safe dtype based on target device capabilities
                target_device = TorchDevice.choose_torch_device()
                model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
                return Qwen3ForCausalLM.from_pretrained(
                    text_encoder_path,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.ControlNet, format=ModelFormat.Checkpoint)
class ZImageControlCheckpointModel(ModelLoader):
    """Class to load Z-Image Control adapter models from safetensors checkpoint.

    Z-Image Control models are standalone adapters containing control layers
    (control_layers, control_all_x_embedder, control_noise_refiner) that can be
    combined with a base ZImageTransformer2DModel at runtime for spatial conditioning
    (Canny, HED, Depth, Pose, MLSD).
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are supported here.")

        # ControlNet type models don't use submodel_type - load the adapter directly
        return self._load_control_adapter(config)

    def _load_control_adapter(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from safetensors.torch import load_file

        from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter

        assert isinstance(config, ControlNet_Checkpoint_ZImage_Config)
        model_path = Path(config.path)

        # Load the safetensors state dict
        sd = load_file(model_path)

        # Determine number of control blocks from state dict
        # Control blocks are named control_layers.0, control_layers.1, etc.
        control_block_indices = set()
        for key in sd.keys():
            if key.startswith("control_layers."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    control_block_indices.add(int(parts[1]))
        num_control_blocks = len(control_block_indices) if control_block_indices else 6

        # Determine number of refiner layers from state dict
        refiner_indices: set[int] = set()
        for key in sd.keys():
            if key.startswith("control_noise_refiner."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    refiner_indices.add(int(parts[1]))
        n_refiner_layers = len(refiner_indices) if refiner_indices else 2

        # Determine control_in_dim from embedder weight shape
        # control_in_dim = weight.shape[1] / (f_patch_size * patch_size * patch_size)
        # For patch_size=2, f_patch_size=1: control_in_dim = weight.shape[1] / 4
        control_in_dim = 16  # Default for V1
        embedder_key = "control_all_x_embedder.2-1.weight"
        if embedder_key in sd:
            weight_shape = sd[embedder_key].shape
            # weight_shape[1] = f_patch_size * patch_size * patch_size * control_in_dim
            control_in_dim = weight_shape[1] // 4  # 4 = 1 * 2 * 2

        # Log detected configuration for debugging
        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        version = "V2.0" if control_in_dim > 16 else "V1"
        logger.info(
            f"Z-Image ControlNet detected: {version} "
            f"(control_in_dim={control_in_dim}, num_control_blocks={num_control_blocks}, "
            f"n_refiner_layers={n_refiner_layers})"
        )

        # Create an empty control adapter
        dim = 3840
        with accelerate.init_empty_weights():
            model = ZImageControlAdapter(
                num_control_blocks=num_control_blocks,
                control_in_dim=control_in_dim,
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                dim=dim,
                n_refiner_layers=n_refiner_layers,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
            )

        # Load state dict with strict=False to handle missing keys like x_pad_token
        # Some control adapters may not include x_pad_token in their checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(sd, assign=True, strict=False)

        # Initialize x_pad_token if it was missing from the checkpoint
        if "x_pad_token" in missing_keys:
            import torch.nn as nn

            model.x_pad_token = nn.Parameter(torch.empty(dim))
            nn.init.normal_(model.x_pad_token, std=0.02)

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.Checkpoint)
class Qwen3EncoderCheckpointLoader(ModelLoader):
    """Class to load single-file Qwen3 Encoder models for Z-Image (safetensors format)."""

    # Default HuggingFace model to load tokenizer from when using single-file Qwen3 encoder
    # Must be Qwen3 (not Qwen2.5) to match Z-Image's text encoder architecture and special tokens
    DEFAULT_TOKENIZER_SOURCE = "Qwen/Qwen3-4B"

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_Checkpoint_Config):
            raise ValueError("Only Qwen3Encoder_Checkpoint_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_singlefile(config)
            case SubModelType.Tokenizer:
                # For single-file Qwen3, load tokenizer from HuggingFace
                return AutoTokenizer.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE)

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from safetensors.torch import load_file
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, Qwen3Encoder_Checkpoint_Config):
            raise TypeError(
                f"Expected Qwen3Encoder_Checkpoint_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the state dict from safetensors file
        sd = load_file(model_path)

        # Determine Qwen model configuration from state dict
        # Count the number of layers by looking at layer keys
        layer_count = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_idx = int(parts[2])
                        layer_count = max(layer_count, layer_idx + 1)
                    except ValueError:
                        pass

        # Get hidden size from embed_tokens weight shape
        embed_weight = sd.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise ValueError("Could not find model.embed_tokens.weight in state dict")
        if embed_weight.ndim != 2:
            raise ValueError(
                f"Expected 2D embed_tokens weight tensor, got shape {embed_weight.shape}. "
                "The model file may be corrupted or incompatible."
            )
        hidden_size = embed_weight.shape[1]
        vocab_size = embed_weight.shape[0]

        # Detect attention configuration from layer 0 weights
        q_proj_weight = sd.get("model.layers.0.self_attn.q_proj.weight")
        k_proj_weight = sd.get("model.layers.0.self_attn.k_proj.weight")
        gate_proj_weight = sd.get("model.layers.0.mlp.gate_proj.weight")

        if q_proj_weight is None or k_proj_weight is None or gate_proj_weight is None:
            raise ValueError("Could not find attention/mlp weights in state dict to determine configuration")

        # Calculate dimensions from actual weights
        # Qwen3 uses head_dim separately from hidden_size
        head_dim = 128  # Standard head dimension for Qwen3 models
        num_attention_heads = q_proj_weight.shape[0] // head_dim
        num_kv_heads = k_proj_weight.shape[0] // head_dim
        intermediate_size = gate_proj_weight.shape[0]

        # Create Qwen3 config - matches the diffusers text_encoder/config.json
        qwen_config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layer_count,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            rope_theta=1000000.0,
            use_sliding_window=False,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=model_dtype,
        )

        # Handle memory management
        new_sd_size = sum([ten.nelement() * model_dtype.itemsize for ten in sd.values()])
        self._ram_cache.make_room(new_sd_size)

        # Convert to target dtype
        for k in sd.keys():
            sd[k] = sd[k].to(model_dtype)

        # Use Qwen3ForCausalLM - the correct model class for Z-Image text encoder
        # Use init_empty_weights for fast model creation, then load weights with assign=True
        with accelerate.init_empty_weights():
            model = Qwen3ForCausalLM(qwen_config)

        # Load the text model weights from checkpoint
        # assign=True replaces meta tensors with real ones from state dict
        model.load_state_dict(sd, strict=False, assign=True)

        # Handle tied weights: lm_head shares weight with embed_tokens when tie_word_embeddings=True
        # This doesn't work automatically with init_empty_weights, so we need to manually tie them
        if qwen_config.tie_word_embeddings:
            model.tie_weights()

        # Re-initialize any remaining meta tensor buffers (like rotary embeddings inv_freq)
        # These are computed from config, not loaded from checkpoint
        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta:
                # Get parent module and buffer name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    buffer_name = parts[1]
                else:
                    parent = model
                    buffer_name = name

                # Re-initialize the buffer based on expected shape and dtype
                # For rotary embeddings, this is inv_freq which is computed from config
                if buffer_name == "inv_freq":
                    # Compute inv_freq from config (same logic as Qwen3RotaryEmbedding.__init__)
                    base = qwen_config.rope_theta
                    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    parent.register_buffer(buffer_name, inv_freq.to(model_dtype), persistent=False)
                else:
                    # For other buffers, log warning
                    logger.warning(f"Re-initializing unknown meta buffer: {name}")

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.GGUFQuantized)
class Qwen3EncoderGGUFLoader(ModelLoader):
    """Class to load GGUF-quantized Qwen3 Encoder models for Z-Image."""

    # Default HuggingFace model to load tokenizer from when using GGUF Qwen3 encoder
    # Must be Qwen3 (not Qwen2.5) to match Z-Image's text encoder architecture and special tokens
    DEFAULT_TOKENIZER_SOURCE = "Qwen/Qwen3-4B"

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_GGUF_Config):
            raise ValueError("Only Qwen3Encoder_GGUF_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_gguf(config)
            case SubModelType.Tokenizer:
                # For GGUF Qwen3, load tokenizer from HuggingFace
                return AutoTokenizer.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE)

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_gguf(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, Qwen3Encoder_GGUF_Config):
            raise TypeError(
                f"Expected Qwen3Encoder_GGUF_Config, got {type(config).__name__}. Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the GGUF state dict - this returns GGMLTensor wrappers (on CPU)
        # We keep them on CPU and let the model cache system handle GPU movement
        # via apply_custom_layers_to_model() and the partial loading cache
        sd = gguf_sd_loader(model_path, compute_dtype=compute_dtype)

        # Check if this is llama.cpp format (blk.X.) or PyTorch format (model.layers.X.)
        is_llamacpp_format = any(k.startswith("blk.") for k in sd.keys() if isinstance(k, str))

        if is_llamacpp_format:
            logger.info("Detected llama.cpp GGUF format, converting keys to PyTorch format")
            sd = self._convert_llamacpp_to_pytorch(sd)

        # Determine Qwen model configuration from state dict
        # Count the number of layers by looking at layer keys
        layer_count = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_idx = int(parts[2])
                        layer_count = max(layer_count, layer_idx + 1)
                    except ValueError:
                        pass

        # Get hidden size from embed_tokens weight shape
        embed_weight = sd.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise ValueError("Could not find model.embed_tokens.weight in state dict")

        # Handle GGMLTensor shape access
        embed_shape = embed_weight.shape if hasattr(embed_weight, "shape") else embed_weight.tensor_shape
        if len(embed_shape) != 2:
            raise ValueError(
                f"Expected 2D embed_tokens weight tensor, got shape {embed_shape}. "
                "The model file may be corrupted or incompatible."
            )
        hidden_size = embed_shape[1]
        vocab_size = embed_shape[0]

        # Detect attention configuration from layer 0 weights
        q_proj_weight = sd.get("model.layers.0.self_attn.q_proj.weight")
        k_proj_weight = sd.get("model.layers.0.self_attn.k_proj.weight")
        gate_proj_weight = sd.get("model.layers.0.mlp.gate_proj.weight")

        if q_proj_weight is None or k_proj_weight is None or gate_proj_weight is None:
            raise ValueError("Could not find attention/mlp weights in state dict to determine configuration")

        # Handle GGMLTensor shape access
        q_shape = q_proj_weight.shape if hasattr(q_proj_weight, "shape") else q_proj_weight.tensor_shape
        k_shape = k_proj_weight.shape if hasattr(k_proj_weight, "shape") else k_proj_weight.tensor_shape
        gate_shape = gate_proj_weight.shape if hasattr(gate_proj_weight, "shape") else gate_proj_weight.tensor_shape

        # Calculate dimensions from actual weights
        head_dim = 128  # Standard head dimension for Qwen3 models
        num_attention_heads = q_shape[0] // head_dim
        num_kv_heads = k_shape[0] // head_dim
        intermediate_size = gate_shape[0]

        logger.info(
            f"Qwen3 GGUF Encoder config detected: layers={layer_count}, hidden={hidden_size}, "
            f"heads={num_attention_heads}, kv_heads={num_kv_heads}, intermediate={intermediate_size}, "
            f"head_dim={head_dim}"
        )

        # Create Qwen3 config
        qwen_config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layer_count,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            rope_theta=1000000.0,
            use_sliding_window=False,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=compute_dtype,
        )

        # Use Qwen3ForCausalLM with empty weights, then load GGUF tensors
        with accelerate.init_empty_weights():
            model = Qwen3ForCausalLM(qwen_config)

        # Load the GGUF weights with assign=True
        # GGMLTensor wrappers will be dequantized on-the-fly during inference
        model.load_state_dict(sd, strict=False, assign=True)

        # Dequantize embed_tokens weight - embedding lookups require indexed access
        # which quantized GGMLTensors can't efficiently provide (no __torch_dispatch__ for embedding)
        from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

        embed_tokens_weight = model.model.embed_tokens.weight
        if isinstance(embed_tokens_weight, GGMLTensor):
            dequantized = embed_tokens_weight.get_dequantized_tensor()
            model.model.embed_tokens.weight = torch.nn.Parameter(dequantized, requires_grad=False)
            logger.info("Dequantized embed_tokens weight for embedding lookups")

        # Handle tied weights - llama.cpp GGUF doesn't include lm_head.weight when embeddings are tied
        # So we need to manually tie them after loading
        if qwen_config.tie_word_embeddings:
            # Check if lm_head.weight is still a meta tensor (wasn't in GGUF state dict)
            if model.lm_head.weight.is_meta:
                # Directly assign embed_tokens weight to lm_head (now dequantized)
                model.lm_head.weight = model.model.embed_tokens.weight
                logger.info("Tied lm_head.weight to embed_tokens.weight (GGUF tied embeddings)")
            else:
                # If lm_head.weight was loaded, use standard tie_weights
                model.tie_weights()

        # Re-initialize any remaining meta tensor buffers (like rotary embeddings inv_freq)
        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    buffer_name = parts[1]
                else:
                    parent = model
                    buffer_name = name

                if buffer_name == "inv_freq":
                    # Compute inv_freq from config - keep on CPU, cache system will move to GPU as needed
                    base = qwen_config.rope_theta
                    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    parent.register_buffer(buffer_name, inv_freq.to(dtype=compute_dtype), persistent=False)
                else:
                    logger.warning(f"Re-initializing unknown meta buffer: {name}")

        # Final check: ensure no meta tensors remain in parameters
        meta_params = [(name, p) for name, p in model.named_parameters() if p.is_meta]
        if meta_params:
            meta_names = [name for name, _ in meta_params]
            raise RuntimeError(
                f"Failed to load all parameters from GGUF. The following remain as meta tensors: {meta_names}. "
                "This may indicate missing keys in the GGUF file or a key mapping issue."
            )

        return model

    def _convert_llamacpp_to_pytorch(self, sd: dict[str, Any]) -> dict[str, Any]:
        """Convert llama.cpp GGUF keys to PyTorch/HuggingFace format for Qwen models.

        llama.cpp format:
        - blk.X.attn_q.weight -> model.layers.X.self_attn.q_proj.weight
        - blk.X.attn_k.weight -> model.layers.X.self_attn.k_proj.weight
        - blk.X.attn_v.weight -> model.layers.X.self_attn.v_proj.weight
        - blk.X.attn_output.weight -> model.layers.X.self_attn.o_proj.weight
        - blk.X.attn_q_norm.weight -> model.layers.X.self_attn.q_norm.weight (Qwen3 QK norm)
        - blk.X.attn_k_norm.weight -> model.layers.X.self_attn.k_norm.weight (Qwen3 QK norm)
        - blk.X.ffn_gate.weight -> model.layers.X.mlp.gate_proj.weight
        - blk.X.ffn_up.weight -> model.layers.X.mlp.up_proj.weight
        - blk.X.ffn_down.weight -> model.layers.X.mlp.down_proj.weight
        - blk.X.attn_norm.weight -> model.layers.X.input_layernorm.weight
        - blk.X.ffn_norm.weight -> model.layers.X.post_attention_layernorm.weight
        - token_embd.weight -> model.embed_tokens.weight
        - output_norm.weight -> model.norm.weight
        - output.weight -> lm_head.weight (if not tied)
        """
        import re

        key_map = {
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "attn_q_norm": "self_attn.q_norm",  # Qwen3 QK normalization
            "attn_k_norm": "self_attn.k_norm",  # Qwen3 QK normalization
            "ffn_gate": "mlp.gate_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "attn_norm": "input_layernorm",
            "ffn_norm": "post_attention_layernorm",
        }

        new_sd: dict[str, Any] = {}
        blk_pattern = re.compile(r"^blk\.(\d+)\.(.+)$")

        for key, value in sd.items():
            if not isinstance(key, str):
                new_sd[key] = value
                continue

            # Handle block layers
            match = blk_pattern.match(key)
            if match:
                layer_idx = match.group(1)
                rest = match.group(2)

                # Split rest into component and suffix (e.g., "attn_q.weight" -> "attn_q", "weight")
                parts = rest.split(".", 1)
                component = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""

                if component in key_map:
                    new_component = key_map[component]
                    new_key = f"model.layers.{layer_idx}.{new_component}"
                    if suffix:
                        new_key += f".{suffix}"
                    new_sd[new_key] = value
                else:
                    # Unknown component, keep as-is with model.layers prefix
                    new_sd[f"model.layers.{layer_idx}.{rest}"] = value
                continue

            # Handle non-block keys
            if key == "token_embd.weight":
                new_sd["model.embed_tokens.weight"] = value
            elif key == "output_norm.weight":
                new_sd["model.norm.weight"] = value
            elif key == "output.weight":
                new_sd["lm_head.weight"] = value
            else:
                # Keep other keys as-is
                new_sd[key] = value

        return new_sd
