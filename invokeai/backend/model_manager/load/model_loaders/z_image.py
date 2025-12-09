# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Z-Image model loading in InvokeAI."""

from pathlib import Path
from typing import Any, Optional

import accelerate
import torch

from transformers import AutoTokenizer, Qwen3ForCausalLM

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ZImage_Config, Main_GGUF_ZImage_Config
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
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


def _convert_z_image_gguf_to_diffusers(sd: dict[str, Any]) -> dict[str, Any]:
    """Convert Z-Image GGUF state dict keys to diffusers format.

    The GGUF format uses original model keys that differ from diffusers:
    - qkv.weight (fused) -> to_q.weight, to_k.weight, to_v.weight (split)
    - out.weight -> to_out.0.weight
    - q_norm.weight -> norm_q.weight
    - k_norm.weight -> norm_k.weight
    - x_embedder.* -> all_x_embedder.2-1.*
    - final_layer.* -> all_final_layer.2-1.*
    """
    new_sd: dict[str, Any] = {}

    for key, value in sd.items():
        if not isinstance(key, str):
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

        # Handle fused QKV weights - need to split
        if ".attention.qkv." in key:
            # Get the layer prefix and suffix
            prefix = key.rsplit(".attention.qkv.", 1)[0]
            suffix = key.rsplit(".attention.qkv.", 1)[1]  # "weight" or "bias"

            # Split the fused QKV tensor into Q, K, V
            tensor = value
            if hasattr(tensor, "shape"):
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

        # Z-Image requires bfloat16 for correct inference.
        dtype = torch.bfloat16
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

        assert isinstance(config, Main_Checkpoint_ZImage_Config)
        model_path = Path(config.path)

        # Load the state dict from safetensors/checkpoint file
        sd = load_file(model_path)

        # Some Z-Image checkpoint files have keys prefixed with "diffusion_model."
        # Check if we need to strip this prefix
        has_prefix = any(k.startswith("diffusion_model.") for k in sd.keys() if isinstance(k, str))

        if has_prefix:
            stripped_sd = {}
            prefix = "diffusion_model."
            for key, value in sd.items():
                if isinstance(key, str) and key.startswith(prefix):
                    stripped_sd[key[len(prefix) :]] = value
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

        # Handle memory management and dtype conversion
        new_sd_size = sum([ten.nelement() * torch.bfloat16.itemsize for ten in sd.values()])
        self._ram_cache.make_room(new_sd_size)

        # Convert to bfloat16 (required for Z-Image)
        for k in sd.keys():
            sd[k] = sd[k].to(torch.bfloat16)

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

        assert isinstance(config, Main_GGUF_ZImage_Config)
        model_path = Path(config.path)

        # Load the GGUF state dict
        sd = gguf_sd_loader(model_path, compute_dtype=torch.bfloat16)

        # Some Z-Image GGUF models have keys prefixed with "diffusion_model."
        # Check if we need to strip this prefix
        has_prefix = any(k.startswith("diffusion_model.") for k in sd.keys() if isinstance(k, str))

        if has_prefix:
            stripped_sd = {}
            prefix = "diffusion_model."
            for key, value in sd.items():
                if isinstance(key, str) and key.startswith(prefix):
                    stripped_sd[key[len(prefix) :]] = value
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

        match submodel_type:
            case SubModelType.Tokenizer:
                return AutoTokenizer.from_pretrained(Path(config.path) / "tokenizer")
            case SubModelType.TextEncoder:
                return Qwen3ForCausalLM.from_pretrained(
                    Path(config.path) / "text_encoder",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.Checkpoint)
class Qwen3EncoderCheckpointLoader(ModelLoader):
    """Class to load single-file Qwen3 Encoder models for Z-Image (safetensors format)."""

    # Default HuggingFace model to load tokenizer from when using single-file Qwen3 encoder
    DEFAULT_TOKENIZER_SOURCE = "Qwen/Qwen2.5-3B"

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
        from invokeai.backend.util.logging import InvokeAILogger
        from safetensors.torch import load_file
        from transformers import Qwen3Config, Qwen3ForCausalLM

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        assert isinstance(config, Qwen3Encoder_Checkpoint_Config)
        model_path = Path(config.path)

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

        logger.info(
            f"Qwen3 Encoder config detected: layers={layer_count}, hidden={hidden_size}, "
            f"heads={num_attention_heads}, kv_heads={num_kv_heads}, intermediate={intermediate_size}, "
            f"head_dim={head_dim}"
        )

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
            torch_dtype=torch.bfloat16,
        )

        # Handle memory management
        new_sd_size = sum([ten.nelement() * torch.bfloat16.itemsize for ten in sd.values()])
        self._ram_cache.make_room(new_sd_size)

        # Convert to bfloat16
        for k in sd.keys():
            sd[k] = sd[k].to(torch.bfloat16)

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
                    parent.register_buffer(buffer_name, inv_freq.to(torch.bfloat16), persistent=False)
                else:
                    # For other buffers, log warning
                    logger.warning(f"Re-initializing unknown meta buffer: {name}")

        return model
