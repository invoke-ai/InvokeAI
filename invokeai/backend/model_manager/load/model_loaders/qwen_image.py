from pathlib import Path
from typing import Optional

import accelerate
import torch

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_GGUF_QwenImage_Config
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

        # Strip ComfyUI-style prefixes if present
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

        # Auto-detect architecture from state dict
        num_layers = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("transformer_blocks."):
                parts = key.split(".")
                if len(parts) >= 2:
                    try:
                        layer_idx = int(parts[1])
                        num_layers = max(num_layers, layer_idx + 1)
                    except ValueError:
                        pass

        # Detect dimensions from weights
        num_attention_heads = 24  # default
        attention_head_dim = 128  # default

        if "img_in.weight" in sd:
            w = sd["img_in.weight"]
            shape = w.tensor_shape if isinstance(w, GGMLTensor) else w.shape
            hidden_dim = shape[0]
            in_channels = shape[1]
            num_attention_heads = hidden_dim // attention_head_dim

        joint_attention_dim = 3584  # default
        if "txt_in.weight" in sd:
            w = sd["txt_in.weight"]
            shape = w.tensor_shape if isinstance(w, GGMLTensor) else w.shape
            joint_attention_dim = shape[1]

        model_config: dict = {
            "patch_size": 2,
            "in_channels": in_channels if "img_in.weight" in sd else 64,
            "out_channels": 16,
            "num_layers": num_layers if num_layers > 0 else 60,
            "attention_head_dim": attention_head_dim,
            "num_attention_heads": num_attention_heads,
            "joint_attention_dim": joint_attention_dim,
            "guidance_embeds": False,
            "axes_dims_rope": (16, 56, 56),
        }

        # zero_cond_t is only used by edit-variant models. It enables dual modulation
        # for noisy vs reference patches. Setting it on txt2img models produces garbage.
        # Also requires diffusers 0.37+ (the parameter doesn't exist in older versions).
        import inspect

        is_edit = getattr(config, "variant", None) == QwenImageVariantType.Edit
        if is_edit and "zero_cond_t" in inspect.signature(QwenImageTransformer2DModel.__init__).parameters:
            model_config["zero_cond_t"] = True

        with accelerate.init_empty_weights():
            model = QwenImageTransformer2DModel(**model_config)

        model.load_state_dict(sd, strict=False, assign=True)
        return model
