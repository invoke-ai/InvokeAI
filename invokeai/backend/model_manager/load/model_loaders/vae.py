# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for VAE model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import torch
import safetensors
from omegaconf import OmegaConf, DictConfig
from invokeai.backend.util.devices import torch_dtype
from invokeai.backend.model_manager import AnyModel, AnyModelConfig, BaseModelType, ModelFormat, ModelRepoVariant, ModelType, SubModelType
from invokeai.backend.model_manager.load.load_base import AnyModelLoader
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.convert_ckpt_to_diffusers import convert_ldm_vae_to_diffusers

@AnyModelLoader.register(base=BaseModelType.Any, type=ModelType.Vae, format=ModelFormat.Diffusers)
@AnyModelLoader.register(base=BaseModelType.StableDiffusion1, type=ModelType.Vae, format=ModelFormat.Checkpoint)
@AnyModelLoader.register(base=BaseModelType.StableDiffusion2, type=ModelType.Vae, format=ModelFormat.Checkpoint)
class VaeDiffusersModel(ModelLoader):
    """Class to load VAE models."""

    def _load_model(
        self,
        model_path: Path,
        model_variant: Optional[ModelRepoVariant] = None,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise Exception("There are no submodels in VAEs")
        vae_class = self._get_hf_load_class(model_path)
        variant = model_variant.value if model_variant else None
        result: AnyModel = vae_class.from_pretrained(
            model_path, torch_dtype=self._torch_dtype, variant=variant
        )  # type: ignore
        return result

    def _needs_conversion(self, config: AnyModelConfig, model_path: Path, dest_path: Path) -> bool:
        print(f'DEBUG: last_modified={config.last_modified}')
        print(f'DEBUG: cache_path={(dest_path / "config.json").stat().st_mtime}')
        print(f'DEBUG: model_path={model_path.stat().st_mtime}')
        if config.format != ModelFormat.Checkpoint:
            return False
        elif dest_path.exists() \
             and (dest_path / "config.json").stat().st_mtime >= config.last_modified \
             and (dest_path / "config.json").stat().st_mtime >= model_path.stat().st_mtime:
            return False
        else:
            return True

    def _convert_model(self,
                       config: AnyModelConfig,
                       weights_path: Path,
                       output_path: Path
                       ) -> Path:
        if config.base not in {BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2}:
            raise Exception(f"Vae conversion not supported for model type: {config.base}")
        else:
            config_file = 'v1-inference.yaml' if config.base == BaseModelType.StableDiffusion1 else "v2-inference-v.yaml"

        if weights_path.suffix == ".safetensors":
            checkpoint = safetensors.torch.load_file(weights_path, device="cpu")
        else:
            checkpoint = torch.load(weights_path, map_location="cpu")

        dtype = torch_dtype()

        # sometimes weights are hidden under "state_dict", and sometimes not
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        ckpt_config = OmegaConf.load(self._app_config.legacy_conf_path / config_file)
        assert isinstance(ckpt_config, DictConfig)

        print(f'DEBUG: CONVERTIGN')
        vae_model = convert_ldm_vae_to_diffusers(
            checkpoint=checkpoint,
            vae_config=ckpt_config,
            image_size=512,
        )
        vae_model.to(dtype) # set precision appropriately
        vae_model.save_pretrained(output_path, safe_serialization=True, torch_dtype=dtype)
        return output_path

