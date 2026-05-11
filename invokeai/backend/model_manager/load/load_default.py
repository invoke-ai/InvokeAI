# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Default implementation of model loading in InvokeAI."""

from logging import Logger
from pathlib import Path
from typing import Optional

import torch

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager.configs.base import Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_base import LoadedModel, ModelLoaderBase
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache, get_model_cache_key
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_fs
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    SubModelType,
)
from invokeai.backend.util.devices import TorchDevice


# TO DO: The loader is not thread safe!
class ModelLoader(ModelLoaderBase):
    """Default implementation of ModelLoaderBase."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCache,
    ):
        """Initialize the loader."""
        self._app_config = app_config
        self._logger = logger
        self._ram_cache = ram_cache
        self._torch_dtype = TorchDevice.choose_torch_dtype()
        self._torch_device = TorchDevice.choose_torch_device()

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its configuration.

        Given a model's configuration as returned by the ModelRecordConfigStore service,
        return a LoadedModel object that can be used for inference.

        :param model config: Configuration record for this model
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        model_path = self._get_model_path(model_config)

        if not model_path.exists():
            raise FileNotFoundError(f"Files for model '{model_config.name}' not found at {model_path}")

        with skip_torch_weight_init():
            cache_record = self._load_and_cache(model_config, submodel_type)
        return LoadedModel(config=model_config, cache_record=cache_record, cache=self._ram_cache)

    @property
    def ram_cache(self) -> ModelCache:
        """Return the ram cache associated with this loader."""
        return self._ram_cache

    def _get_model_path(self, config: AnyModelConfig) -> Path:
        model_base = self._app_config.models_path
        return (model_base / config.path).resolve()

    def _get_execution_device(
        self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None
    ) -> Optional[torch.device]:
        """Determine the execution device for a model based on its configuration.

        CPU-only execution is only applied to text encoder submodels to save VRAM while keeping
        the denoiser on GPU for performance. Conditioning tensors are moved to GPU after encoding.

        Returns:
            torch.device("cpu") if the model should run on CPU only, None otherwise (use cache default).
        """
        # Check if this is a text encoder submodel of a main model with cpu_only setting
        if hasattr(config, "default_settings") and config.default_settings is not None:
            if hasattr(config.default_settings, "cpu_only") and config.default_settings.cpu_only is True:
                # Only apply CPU execution to text encoder submodels
                if submodel_type in [SubModelType.TextEncoder, SubModelType.TextEncoder2, SubModelType.TextEncoder3]:
                    return torch.device("cpu")

        # Check if this is a standalone text encoder config with cpu_only field (T5Encoder, Qwen3Encoder, etc.)
        if hasattr(config, "cpu_only") and config.cpu_only is True:
            return torch.device("cpu")

        return None

    def _load_and_cache(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> CacheRecord:
        stats_name = ":".join([config.base, config.type, config.name, (submodel_type or "")])
        try:
            return self._ram_cache.get(key=get_model_cache_key(config.key, submodel_type), stats_name=stats_name)
        except IndexError:
            pass

        config.path = str(self._get_model_path(config))
        self._ram_cache.make_room(self.get_size_fs(config, Path(config.path), submodel_type))
        loaded_model = self._load_model(config, submodel_type)

        # Determine execution device from model config, considering submodel type
        execution_device = self._get_execution_device(config, submodel_type)

        self._ram_cache.put(
            get_model_cache_key(config.key, submodel_type),
            model=loaded_model,
            execution_device=execution_device,
        )

        return self._ram_cache.get(key=get_model_cache_key(config.key, submodel_type), stats_name=stats_name)

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Get the size of the model on disk."""
        return calc_model_size_by_fs(
            model_path=model_path,
            subfolder=submodel_type.value if submodel_type else None,
            variant=config.repo_variant if isinstance(config, Diffusers_Config_Base) else None,
        )

    def _should_use_fp8(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> bool:
        """Check if FP8 layerwise casting should be applied to a model."""
        # FP8 storage only works on CUDA
        if self._torch_device.type != "cuda":
            return False

        # Z-Image has dtype mismatch issues with diffusers' layerwise casting
        # (skipped modules produce bf16, hooked modules expect fp16).
        from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

        if hasattr(config, "base") and config.base == BaseModelType.ZImage:
            return False

        # VAEs are excluded — fp8 storage causes noticeable quality degradation in decode.
        if hasattr(config, "type") and config.type == ModelType.VAE:
            return False

        # LoRAs (including ControlLoRA) are excluded — they are not run as a standalone forward pass,
        # they are patched into a base model, so the layerwise-casting hooks would never fire. The
        # toggle is also hidden in the UI for ControlLoRA; this guard handles legacy persisted values.
        if hasattr(config, "type") and config.type in (ModelType.LoRA, ModelType.ControlLoRa):
            return False

        # Don't apply FP8 to text encoders, tokenizers, schedulers, VAEs, etc.
        _excluded_submodel_types = {
            SubModelType.TextEncoder,
            SubModelType.TextEncoder2,
            SubModelType.TextEncoder3,
            SubModelType.Tokenizer,
            SubModelType.Tokenizer2,
            SubModelType.Tokenizer3,
            SubModelType.Scheduler,
            SubModelType.SafetyChecker,
            SubModelType.VAE,
            SubModelType.VAEDecoder,
            SubModelType.VAEEncoder,
        }
        if submodel_type in _excluded_submodel_types:
            return False

        # Check default_settings.fp8_storage (Main models, ControlNet)
        if hasattr(config, "default_settings") and config.default_settings is not None:
            if hasattr(config.default_settings, "fp8_storage") and config.default_settings.fp8_storage is True:
                return True

        return False

    def _apply_fp8_layerwise_casting(
        self, model: AnyModel, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None
    ) -> AnyModel:
        """Apply FP8 layerwise casting to a model if enabled in its config."""
        if not self._should_use_fp8(config, submodel_type):
            return model

        storage_dtype = torch.float8_e4m3fn
        compute_dtype = self._torch_dtype

        # Detect the model's current dtype to use as compute dtype, since models
        # (e.g. Flux) may require a specific dtype (bf16) that differs from the global torch dtype (fp16).
        if isinstance(model, torch.nn.Module):
            first_param = next(model.parameters(), None)
            if first_param is not None:
                compute_dtype = first_param.dtype

        from diffusers.models.modeling_utils import ModelMixin

        if isinstance(model, ModelMixin):
            model.enable_layerwise_casting(
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
            )
        elif isinstance(model, torch.nn.Module):
            self._apply_fp8_to_nn_module(model, storage_dtype=storage_dtype, compute_dtype=compute_dtype)
        else:
            return model

        param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        self._logger.info(
            f"FP8 layerwise casting enabled for {config.name} "
            f"(storage=float8_e4m3fn, compute={compute_dtype}, "
            f"param_size={param_bytes / (1024**2):.0f}MB)"
        )
        return model

    @staticmethod
    def _apply_fp8_to_nn_module(model: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype) -> None:
        """Apply FP8 layerwise casting to a plain nn.Module by wrapping forward in a try/finally.

        A pre-hook/post-hook pair is not exception-safe: `register_forward_hook` only fires on a
        successful forward, so an exception would leave parameters in `compute_dtype` and silently
        defeat the FP8 storage savings (and make cache size accounting stale). Wrapping `forward`
        directly guarantees the storage-dtype cast runs even when forward raises.
        """
        for module in model.modules():
            params = list(module.parameters(recurse=False))
            if not params:
                continue

            for param in params:
                param.data = param.data.to(storage_dtype)

            ModelLoader._wrap_forward_with_fp8_cast(module, storage_dtype, compute_dtype)

    @staticmethod
    def _wrap_forward_with_fp8_cast(
        module: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype
    ) -> None:
        original_forward = module.forward

        def forward(*args: object, **kwargs: object) -> object:
            for p in module.parameters(recurse=False):
                p.data = p.data.to(compute_dtype)
            try:
                return original_forward(*args, **kwargs)
            finally:
                for p in module.parameters(recurse=False):
                    p.data = p.data.to(storage_dtype)

        module.forward = forward  # type: ignore[method-assign]

    # This needs to be implemented in the subclass
    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        raise NotImplementedError
