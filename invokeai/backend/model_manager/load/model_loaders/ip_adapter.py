# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for IP Adapter model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.ip_adapter.ip_adapter import build_ip_adapter
from invokeai.backend.model_manager import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.load import ModelLoader, ModelLoaderRegistry


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.IPAdapter, format=ModelFormat.InvokeAI)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.IPAdapter, format=ModelFormat.Checkpoint)
class IPAdapterInvokeAILoader(ModelLoader):
    """Class to load IP Adapter diffusers models."""

    def _load_model(
        self,
        model_path: Path,
        model_variant: Optional[ModelRepoVariant] = None,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("There are no submodels in an IP-Adapter model.")
        model = build_ip_adapter(
            ip_adapter_ckpt_path=str(model_path),
            device=torch.device("cpu"),
            dtype=self._torch_dtype,
        )
        return model
