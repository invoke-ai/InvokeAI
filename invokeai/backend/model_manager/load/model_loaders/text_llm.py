from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.TextLLM, format=ModelFormat.Diffusers)
class TextLLMModelLoader(ModelLoader):
    """Class for loading text causal language models (Llama, Phi, Qwen, Mistral, etc.)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("Unexpected submodel requested for TextLLM model.")

        # Use float32 for CPU-only models since CPU fp16 is emulated and slow.
        dtype = self._torch_dtype
        if getattr(config, "cpu_only", False) is True:
            dtype = torch.float32

        model_path = Path(config.path)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
        return model
