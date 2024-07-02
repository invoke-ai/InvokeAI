# Copyright (c) 2024 The InvokeAI Development Team
"""Various utility functions needed by the loader and caching system."""

import torch
from diffusers import DiffusionPipeline

from invokeai.backend.model_manager.config import AnyModel
from invokeai.backend.model_manager.load.model_size_utils import calc_module_size
from invokeai.backend.onnx.onnx_runtime import IAIOnnxRuntimeModel


def calc_model_size_by_data(model: AnyModel) -> int:
    """Get size of a model in memory in bytes."""
    if isinstance(model, DiffusionPipeline):
        return _calc_pipeline_by_data(model)
    elif isinstance(model, torch.nn.Module):
        return calc_module_size(model)
    elif isinstance(model, IAIOnnxRuntimeModel):
        return _calc_onnx_model_by_data(model)
    else:
        return 0


def _calc_pipeline_by_data(pipeline: DiffusionPipeline) -> int:
    res = 0
    assert hasattr(pipeline, "components")
    for submodel_key in pipeline.components.keys():
        submodel = getattr(pipeline, submodel_key)
        if submodel is not None and isinstance(submodel, torch.nn.Module):
            res += calc_module_size(submodel)
    return res


def _calc_onnx_model_by_data(model: IAIOnnxRuntimeModel) -> int:
    tensor_size = model.tensors.size() * 2  # The session doubles this
    mem = tensor_size  # in bytes
    return mem
