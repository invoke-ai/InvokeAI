# Copyright (c) 2024 The InvokeAI Development Team
"""Various utility functions needed by the loader and caching system."""

import json
from pathlib import Path
from typing import Optional

import torch
from diffusers import DiffusionPipeline

from invokeai.backend.model_manager.config import AnyModel
from invokeai.backend.onnx.onnx_runtime import IAIOnnxRuntimeModel


def calc_model_size_by_data(model: AnyModel) -> int:
    """Get size of a model in memory in bytes."""
    if isinstance(model, DiffusionPipeline):
        return _calc_pipeline_by_data(model)
    elif isinstance(model, torch.nn.Module):
        return _calc_model_by_data(model)
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
            res += _calc_model_by_data(submodel)
    return res


def _calc_model_by_data(model: torch.nn.Module) -> int:
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem: int = mem_params + mem_bufs  # in bytes
    return mem


def _calc_onnx_model_by_data(model: IAIOnnxRuntimeModel) -> int:
    tensor_size = model.tensors.size() * 2  # The session doubles this
    mem = tensor_size  # in bytes
    return mem


def calc_model_size_by_fs(model_path: Path, subfolder: Optional[str] = None, variant: Optional[str] = None) -> int:
    """Estimate the size of a model on disk in bytes."""
    if model_path.is_file():
        return model_path.stat().st_size

    if subfolder is not None:
        model_path = model_path / subfolder

    # this can happen when, for example, the safety checker is not downloaded.
    if not model_path.exists():
        return 0

    all_files = [f for f in model_path.iterdir() if (model_path / f).is_file()]

    fp16_files = {f for f in all_files if ".fp16." in f.name or ".fp16-" in f.name}
    bit8_files = {f for f in all_files if ".8bit." in f.name or ".8bit-" in f.name}
    other_files = set(all_files) - fp16_files - bit8_files

    if variant is None:
        files = other_files
    elif variant == "fp16":
        files = fp16_files
    elif variant == "8bit":
        files = bit8_files
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")

    # try read from index if exists
    index_postfix = ".index.json"
    if variant is not None:
        index_postfix = f".index.{variant}.json"

    for file in files:
        if not file.name.endswith(index_postfix):
            continue
        try:
            with open(model_path / file, "r") as f:
                index_data = json.loads(f.read())
            return int(index_data["metadata"]["total_size"])
        except Exception:
            pass

    # calculate files size if there is no index file
    formats = [
        (".safetensors",),  # safetensors
        (".bin",),  # torch
        (".onnx", ".pb"),  # onnx
        (".msgpack",),  # flax
        (".ckpt",),  # tf
        (".h5",),  # tf2
    ]

    for file_format in formats:
        model_files = [f for f in files if f.suffix in file_format]
        if len(model_files) == 0:
            continue

        model_size = 0
        for model_file in model_files:
            file_stats = (model_path / model_file).stat()
            model_size += file_stats.st_size
        return model_size

    return 0  # scheduler/feature_extractor/tokenizer - models without loading to gpu
