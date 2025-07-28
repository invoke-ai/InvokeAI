# Copyright (c) 2024 The InvokeAI Development Team
"""Various utility functions needed by the loader and caching system."""

import json
import logging
from pathlib import Path
from typing import Optional

import onnxruntime as ort
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from transformers import CLIPTokenizer, T5Tokenizer, T5TokenizerFast

from invokeai.backend.image_util.depth_anything.depth_anything_pipeline import DepthAnythingPipeline
from invokeai.backend.image_util.grounding_dino.grounding_dino_pipeline import GroundingDinoPipeline
from invokeai.backend.image_util.segment_anything.segment_anything_pipeline import SegmentAnythingPipeline
from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.model_manager.taxonomy import AnyModel
from invokeai.backend.onnx.onnx_runtime import IAIOnnxRuntimeModel
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.spandrel_image_to_image_model import SpandrelImageToImageModel
from invokeai.backend.textual_inversion import TextualInversionModelRaw
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


def calc_model_size_by_data(logger: logging.Logger, model: AnyModel) -> int:
    """Get size of a model in memory in bytes."""
    # TODO(ryand): We should create a CacheableModel interface for all models, and move the size calculations down to
    # the models themselves.
    if isinstance(model, DiffusionPipeline):
        return _calc_pipeline_by_data(model)
    elif isinstance(model, torch.nn.Module):
        return calc_module_size(model)
    elif isinstance(model, IAIOnnxRuntimeModel):
        return _calc_onnx_model_by_data(model)
    elif isinstance(model, SchedulerMixin):
        return 0
    elif isinstance(model, CLIPTokenizer):
        # TODO(ryand): Accurately calculate the tokenizer's size. It's small enough that it shouldn't matter for now.
        return 0
    elif isinstance(
        model,
        (
            TextualInversionModelRaw,
            IPAdapter,
            ModelPatchRaw,
            SpandrelImageToImageModel,
            GroundingDinoPipeline,
            SegmentAnythingPipeline,
            DepthAnythingPipeline,
        ),
    ):
        return model.calc_size()
    elif isinstance(model, ort.InferenceSession):
        if model._model_bytes is not None:
            # If the model is already loaded, return the size of the model bytes
            return len(model._model_bytes)
        elif model._model_path is not None:
            # If the model is not loaded, return the size of the model path
            return calc_model_size_by_fs(Path(model._model_path))
        else:
            # If neither is available, return 0
            return 0
    elif isinstance(
        model,
        (
            T5TokenizerFast,
            T5Tokenizer,
        ),
    ):
        # HACK(ryand): len(model) just returns the vocabulary size, so this is blatantly wrong. It should be small
        # relative to the text encoder that it's used with, so shouldn't matter too much, but we should fix this at some
        # point.
        return len(model)
    else:
        # TODO(ryand): Promote this from a log to an exception once we are confident that we are handling all of the
        # supported model types.
        logger.warning(
            f"Failed to calculate model size for unexpected model type: {type(model)}. The model will be treated as "
            "having size 0."
        )
        return 0


def _calc_pipeline_by_data(pipeline: DiffusionPipeline) -> int:
    res = 0
    assert hasattr(pipeline, "components")
    for submodel_key in pipeline.components.keys():
        submodel = getattr(pipeline, submodel_key)
        if submodel is not None and isinstance(submodel, torch.nn.Module):
            res += calc_module_size(submodel)
    return res


def calc_module_size(model: torch.nn.Module) -> int:
    """Calculate the size (in bytes) of a torch.nn.Module."""
    mem_params = sum([calc_tensor_size(param) for param in model.parameters()])
    mem_bufs = sum([calc_tensor_size(buf) for buf in model.buffers()])
    return mem_params + mem_bufs


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

    if not variant:  # ModelRepoVariant.DEFAULT evaluates to empty string for compatability with HF
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
