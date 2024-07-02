import json
from pathlib import Path
from typing import Optional

import torch


def calc_module_size(model: torch.nn.Module) -> int:
    """Estimate the size of a torch.nn.Module in bytes."""
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem: int = mem_params + mem_bufs  # in bytes
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
