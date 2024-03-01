# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Select the files from a HuggingFace repository needed for a particular model variant.

Usage:
```
from invokeai.backend.model_manager.util.select_hf_files import select_hf_model_files
from invokeai.backend.model_manager.metadata.fetch import HuggingFaceMetadataFetch

metadata = HuggingFaceMetadataFetch().from_url("https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0")
files_to_download = select_hf_model_files(metadata.files, variant='onnx')
```
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..config import ModelRepoVariant


def filter_files(
    files: List[Path],
    variant: Optional[ModelRepoVariant] = None,
    subfolder: Optional[Path] = None,
) -> List[Path]:
    """
    Take a list of files in a HuggingFace repo root and return paths to files needed to load the model.

    :param files: List of files relative to the repo root.
    :param subfolder: Filter by the indicated subfolder.
    :param variant: Filter by files belonging to a particular variant, such as fp16.

    The file list can be obtained from the `files` field of HuggingFaceMetadata,
    as defined in `invokeai.backend.model_manager.metadata.metadata_base`.
    """
    variant = variant or ModelRepoVariant.DEFAULT
    paths: List[Path] = []
    root = files[0].parts[0]

    # if the subfolder is a single file, then bypass the selection and just return it
    if subfolder and subfolder.suffix in [".safetensors", ".bin", ".onnx", ".xml", ".pth", ".pt", ".ckpt", ".msgpack"]:
        return [root / subfolder]

    # Start by filtering on model file extensions, discarding images, docs, etc
    for file in files:
        if file.name.endswith((".json", ".txt")):
            paths.append(file)
        elif file.name.endswith(
            (
                "learned_embeds.bin",
                "ip_adapter.bin",
                "lora_weights.safetensors",
                "weights.pb",
                "onnx_data",
            )
        ):
            paths.append(file)
        # BRITTLENESS WARNING!!
        # Diffusers models always seem to have "model" in their name, and the regex filter below is applied to avoid
        # downloading random checkpoints that might also be in the repo. However there is no guarantee
        # that a checkpoint doesn't contain "model" in its name, and no guarantee that future diffusers models
        # will adhere to this naming convention, so this is an area to be careful of.
        elif re.search(r"model(\.[^.]+)?\.(safetensors|bin|onnx|xml|pth|pt|ckpt|msgpack)$", file.name):
            paths.append(file)

    # limit search to subfolder if requested
    if subfolder:
        subfolder = root / subfolder
        paths = [x for x in paths if x.parent == Path(subfolder)]

    # _filter_by_variant uniquifies the paths and returns a set
    return sorted(_filter_by_variant(paths, variant))


def _filter_by_variant(files: List[Path], variant: ModelRepoVariant) -> Set[Path]:
    """Select the proper variant files from a list of HuggingFace repo_id paths."""
    result = set()
    basenames: Dict[Path, Path] = {}
    for path in files:
        if path.suffix in [".onnx", ".pb", ".onnx_data"]:
            if variant == ModelRepoVariant.ONNX:
                result.add(path)

        elif "openvino_model" in path.name:
            if variant == ModelRepoVariant.OPENVINO:
                result.add(path)

        elif "flax_model" in path.name:
            if variant == ModelRepoVariant.FLAX:
                result.add(path)

        elif path.suffix in [".json", ".txt"]:
            result.add(path)

        elif path.suffix in [".bin", ".safetensors", ".pt", ".ckpt"] and variant in [
            ModelRepoVariant.FP16,
            ModelRepoVariant.FP32,
            ModelRepoVariant.DEFAULT,
        ]:
            parent = path.parent
            suffixes = path.suffixes
            if len(suffixes) == 2:
                variant_label, suffix = suffixes
                basename = parent / Path(path.stem).stem
            else:
                variant_label = ""
                suffix = suffixes[0]
                basename = parent / path.stem

            if previous := basenames.get(basename):
                if (
                    previous.suffix != ".safetensors" and suffix == ".safetensors"
                ):  # replace non-safetensors with safetensors when available
                    basenames[basename] = path
                if variant_label == f".{variant}":
                    basenames[basename] = path
                elif not variant_label and variant in [ModelRepoVariant.FP32, ModelRepoVariant.DEFAULT]:
                    basenames[basename] = path
            else:
                basenames[basename] = path

        else:
            continue

    for v in basenames.values():
        result.add(v)

    # If one of the architecture-related variants was specified and no files matched other than
    # config and text files then we return an empty list
    if (
        variant
        and variant in [ModelRepoVariant.ONNX, ModelRepoVariant.OPENVINO, ModelRepoVariant.FLAX]
        and not any(variant.value in x.name for x in result)
    ):
        return set()

    # Prune folders that contain just a `config.json`. This happens when
    # the requested variant (e.g. "onnx") is missing
    directories: Dict[Path, int] = {}
    for x in result:
        if not x.parent:
            continue
        directories[x.parent] = directories.get(x.parent, 0) + 1

    return {x for x in result if directories[x.parent] > 1 or x.name != "config.json"}
