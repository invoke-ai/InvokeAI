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
from dataclasses import dataclass
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
    variant = variant or ModelRepoVariant.Default
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
                # jennifer added a bunch of these, probably will break something
                ".safetensors",
                ".bin",
                ".onnx",
                ".xml",
                ".pth",
                ".pt",
                ".ckpt",
                ".msgpack",
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
    # jennifer removed the filter since it removed models, probably will break something but i dont understand why it removes valid models :|
    return sorted(paths)


@dataclass
class SubfolderCandidate:
    path: Path
    score: int


def _filter_by_variant(files: List[Path], variant: ModelRepoVariant) -> Set[Path]:
    """Select the proper variant files from a list of HuggingFace repo_id paths."""
    result: set[Path] = set()
    subfolder_weights: dict[Path, list[SubfolderCandidate]] = {}
    for path in files:
        if path.suffix in [".onnx", ".pb", ".onnx_data"]:
            if variant == ModelRepoVariant.ONNX:
                result.add(path)

        elif "openvino_model" in path.name:
            if variant == ModelRepoVariant.OpenVINO:
                result.add(path)

        elif "flax_model" in path.name:
            if variant == ModelRepoVariant.Flax:
                result.add(path)

        elif path.suffix in [".json", ".txt"]:
            result.add(path)

        elif variant in [
            ModelRepoVariant.FP16,
            ModelRepoVariant.FP32,
            ModelRepoVariant.Default,
        ] and path.suffix in [".bin", ".safetensors", ".pt", ".ckpt"]:
            # For weights files, we want to select the best one for each subfolder. For example, we may have multiple
            # text encoders:
            #
            # - text_encoder/model.fp16.safetensors
            # - text_encoder/model.safetensors
            # - text_encoder/pytorch_model.bin
            # - text_encoder/pytorch_model.fp16.bin
            #
            # We prefer safetensors over other file formats and an exact variant match. We'll score each file based on
            # variant and format and select the best one.

            parent = path.parent
            score = 0

            if path.suffix == ".safetensors":
                score += 1

            candidate_variant_label = path.suffixes[0] if len(path.suffixes) == 2 else None

            # Some special handling is needed here if there is not an exact match and if we cannot infer the variant
            # from the file name. In this case, we only give this file a point if the requested variant is FP32 or DEFAULT.
            if candidate_variant_label == f".{variant}" or (
                not candidate_variant_label and variant in [ModelRepoVariant.FP32, ModelRepoVariant.Default]
            ):
                score += 1

            if parent not in subfolder_weights:
                subfolder_weights[parent] = []

            subfolder_weights[parent].append(SubfolderCandidate(path=path, score=score))

        else:
            continue

    for candidate_list in subfolder_weights.values():
        highest_score_candidate = max(candidate_list, key=lambda candidate: candidate.score)
        if highest_score_candidate:
            result.add(highest_score_candidate.path)

    # If one of the architecture-related variants was specified and no files matched other than
    # config and text files then we return an empty list
    if (
        variant
        and variant in [ModelRepoVariant.ONNX, ModelRepoVariant.OpenVINO, ModelRepoVariant.Flax]
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
