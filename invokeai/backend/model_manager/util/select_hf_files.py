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

from invokeai.backend.model_manager.taxonomy import ModelRepoVariant


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
                "lora_weights.safetensors",
                "weights.pb",
                "onnx_data",
                "spiece.model",  # Added for `black-forest-labs/FLUX.1-schnell`.
            )
        ):
            paths.append(file)
        # BRITTLENESS WARNING!!
        # Diffusers models always seem to have "model" in their name, and the regex filter below is applied to avoid
        # downloading random checkpoints that might also be in the repo. However there is no guarantee
        # that a checkpoint doesn't contain "model" in its name, and no guarantee that future diffusers models
        # will adhere to this naming convention, so this is an area to be careful of.
        elif re.search(r"model.*\.(safetensors|bin|onnx|xml|pth|pt|ckpt|msgpack)$", file.name):
            paths.append(file)

    # limit search to subfolder if requested
    if subfolder:
        subfolder = root / subfolder
        paths = [x for x in paths if Path(subfolder) in x.parents]

    # _filter_by_variant uniquifies the paths and returns a set
    return sorted(_filter_by_variant(paths, variant))


@dataclass
class SubfolderCandidate:
    path: Path
    score: int


def _filter_by_variant(files: List[Path], variant: ModelRepoVariant) -> Set[Path]:
    """Select the proper variant files from a list of HuggingFace repo_id paths."""
    result: set[Path] = set()
    subfolder_weights: dict[Path, list[SubfolderCandidate]] = {}
    safetensors_detected = False
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

        # Note: '.model' was added to support:
        # https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/768d12a373ed5cc9ef9a9dea7504dc09fcc14842/tokenizer_2/spiece.model
        elif path.suffix in [".json", ".txt", ".model"]:
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

            if safetensors_detected and path.suffix == ".bin":
                continue

            parent = path.parent
            score = 0

            if path.suffix == ".safetensors":
                safetensors_detected = True
                if parent in subfolder_weights:
                    subfolder_weights[parent] = [sfc for sfc in subfolder_weights[parent] if sfc.path.suffix != ".bin"]
                score += 1

            candidate_variant_label = path.suffixes[0] if len(path.suffixes) == 2 else None

            # Some special handling is needed here if there is not an exact match and if we cannot infer the variant
            # from the file name. In this case, we only give this file a point if the requested variant is FP32 or DEFAULT.
            if (
                variant is not ModelRepoVariant.Default
                and candidate_variant_label
                and candidate_variant_label.startswith(f".{variant.value}")
            ) or (not candidate_variant_label and variant in [ModelRepoVariant.FP32, ModelRepoVariant.Default]):
                score += 1

            if parent not in subfolder_weights:
                subfolder_weights[parent] = []

            subfolder_weights[parent].append(SubfolderCandidate(path=path, score=score))

        else:
            continue

    for candidate_list in subfolder_weights.values():
        # Check if at least one of the files has the explicit fp16 variant.
        at_least_one_fp16 = False
        for candidate in candidate_list:
            if len(candidate.path.suffixes) == 2 and candidate.path.suffixes[0].startswith(".fp16"):
                at_least_one_fp16 = True
                break

        if not at_least_one_fp16:
            # If none of the candidates in this candidate_list have the explicit fp16 variant label, then this
            # candidate_list probably doesn't adhere to the variant naming convention that we expected. In this case,
            # we'll simply keep all the candidates. An example of a model that hits this case is
            # `black-forest-labs/FLUX.1-schnell` (as of commit 012d2fd).
            for candidate in candidate_list:
                result.add(candidate.path)

        # The candidate_list seems to have the expected variant naming convention. We'll select the highest scoring
        # candidate.
        highest_score_candidate = max(candidate_list, key=lambda candidate: candidate.score)
        if highest_score_candidate:
            pattern = r"^(.*?)-\d+-of-\d+(\.\w+)$"
            match = re.match(pattern, highest_score_candidate.path.as_posix())
            if match:
                for candidate in candidate_list:
                    if candidate.path.as_posix().startswith(match.group(1)) and candidate.path.as_posix().endswith(
                        match.group(2)
                    ):
                        result.add(candidate.path)
            else:
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
