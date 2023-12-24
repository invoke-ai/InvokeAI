# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

"""
This module fetches model metadata objects from the HuggingFace model repository,
using either a `repo_id` or the model page URL.

Usage:

from invokeai.backend.model_manager.metadata.fetch import HuggingFaceMetadataFetch

fetcher = HuggingFaceMetadataFetch()
metadata = fetcher.from_url("https://huggingface.co/stabilityai/sdxl-turbo")
print(metadata.tags)
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from huggingface_hub import HfApi, configure_http_backend, hf_hub_url
from huggingface_hub.utils._errors import RepositoryNotFoundError
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from invokeai.app.services.model_records import UnknownModelException
from invokeai.backend.model_manager.config import DiffusersVariant

from ..metadata_base import AnyModelRepoMetadata, AnyModelRepoMetadataValidator, HuggingFaceMetadata
from .fetch_base import ModelMetadataFetchBase

HF_MODEL_RE = r"https?://huggingface.co/([\w\-.]+/[\w\-.]+)"


class HuggingFaceMetadataFetch(ModelMetadataFetchBase):
    """Fetch model metadata from HuggingFace."""

    _requests: Session

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the fetcher with an optional requests.sessions.Session object.

        By providing a configurable Session object, we can support unit tests on
        this module without an internet connection.
        """
        self._requests = session or requests.Session()
        configure_http_backend(backend_factory=lambda: self._requests)

    @classmethod
    def from_json(cls, json: str) -> HuggingFaceMetadata:
        """Given the JSON representation of the metadata, return the corresponding Pydantic object."""
        metadata = AnyModelRepoMetadataValidator.validate_json(json)
        assert isinstance(metadata, HuggingFaceMetadata)
        return metadata

    def from_id(self, id: str) -> AnyModelRepoMetadata:
        """Return a HuggingFaceMetadata object given the model's repo_id."""
        try:
            model_info = HfApi().model_info(repo_id=id, files_metadata=True)
        except RepositoryNotFoundError as excp:
            raise UnknownModelException(f"'{id}' not found. See trace for details.") from excp

        _, name = id.split("/")
        return HuggingFaceMetadata(
            id=model_info.modelId,
            author=model_info.author,
            name=name,
            last_modified=model_info.lastModified,
            tag_dict=model_info.card_data.to_dict(),
            tags=model_info.tags,
            files=[Path(x.rfilename) for x in model_info.siblings],
        )

    def from_url(self, url: AnyHttpUrl) -> AnyModelRepoMetadata:
        """
        Return a HuggingFaceMetadata object given the model's web page URL.

        In the case of an invalid or missing URL, raises a ModelNotFound exception.
        """
        if match := re.match(HF_MODEL_RE, str(url)):
            repo_id = match.group(1)
            return self.from_id(repo_id)
        else:
            raise UnknownModelException(f"'{url}' does not look like a HuggingFace model page")

    def list_download_urls(
        self,
        metadata: HuggingFaceMetadata,
        variant: Optional[DiffusersVariant] = None,
        subfolder: Optional[Path] = None,
    ) -> List[Tuple[AnyHttpUrl, Path]]:
        """For a HuggingFace model, return a list of tuples of (URL, Path) needed to download model."""
        requests = self._requests
        paths = self._filter_files(metadata.files, variant, subfolder)  #  all files in the model

        prefix = f"{subfolder}/" if subfolder else ""

        # the next step reads model_index.json to determine which subdirectories belong
        # to the model
        if Path(f"{prefix}model_index.json") in paths:
            url = hf_hub_url(metadata.id, filename="model_index.json", subfolder=subfolder)
            resp = requests.get(url)
            resp.raise_for_status()
            submodels = resp.json()
            paths = [Path(subfolder or "", x) for x in paths if Path(x).parent.as_posix() in submodels]
            paths.insert(0, Path(f"{prefix}model_index.json"))

        return [
            (AnyHttpUrl(hf_hub_url(metadata.id, filename=x.as_posix())), Path(metadata.name, x.relative_to(prefix)))
            for x in paths
        ]

    def _filter_files(
        self,
        files: List[Path],
        variant: Optional[DiffusersVariant] = None,
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
        if not variant:
            variant = DiffusersVariant.DEFAULT
        paths: List[Path] = []

        # Start by filtering on model file extensions, discarding images, docs, etc
        for file in files:
            if file.name.endswith((".json", ".txt")):
                paths.append(file)
            elif file.name.endswith(("learned_embeds.bin", "ip_adapter.bin")):
                paths.append(file)
            elif re.search(r"model(\.[^.]+)?\.(safetensors|bin|onnx|xml|pth|pt|ckpt|msgpack)$", file.name):
                paths.append(file)

        # limit search to subfolder if requested
        if subfolder:
            paths = [x for x in paths if x.parent == Path(subfolder)]

        # _filter_by_variant uniquifies the paths and returns a set
        return sorted(self._filter_by_variant(paths, variant))

    def _filter_by_variant(
        self, files: List[Path], variant: Optional[DiffusersVariant] = DiffusersVariant.DEFAULT
    ) -> Set[Path]:
        """Select the proper variant files from a list of HuggingFace repo_id paths."""
        result = set()
        basenames: Dict[Path, Path] = {}
        for path in files:
            if path.suffix == ".onnx":
                if variant == DiffusersVariant.ONNX:
                    result.add(path)

            elif "openvino_model" in path.name:
                if variant == DiffusersVariant.OPENVINO:
                    result.add(path)

            elif "flax_model" in path.name:
                if variant == DiffusersVariant.FLAX:
                    result.add(path)

            elif path.suffix in [".json", ".txt"]:
                result.add(path)

            elif path.suffix in [".bin", ".safetensors", ".pt", ".ckpt"] and variant in [
                DiffusersVariant.FP16,
                DiffusersVariant.DEFAULT,
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
                    if previous.suffix != ".safetensors" and suffix == ".safetensors":
                        basenames[basename] = path
                    if variant_label == f".{variant}":
                        basenames[basename] = path
                    elif not variant_label and variant == DiffusersVariant.DEFAULT:
                        basenames[basename] = path
                else:
                    basenames[basename] = path

            else:
                continue

        for v in basenames.values():
            result.add(v)

        # Prune folders that contain just a `config.json`. This happens when
        # the requested variant (e.g. "onnx") is missing
        directories: Dict[Path, int] = {}
        for x in result:
            if not x.parent:
                continue
            directories[x.parent] = directories.get(x.parent, 0) + 1

        return {x for x in result if directories[x.parent] > 1 or x.name != "config.json"}
