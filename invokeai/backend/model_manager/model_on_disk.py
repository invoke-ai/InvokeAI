from pathlib import Path
from typing import Any, Optional, TypeAlias

import safetensors.torch
import torch
from picklescan.scanner import scan_file_path
from safetensors import safe_open

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS, ModelHash
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.silence_warnings import SilenceWarnings

StateDict: TypeAlias = dict[str | int, Any]  # When are the keys int?

logger = InvokeAILogger.get_logger()


class ModelOnDisk:
    """A utility class representing a model stored on disk."""

    def __init__(self, path: Path, hash_algo: HASHING_ALGORITHMS = "blake3_single"):
        self.path = path
        if self.path.suffix in {".safetensors", ".bin", ".pt", ".ckpt"}:
            self.name = path.stem
        else:
            self.name = path.name
        self.hash_algo = hash_algo
        # Having a cache helps users of ModelOnDisk (i.e. configs) to save state
        # This prevents redundant computations during matching and parsing
        self.cache = {"_CACHED_STATE_DICTS": {}}

    def hash(self) -> str:
        return ModelHash(algorithm=self.hash_algo).hash(self.path)

    def size(self) -> int:
        if self.path.is_file():
            return self.path.stat().st_size
        return sum(file.stat().st_size for file in self.path.rglob("*"))

    def weight_files(self) -> set[Path]:
        if self.path.is_file():
            return {self.path}
        extensions = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf"}
        return {f for f in self.path.rglob("*") if f.suffix in extensions}

    def metadata(self, path: Optional[Path] = None) -> dict[str, str]:
        try:
            with safe_open(self.path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                assert isinstance(metadata, dict)
                return metadata
        except Exception:
            return {}

    def repo_variant(self) -> Optional[ModelRepoVariant]:
        if self.path.is_file():
            return None

        weight_files = list(self.path.glob("**/*.safetensors"))
        weight_files.extend(list(self.path.glob("**/*.bin")))
        for x in weight_files:
            if ".fp16" in x.suffixes:
                return ModelRepoVariant.FP16
            if "openvino_model" in x.name:
                return ModelRepoVariant.OpenVINO
            if "flax_model" in x.name:
                return ModelRepoVariant.Flax
            if x.suffix == ".onnx":
                return ModelRepoVariant.ONNX
        return ModelRepoVariant.Default

    def load_state_dict(self, path: Optional[Path] = None) -> StateDict:
        sd_cache = self.cache["_CACHED_STATE_DICTS"]

        if path in sd_cache:
            return sd_cache[path]

        path = self.resolve_weight_file(path)

        with SilenceWarnings():
            if path.suffix.endswith((".ckpt", ".pt", ".pth", ".bin")):
                scan_result = scan_file_path(path)
                if scan_result.infected_files != 0:
                    if get_config().unsafe_disable_picklescan:
                        logger.warning(
                            f"The model {path.stem} is potentially infected by malware, but picklescan is disabled. "
                            "Proceeding with caution."
                        )
                    else:
                        raise RuntimeError(
                            f"The model {path.stem} is potentially infected by malware. Aborting import."
                        )
                if scan_result.scan_err:
                    if get_config().unsafe_disable_picklescan:
                        logger.warning(
                            f"Error scanning the model at {path.stem} for malware, but picklescan is disabled. "
                            "Proceeding with caution."
                        )
                    else:
                        raise RuntimeError(f"Error scanning the model at {path.stem} for malware. Aborting import.")
                checkpoint = torch.load(path, map_location="cpu")
                assert isinstance(checkpoint, dict)
            elif path.suffix.endswith(".gguf"):
                checkpoint = gguf_sd_loader(path, compute_dtype=torch.float32)
            elif path.suffix.endswith(".safetensors"):
                checkpoint = safetensors.torch.load_file(path)
            else:
                raise ValueError(f"Unrecognized model extension: {path.suffix}")

        state_dict = checkpoint.get("state_dict", checkpoint)
        sd_cache[path] = state_dict
        return state_dict

    def resolve_weight_file(self, path: Optional[Path] = None) -> Path:
        if not path:
            weight_files = list(self.weight_files())
            match weight_files:
                case []:
                    raise ValueError("No weight files found for this model")
                case [p]:
                    return p
                case ps if len(ps) >= 2:
                    raise ValueError(
                        f"Multiple weight files found for this model: {ps}. "
                        f"Please specify the intended file using the 'path' argument"
                    )
        return path
