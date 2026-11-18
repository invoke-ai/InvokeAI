import json
import os
from pathlib import Path
from typing import Any, Optional, TypeAlias

import safetensors.torch
import torch
from gguf import GGUFValueType
from picklescan.scanner import scan_file_path
from safetensors import safe_open

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS, ModelHash
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant
from invokeai.backend.quantization.gguf.loaders import WrappedGGUFReader, gguf_header_state_dict
from invokeai.backend.quantization.sdnq.loaders import sdnq_sd_loader
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.silence_warnings import SilenceWarnings

StateDict: TypeAlias = dict[str | int, Any]  # When are the keys int?

logger = InvokeAILogger.get_logger()


SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E4M3FNUZ": torch.float8_e4m3fnuz,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E5M2FNUZ": torch.float8_e5m2fnuz,
    "U16": torch.uint16,
    "I16": torch.int16,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "U32": torch.uint32,
    "I32": torch.int32,
    "F32": torch.float32,
    "U64": torch.uint64,
    "I64": torch.int64,
    "F64": torch.float64,
    "C64": torch.complex64,
}
"""safetensors dtype names mapped to their torch equivalents: every dtype `safetensors.torch` reads but F4, because the
single-file loaders read through the header, and a missing name fails a file `load_file` loads -- at generation, since
identification falls back to the full load and installs it. The sub-byte formats (F4, F6_E2M3, F6_E3M2) are absent:
their header shapes count 4- and 6-bit elements, which no torch dtype stores one per element. `load_file` reads F4 as
`float4_e2m1fn_x2` with the last dimension halved; no loader casts such a tensor, so it is refused here by name."""

# MXFP8 exponent grids, which loaders refuse by name (`fp8_scaled.reject_undecoded_mx_scale`), so they have to get as
# far as that check. Guarded like `fp8_scaled._MX_SCALE_DTYPES`: older torch builds lack the dtype.
if hasattr(torch, "float8_e8m0fnu"):
    SAFETENSORS_DTYPES["F8_E8M0"] = torch.float8_e8m0fnu


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read a safetensors file's JSON header: dtype, shape and data offsets per tensor key, without `__metadata__`.

    Raises:
        OSError: The file could not be read.
        ValueError: The header is not valid JSON (an empty or truncated file, or not safetensors at all).
    """
    with open(path, "rb") as f:
        header_length = int.from_bytes(f.read(8), "little")
        # Any 8 bytes decode to a length; for a file that is not safetensors that length is garbage, and reading it
        # would try to allocate it.
        if header_length > os.fstat(f.fileno()).st_size - 8:
            raise ValueError(f"{path} is not a safetensors file: its header length exceeds the file size")
        header = json.loads(f.read(header_length))

    if not isinstance(header, dict):
        raise ValueError(f"{path} is not a safetensors file: its header is not a JSON object")
    header.pop("__metadata__", None)
    return header


def safetensors_meta_state_dict(path: Path) -> StateDict:
    """Build a state dict of `meta` tensors from a safetensors file's header.

    Identification only ever reads keys, shapes and dtypes off a state dict - never tensor data - so loading
    the tensors costs the model's full size in RAM for every probe and buys nothing. The header is a few KB at
    the front of the file, and `meta` tensors carry the shape and dtype without any storage behind them. A loader
    uses it the same way, to decide which tensors to read and how much RAM they need before reading any.

    Raises:
        OSError: The file could not be read.
        ValueError: The header could not be parsed, or names a dtype with no entry in `SAFETENSORS_DTYPES`.
            Identification falls back to loading the file for real.
    """
    header = read_safetensors_header(path)
    state_dict: StateDict = {}
    for key, info in header.items():
        dtype = SAFETENSORS_DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"{path.name} stores '{key}' as {info['dtype']}, a dtype this build cannot read.")
        state_dict[key] = torch.empty(info["shape"], dtype=dtype, device="meta")
    return state_dict


def unwrap_state_dict(checkpoint: dict[Any, Any]) -> StateDict:
    """A checkpoint's weights: training checkpoints nest them under `state_dict`.

    Identification reads its state dicts through this (`ModelOnDisk.load_state_dict`). A loader that reads a pickle
    itself unwraps through it too, so it loads the weights identification accepted.
    """
    return checkpoint.get("state_dict", checkpoint)


def _is_sdnq_safetensors(path: Path) -> bool:
    """Check if a safetensors file contains SDNQ-quantized weights by checking for weight+scale pairs."""
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for key in keys:
                if key.endswith(".weight"):
                    base = key[:-7]
                    if f"{base}.scale" in keys:
                        return True
    except Exception:
        pass
    return False


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
        self._state_dict_cache: dict[Path, Any] = {}
        self._metadata_cache: dict[Path, Any] = {}
        self._metadata_errors: dict[Path, str] = {}

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
        return {f for f in self.path.rglob("*") if f.suffix in extensions and f.is_file()}

    def metadata(self, path: Optional[Path] = None) -> dict[str, Any]:
        """The file's metadata: safetensors' string map, or a GGUF's scalar fields.

        GGUF values keep their type -- strings, numbers, booleans -- because llama.cpp records hyperparameters
        as numbers. Array fields (the tokenizer vocabulary, for one) are left out: nothing identifies by them,
        and they are most of a header's size.
        """
        path = path or self.path
        if path in self._metadata_cache:
            return self._metadata_cache[path]
        try:
            if path.suffix == ".gguf":
                with WrappedGGUFReader(path) as reader:
                    metadata = {
                        name: field.contents()
                        for name, field in reader.fields.items()
                        if field.types and field.types[0] != GGUFValueType.ARRAY
                    }
            else:
                with safe_open(path, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
                    assert isinstance(metadata, dict)
        except Exception as e:
            metadata = {}
            # The message only: a stored exception would pin the traceback frames.
            self._metadata_errors[path] = f"{type(e).__name__}: {e}"

        self._metadata_cache[path] = metadata
        return metadata

    def metadata_read_error(self, path: Optional[Path] = None) -> Optional[str]:
        """Why `metadata()` could not read the file, or None if it could.

        `metadata()` returns `{}` for an unreadable file, which is indistinguishable from a readable one
        without metadata; this tells the two apart without a second parse.
        """
        path = path or self.path
        self.metadata(path)
        return self._metadata_errors.get(path)

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
        if path in self._state_dict_cache:
            return self._state_dict_cache[path]

        path = self.resolve_weight_file(path)

        if path in self._state_dict_cache:
            return self._state_dict_cache[path]

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
                # `meta` for the same reason the safetensors branch reads only the header: identification
                # needs the shapes, not the weights, and torch skips reading the storages entirely.
                checkpoint = torch.load(path, map_location="meta")
                assert isinstance(checkpoint, dict)
            elif path.suffix.endswith(".gguf"):
                # Shapes and quantization types from the header, like the safetensors branch: the tensor data
                # is never read. Whether a Q8_CR or Q4_CR file's quantization is supported is decided from its
                # metadata by `ModelConfigFactory`, once, not per candidate.
                checkpoint = gguf_header_state_dict(path, compute_dtype=torch.float32)
            elif path.suffix.endswith(".safetensors"):
                if _is_sdnq_safetensors(path):
                    checkpoint = sdnq_sd_loader(path, compute_dtype=torch.float32)
                else:
                    try:
                        checkpoint = safetensors_meta_state_dict(path)
                    except Exception:
                        logger.debug(f"Could not read the safetensors header of {path}, loading the whole file")
                        checkpoint = safetensors.torch.load_file(path)
            else:
                raise ValueError(f"Unrecognized model extension: {path.suffix}")

        state_dict = unwrap_state_dict(checkpoint)

        # Normalize PEFT named-adapter keys (e.g. `lora_A.default.weight` → `lora_A.weight`).
        # Pattern is LoRA-specific, so this is a no-op for non-LoRA state dicts.
        from invokeai.backend.patches.lora_conversions.peft_adapter_utils import normalize_peft_adapter_names

        state_dict = normalize_peft_adapter_names(state_dict)

        self._state_dict_cache[path] = state_dict
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
