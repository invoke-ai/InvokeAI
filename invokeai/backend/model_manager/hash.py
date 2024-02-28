# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Fast hashing of diffusers and checkpoint-style models.

Usage:
from invokeai.backend.model_managre.model_hash import FastModelHash
>>> FastModelHash.hash('/home/models/stable-diffusion-v1.5')
'a8e693a126ea5b831c96064dc569956f'
"""
import hashlib
import os
from pathlib import Path
from typing import Callable, Literal, Union

from blake3 import blake3

MODEL_FILE_EXTENSIONS = (".ckpt", ".safetensors", ".bin", ".pt", ".pth")

ALGORITHM = Literal[
    "md5",
    "sha1",
    "sha1_fast",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "blake2b",
    "blake2s",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "shake_128",
    "shake_256",
    "blake3",
]


class ModelHash:
    """
    Creates a hash of a model using a specified algorithm.

    :param algorithm: Hashing algorithm to use. Defaults to BLAKE3.

    If the model is a single file, it is hashed directly using the provided algorithm.

    If the model is a directory, each model weights file in the directory is hashed using the provided algorithm.

    Only files with the following extensions are hashed: .ckpt, .safetensors, .bin, .pt, .pth

    The final hash is computed by hashing the hashes of all model files in the directory using BLAKE3, ensuring
    that directory hashes are never weaker than the file hashes.

    Usage

    ```py
    ModelHash().hash("path/to/some/model.safetensors")
    ModelHash("md5").hash("path/to/model/dir/")
    ```
    """

    def __init__(self, algorithm: ALGORITHM = "blake3") -> None:
        if algorithm == "blake3":
            self._hash_file = self._blake3
        elif algorithm == "sha1_fast":
            self._hash_file = self._sha1_fast
        elif algorithm in hashlib.algorithms_available:
            self._hash_file = self._get_hashlib(algorithm)
        else:
            raise ValueError(f"Algorithm {algorithm} not available")

    def hash(self, model_location: Union[str, Path]) -> str:
        """
        Return hexdigest string for model located at model_location.

        If model_location is a directory, the hash is computed by hashing the hashes of all model files in the
        directory. The final composite hash is always computed using BLAKE3.

        :param model_location: Path to the model
        """

        model_location = Path(model_location)
        if model_location.is_file():
            return self._hash_file(model_location)
        elif model_location.is_dir():
            return self._hash_dir(model_location)
        else:
            raise OSError(f"Not a valid file or directory: {model_location}")

    def _hash_dir(self, model_location: Path) -> str:
        """Compute the hash for all files in a directory and return a hexdigest."""
        model_component_paths = self._get_file_paths(model_location)

        component_hashes: list[str] = []
        for component in sorted(model_component_paths):
            component_hashes.append(self._hash_file(component))

        # BLAKE3 is cryptographically secure. We may as well fall back on a secure algorithm
        # for the composite hash
        composite_hasher = blake3()
        for h in component_hashes:
            composite_hasher.update(h.encode("utf-8"))
        return composite_hasher.hexdigest()

    @classmethod
    def _get_file_paths(cls, dir: Path) -> list[Path]:
        """Return a list of all model files in the directory."""
        files: list[Path] = []
        for root, _dirs, _files in os.walk(dir):
            for file in _files:
                if file.endswith(MODEL_FILE_EXTENSIONS):
                    files.append(Path(root, file))
        return files

    @staticmethod
    def _blake3(file_path: Path) -> str:
        """Hashes a file using BLAKE3"""
        file_hasher = blake3(max_threads=blake3.AUTO)
        file_hasher.update_mmap(file_path)
        return file_hasher.hexdigest()

    @staticmethod
    def _sha1_fast(file_path: Path) -> str:
        """Hashes a file using SHA1, but with a block size of 2**16.
        The result is not a correct SHA1 hash for the file, due to the padding introduced by the block size.
        The algorithm is, however, very fast."""
        BLOCK_SIZE = 2**16
        file_hash = hashlib.sha1()
        with open(file_path, "rb") as f:
            data = f.read(BLOCK_SIZE)
            file_hash.update(data)
        return file_hash.hexdigest()

    @staticmethod
    def _get_hashlib(algorithm: ALGORITHM) -> Callable[[Path], str]:
        """Hashes a file using a hashlib algorithm"""

        def hasher(file_path: Path) -> str:
            file_hasher = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                file_hasher.update(f.read())
            return file_hasher.hexdigest()

        return hasher
