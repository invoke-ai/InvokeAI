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
from typing import Callable, Literal, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed


from blake3 import blake3

MODEL_FILE_EXTENSIONS = (".ckpt", ".safetensors", ".bin", ".pt", ".pth")

ALGORITHM = Literal[
    "md5",
    "sha1",
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

    Args:
        algorithm: Hashing algorithm to use. Defaults to BLAKE3.
        file_filter: A function that takes a file name and returns True if the file should be included in the hash.

    If the model is a single file, it is hashed directly using the provided algorithm.

    If the model is a directory, each model weights file in the directory is hashed using the provided algorithm.

    Only files with the following extensions are hashed: .ckpt, .safetensors, .bin, .pt, .pth

    The final hash is computed by hashing the hashes of all model files in the directory using BLAKE3, ensuring
    that directory hashes are never weaker than the file hashes.

    Usage:
        ```py
        # BLAKE3 hash
        ModelHash().hash("path/to/some/model.safetensors")
        # MD5
        ModelHash("md5").hash("path/to/model/dir/")
        ```
    """

    def __init__(self, algorithm: ALGORITHM = "blake3", file_filter: Optional[Callable[[str], bool]] = None) -> None:
        if algorithm == "blake3":
            self._hash_file = self._blake3
        elif algorithm in hashlib.algorithms_available:
            self._hash_file = self._get_hashlib(algorithm)
        else:
            raise ValueError(f"Algorithm {algorithm} not available")

        self._file_filter = file_filter or self._default_file_filter

    def hash(self, model_path: Union[str, Path]) -> str:
        """
        Return hexdigest of hash of model located at model_path using the algorithm provided at class instantiation.

        If model_path is a directory, the hash is computed by hashing the hashes of all model files in the
        directory. The final composite hash is always computed using BLAKE3.

        Args:
            model_path: Path to the model

        Returns:
            str: Hexdigest of the hash of the model
        """

        model_path = Path(model_path)
        if model_path.is_file():
            return self._hash_file(model_path)
        elif model_path.is_dir():
            return self._hash_dir(model_path)
        else:
            raise OSError(f"Not a valid file or directory: {model_path}")

    def _hash_dir(self, dir: Path) -> str:
        """Compute the hash for all files in a directory and return a hexdigest.

        Args:
            dir: Path to the directory

        Returns:
            str: Hexdigest of the hash of the directory
        """
        model_component_paths = self._get_file_paths(dir, self._file_filter)

        # Use ThreadPoolExecutor to hash files in parallel
        with ThreadPoolExecutor(min(((os.cpu_count() or 1) + 4), len(model_component_paths))) as executor:
            future_to_component = {executor.submit(self._hash_file, component): component for component in sorted(model_component_paths)}
            component_hashes = [future.result() for future in as_completed(future_to_component)]

        # BLAKE3 to hash the hashes
        composite_hasher = blake3()
        component_hashes.sort()
        for h in component_hashes:
            composite_hasher.update(h.encode("utf-8"))
        return composite_hasher.hexdigest()

    @staticmethod
    def _get_file_paths(model_path: Path, file_filter: Callable[[str], bool]) -> list[Path]:
        """Return a list of all model files in the directory.

        Args:
            model_path: Path to the model
            file_filter: Function that takes a file name and returns True if the file should be included in the list.

        Returns:
            List of all model files in the directory
        """

        files: list[Path] = []
        entries = [entry for entry in os.scandir(model_path.as_posix()) if not entry.name.startswith(".")]
        dirs = [entry for entry in entries if entry.is_dir()]
        file_paths = [entry.path for entry in entries if entry.is_file() and file_filter(entry.path)]
        files.extend([Path(file) for file in file_paths])
        for dir in dirs:
            files.extend(ModelHash._get_file_paths(Path(dir.path), file_filter))
        return files

    @staticmethod
    def _blake3(file_path: Path) -> str:
        """Hashes a file using BLAKE3

        Args:
            file_path: Path to the file to hash

        Returns:
            Hexdigest of the hash of the file
        """
        file_hasher = blake3(max_threads=blake3.AUTO)
        file_hasher.update_mmap(file_path)
        return file_hasher.hexdigest()

    @staticmethod
    def _get_hashlib(algorithm: ALGORITHM) -> Callable[[Path], str]:
        """Factory function that returns a function to hash a file with the given algorithm.

        Args:
            algorithm: Hashing algorithm to use

        Returns:
            A function that hashes a file using the given algorithm
        """

        def hashlib_hasher(file_path: Path) -> str:
            """Hashes a file using a hashlib algorithm."""
            hasher = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        return hashlib_hasher

    @staticmethod
    def _default_file_filter(file_path: str) -> bool:
        """A default file filter that only includes files with the following extensions: .ckpt, .safetensors, .bin, .pt, .pth

        Args:
            file_path: Path to the file

        Returns:
            True if the file matches the given extensions, otherwise False
        """
        return file_path.endswith(MODEL_FILE_EXTENSIONS)
