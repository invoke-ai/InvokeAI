# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

import hashlib
import os
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from blake3 import blake3
from tqdm import tqdm

from invokeai.app.util.misc import uuid_string

HASHING_ALGORITHMS = Literal[
    "blake3_multi",
    "blake3_single",
    "random",
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
]
MODEL_FILE_EXTENSIONS = (".ckpt", ".safetensors", ".bin", ".pt", ".pth")


class ModelHash:
    """
    Creates a hash of a model using a specified algorithm. The hash is prefixed by the algorithm used.

    Args:
        algorithm: Hashing algorithm to use. Defaults to BLAKE3.
        file_filter: A function that takes a file name and returns True if the file should be included in the hash.

    If the model is a single file, it is hashed directly using the provided algorithm.

    If the model is a directory, each model weights file in the directory is hashed using the provided algorithm.

    Only files with the following extensions are hashed: .ckpt, .safetensors, .bin, .pt, .pth

    The final hash is computed by hashing the hashes of all model files in the directory using BLAKE3, ensuring
    that directory hashes are never weaker than the file hashes.

    A convenience algorithm choice of "random" is also available, which returns a random string. This is not a hash.

    Usage:
        ```py
        # BLAKE3 hash
        ModelHash().hash("path/to/some/model.safetensors") # "blake3:ce3f0c5f3c05d119f4a5dcaf209b50d3149046a0d3a9adee9fed4c83cad6b4d0"
        # MD5
        ModelHash("md5").hash("path/to/model/dir/") # "md5:a0cd925fc063f98dbf029eee315060c3"
        ```
    """

    def __init__(
        self, algorithm: HASHING_ALGORITHMS = "blake3_single", file_filter: Optional[Callable[[str], bool]] = None
    ) -> None:
        self.algorithm: HASHING_ALGORITHMS = algorithm
        if algorithm == "blake3_multi":
            self._hash_file = self._blake3
        elif algorithm == "blake3_single":
            self._hash_file = self._blake3_single
        elif algorithm in hashlib.algorithms_available:
            self._hash_file = self._get_hashlib(algorithm)
        elif algorithm == "random":
            self._hash_file = self._random
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
        # blake3_single is a single-threaded version of blake3, prefix should still be "blake3:"
        prefix = self._get_prefix(self.algorithm)
        if model_path.is_file():
            hash_ = None
            # To give a similar user experience for single files and directories, we use a progress bar even for single files
            pbar = tqdm([model_path], desc=f"Hashing {model_path.name}", unit="file")
            for component in pbar:
                pbar.set_description(f"Hashing {component.name}")
                hash_ = prefix + self._hash_file(model_path)
            assert hash_ is not None
            return hash_
        elif model_path.is_dir():
            return prefix + self._hash_dir(model_path)
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

        component_hashes: list[str] = []
        pbar = tqdm(sorted(model_component_paths), desc=f"Hashing {dir.name}", unit="file")
        for component in pbar:
            pbar.set_description(f"Hashing {component.name}")
            component_hashes.append(self._hash_file(component))

        # BLAKE3 is cryptographically secure. We may as well fall back on a secure algorithm
        # for the composite hash
        composite_hasher = blake3()
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
        for root, _dirs, _files in os.walk(model_path):
            for file in _files:
                if file_filter(file):
                    files.append(Path(root, file))
        return files

    @staticmethod
    def _blake3(file_path: Path) -> str:
        """Hashes a file using BLAKE3, using parallelized and memory-mapped I/O to avoid reading the entire file into memory.

        Args:
            file_path: Path to the file to hash

        Returns:
            Hexdigest of the hash of the file
        """
        file_hasher = blake3(max_threads=blake3.AUTO)
        file_hasher.update_mmap(file_path)
        return file_hasher.hexdigest()

    @staticmethod
    def _blake3_single(file_path: Path) -> str:
        """Hashes a file using BLAKE3, without parallelism. Suitable for spinning hard drives.

        Args:
            file_path: Path to the file to hash

        Returns:
            Hexdigest of the hash of the file
        """
        file_hasher = blake3()
        file_hasher.update_mmap(file_path)
        return file_hasher.hexdigest()

    @staticmethod
    def _get_hashlib(algorithm: HASHING_ALGORITHMS) -> Callable[[Path], str]:
        """Factory function that returns a function to hash a file with the given algorithm.

        Args:
            algorithm: Hashing algorithm to use

        Returns:
            A function that hashes a file using the given algorithm
        """

        def hashlib_hasher(file_path: Path) -> str:
            """Hashes a file using a hashlib algorithm. Uses `memoryview` to avoid reading the entire file into memory."""
            hasher = hashlib.new(algorithm)
            buffer = bytearray(128 * 1024)
            mv = memoryview(buffer)
            with open(file_path, "rb", buffering=0) as f:
                while n := f.readinto(mv):
                    hasher.update(mv[:n])
            return hasher.hexdigest()

        return hashlib_hasher

    @staticmethod
    def _random(_file_path: Path) -> str:
        """Returns a random string. This is not a hash.

        The string is a UUID, hashed with BLAKE3 to ensure that it is unique."""
        return blake3(uuid_string().encode()).hexdigest()

    @staticmethod
    def _default_file_filter(file_path: str) -> bool:
        """A default file filter that only includes files with the following extensions: .ckpt, .safetensors, .bin, .pt, .pth

        Args:
            file_path: Path to the file

        Returns:
            True if the file matches the given extensions, otherwise False
        """
        return file_path.endswith(MODEL_FILE_EXTENSIONS)

    @staticmethod
    def _get_prefix(algorithm: HASHING_ALGORITHMS) -> str:
        """Return the prefix for the given algorithm, e.g. \"blake3:\" or \"md5:\"."""
        # blake3_single is a single-threaded version of blake3, prefix should still be "blake3:"
        return "blake3:" if algorithm == "blake3_single" or algorithm == "blake3_multi" else f"{algorithm}:"
