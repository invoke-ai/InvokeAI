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
from typing import Literal, Union

from blake3 import blake3

MODEL_FILE_EXTENSIONS = (".ckpt", ".safetensors", ".bin", ".pt", ".pth")

ALGORITHMS = Literal[
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
    """ModelHash provides one public class method, hash()."""

    @classmethod
    def hash(cls, model_location: Union[str, Path], algorithm: ALGORITHMS = "blake3") -> str:
        """
        Return hexdigest string for model located at model_location.

        If model_location is a directory, the hash is computed by hashing the hashes of all model files in the
        directory. The final composite hash is always computed using BLAKE3.

        :param model_location: Path to the model
        :param algorithm: Hashing algorithm to use
        """
        model_location = Path(model_location)
        if model_location.is_file():
            return cls._hash_file(model_location, algorithm)
        elif model_location.is_dir():
            return cls._hash_dir(model_location, algorithm)
        else:
            raise OSError(f"Not a valid file or directory: {model_location}")

    @classmethod
    def _hash_file(cls, model_location: Union[str, Path], algorithm: ALGORITHMS) -> str:
        """
        Compute the hash for a single file and return its hexdigest.

        :param model_location: Path to the model file
        :param algorithm: Hashing algorithm to use
        """

        if algorithm == "blake3":
            return cls._blake3(model_location)
        elif algorithm == "sha1_fast":
            return cls._sha1_fast(model_location)
        elif algorithm in hashlib.algorithms_available:
            return cls._hashlib(model_location, algorithm)
        else:
            raise ValueError(f"Algorithm {algorithm} not available")

    @classmethod
    def _hash_dir(cls, model_location: Union[str, Path], algorithm: ALGORITHMS) -> str:
        """
        Compute the hash for all files in a directory and return a hexdigest.

        :param model_location: Path to the model directory
        :param algorithm: Hashing algorithm to use
        """
        components: list[str] = []

        for root, _dirs, files in os.walk(model_location):
            for file in files:
                # only tally tensor files because diffusers config files change slightly
                # depending on how the model was downloaded/converted.
                if file.endswith(MODEL_FILE_EXTENSIONS):
                    components.append((Path(root, file).as_posix()))

        component_hashes: list[str] = []
        for component in sorted(components):
            component_hashes.append(cls._hash_file(component, algorithm))

        # BLAKE3 is cryptographically secure. We may as well fall back on a secure algorithm
        # for the composite hash
        composite_hasher = blake3()
        for h in components:
            composite_hasher.update(h.encode("utf-8"))
        return composite_hasher.hexdigest()

    @staticmethod
    def _blake3(file_path: Union[str, Path]) -> str:
        """Hashes a file using BLAKE3"""
        file_hasher = blake3(max_threads=blake3.AUTO)
        file_hasher.update_mmap(file_path)
        return file_hasher.hexdigest()

    @staticmethod
    def _sha1_fast(file_path: Union[str, Path]) -> str:
        """Hashes a file using SHA1, but with a block size of 2**16. The result is not a standard SHA1 hash due to the
        # padding introduced by the block size. The algorithm is, however, very fast."""
        BLOCK_SIZE = 2**16
        file_hash = hashlib.sha1()
        with open(file_path, "rb") as f:
            data = f.read(BLOCK_SIZE)
            file_hash.update(data)
        return file_hash.hexdigest()

    @staticmethod
    def _hashlib(file_path: Union[str, Path], algorithm: ALGORITHMS) -> str:
        """Hashes a file using a hashlib algorithm"""
        file_hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            file_hasher.update(f.read())
        return file_hasher.hexdigest()
