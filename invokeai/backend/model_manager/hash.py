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
from typing import Dict, Union

from imohash import hashfile


class FastModelHash(object):
    """FastModelHash obect provides one public class method, hash()."""

    @classmethod
    def hash(cls, model_location: Union[str, Path]) -> str:
        """
        Return hexdigest string for model located at model_location.

        :param model_location: Path to the model
        """
        model_location = Path(model_location)
        if model_location.is_file():
            return cls._hash_file_sha1(model_location)
        elif model_location.is_dir():
            return cls._hash_dir(model_location)
        else:
            raise OSError(f"Not a valid file or directory: {model_location}")

    @classmethod
    def _hash_file_sha1(cls, model_location: Union[str, Path]) -> str:
        """
        Compute full sha1 hash over a single file and return its hexdigest.

        :param model_location: Path to the model file
        """
        BLOCK_SIZE = 65536
        file_hash = hashlib.sha1()
        with open(model_location, "rb") as f:
            data = f.read(BLOCK_SIZE)
            file_hash.update(data)
        return file_hash.hexdigest()

    @classmethod
    def _hash_file_fast(cls, model_location: Union[str, Path]) -> str:
        """
        Fasthash a single file and return its hexdigest.

        :param model_location: Path to the model file
        """
        # we return md5 hash of the filehash to make it shorter
        # cryptographic security not needed here
        return hashlib.md5(hashfile(model_location)).hexdigest()

    @classmethod
    def _hash_dir(cls, model_location: Union[str, Path]) -> str:
        components: Dict[str, str] = {}

        for root, _dirs, files in os.walk(model_location):
            for file in files:
                # only tally tensor files because diffusers config files change slightly
                # depending on how the model was downloaded/converted.
                if not file.endswith((".ckpt", ".safetensors", ".bin", ".pt", ".pth")):
                    continue
                path = (Path(root) / file).as_posix()
                fast_hash = cls._hash_file_fast(path)
                components.update({path: fast_hash})

        # hash all the model hashes together, using alphabetic file order
        md5 = hashlib.md5()
        for _path, fast_hash in sorted(components.items()):
            md5.update(fast_hash.encode("utf-8"))
        return md5.hexdigest()
