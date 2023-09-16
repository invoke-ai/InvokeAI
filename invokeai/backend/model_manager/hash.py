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
            return cls._hash_file(model_location)
        elif model_location.is_dir():
            return cls._hash_dir(model_location)
        else:
            # avoid circular import
            from .models import InvalidModelException

            raise InvalidModelException(f"Not a valid file or directory: {model_location}")

    @classmethod
    def _hash_file(cls, model_location: Union[str, Path]) -> str:
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

        for root, dirs, files in os.walk(model_location):
            for file in files:
                # Ignore the config files, which change locally,
                # and just look at the bin files.
                if file in ["config.json", "model_index.json"]:
                    continue
                path = Path(root) / file
                fast_hash = cls._hash_file(path)
                components.update({str(path): fast_hash})

        # hash all the model hashes together, using alphabetic file order
        md5 = hashlib.md5()
        for path, fast_hash in sorted(components.items()):
            md5.update(fast_hash.encode("utf-8"))
        return md5.hexdigest()
