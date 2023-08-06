# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Fast hashing of diffusers and checkpoint-style models.

Usage:
from invokeai.backend.model_management.model_hash import FastModelHash
>>> FastModelHash.hash('/home/models/stable-diffusion-v1.5')
'a8e693a126ea5b831c96064dc569956f'
"""

import os
import hashlib
from imohash import hashfile
from pathlib import Path
from typing import Dict, Union


class FastModelHash(object):
    """FastModelHash obect provides one public class method, hash()."""

    # When traversing directories, ignore files smaller than this
    # minimum value
    MINIMUM_FILE_SIZE = 100000

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
        # we return sha256 hash of the filehash in order to be
        # consistent with length of hashes returned by _hash_dir()
        return hashlib.sha256(hashfile(model_location)).hexdigest()

    @classmethod
    def _hash_dir(cls, model_location: Union[str, Path]) -> str:
        components: Dict[str, str] = {}

        for root, dirs, files in os.walk(model_location):
            for file in files:
                # Only pay attention to the big files. The config
                # files contain things like diffusers point version
                # which change locally.
                path = Path(root) / file
                if path.stat().st_size < cls.MINIMUM_FILE_SIZE:
                    continue
                fast_hash = cls._hash_file(path)
                components.update({str(path): fast_hash})

        # hash all the model hashes together, using alphabetic file order
        sha = hashlib.sha256()
        for path, fast_hash in sorted(components.items()):
            sha.update(fast_hash.encode("utf-8"))
        return sha.hexdigest()
