# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Fast hashing of diffusers and checkpoint-style models.

Usage:
from invokeai.backend.model_managre.model_hash import FastModelHash
>>> FastModelHash.hash('/home/models/stable-diffusion-v1.5')
'a8e693a126ea5b831c96064dc569956f'
"""
import cProfile
import os
import pstats
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

from blake3 import blake3
from tqdm import tqdm


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
            raise OSError(f"Not a valid file or directory: {model_location}")

    @classmethod
    def _hash_file(cls, model_location: Union[str, Path]) -> str:
        """
        Compute full BLAKE3 hash over a single file and return its hexdigest.

        :param model_location: Path to the model file
        """
        file_hasher = blake3(max_threads=blake3.AUTO)
        file_hasher.update_mmap(model_location)
        return file_hasher.hexdigest()

    @classmethod
    def _hash_dir(cls, model_location: Union[str, Path]) -> str:
        """
        Compute full BLAKE3 hash over all files in a directory and return its hexdigest.

        :param model_location: Path to the model directory
        """
        components: list[str] = []

        for root, _dirs, files in os.walk(model_location):
            for file in files:
                # only tally tensor files because diffusers config files change slightly
                # depending on how the model was downloaded/converted.
                if file.endswith((".ckpt", ".safetensors", ".bin", ".pt", ".pth")):
                    components.append((Path(root, file).resolve().as_posix()))

        component_hashes: list[str] = []

        for component in tqdm(sorted(components), desc=f"Hashing model components for {model_location}"):
            file_hasher = blake3(max_threads=blake3.AUTO)
            file_hasher.update_mmap(component)
            component_hashes.append(file_hasher.hexdigest())

        return blake3(b"".join([bytes.fromhex(h) for h in component_hashes])).hexdigest()


if __name__ == "__main__":
    with TemporaryDirectory() as tempdir:
        profile_path = Path(tempdir, "profile_results.pstats").as_posix()
        profiler = cProfile.Profile()
        profiler.enable()
        t = threading.Thread(
            target=FastModelHash.hash, args=("/media/rhino/invokeai/models/sd-1/main/stable-diffusion-v1-5-inpainting",)
        )
        t.start()
        t.join()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(profile_path)

        os.system(f"snakeviz {profile_path}")
