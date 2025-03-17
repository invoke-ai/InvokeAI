#!/bin/env python

"""Little command-line utility for probing a model on disk."""

import argparse
from pathlib import Path
from typing import get_args

from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS
from invokeai.backend.model_manager import InvalidModelConfigException, ModelConfigBase, ModelProbe

algos = ", ".join(set(get_args(HASHING_ALGORITHMS)))

parser = argparse.ArgumentParser(description="Probe model type")
parser.add_argument(
    "model_path",
    type=Path,
    nargs="+",
)
parser.add_argument(
    "--hash_algo",
    type=str,
    default="blake3_single",
    help=f"Hashing algorithm to use (default: blake3_single), one of: {algos}",
)
args = parser.parse_args()


def classify_with_fallback(path: Path, hash_algo: HASHING_ALGORITHMS):
    try:
        return ModelConfigBase.classify(path, hash_algo)
    except InvalidModelConfigException:
        return ModelProbe.probe(path, hash_algo=hash_algo)


for path in args.model_path:
    try:
        config = classify_with_fallback(path, args.hash_algo)
        print(f"{path}:{config.model_dump_json(indent=4)}")
    except InvalidModelConfigException as e:
        print(e)
