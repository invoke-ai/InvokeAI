#!/bin/env python

"""Little command-line utility for probing a model on disk."""

import argparse
from pathlib import Path

from invokeai.backend.model_manager import ModelProbe, InvalidModelConfigException



parser = argparse.ArgumentParser(description="Probe model type")
parser.add_argument(
    "model_path",
    type=Path,
    nargs="+",
)
args = parser.parse_args()

for path in args.model_path:
    try:
        info = ModelProbe.probe(path)
        print(f"{path}:{info.model_dump_json(indent=4)}")
    except InvalidModelConfigException as exc:
        print(exc)
