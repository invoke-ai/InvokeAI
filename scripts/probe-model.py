#!/bin/env python

import argparse
from pathlib import Path
from invokeai.backend.model_management.model_probe import ModelProbe

parser = argparse.ArgumentParser(description="Probe model type")
parser.add_argument(
    "model_path",
    type=Path,
    nargs="+",
)
args = parser.parse_args()

for path in args.model_path:
    info = ModelProbe().probe(path)
    print(f"{path}: {info}")
