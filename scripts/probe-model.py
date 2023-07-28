#!/bin/env python

import argparse
import sys
from pathlib import Path
from invokeai.backend.model_management.model_probe import ModelProbe

parser = argparse.ArgumentParser(description="Probe model type")
parser.add_argument(
    "model_path",
    type=Path,
)
args = parser.parse_args()

info = ModelProbe().probe(args.model_path)
print(info)
