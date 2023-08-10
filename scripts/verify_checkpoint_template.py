#!/usr/bin/env python
"""
Read a checkpoint/safetensors file and compare it to a template .json.
Returns True if their metadata match.
"""

import sys
import argparse
import json

from pathlib import Path

from invokeai.backend.model_management.models.base import read_checkpoint_meta

parser = argparse.ArgumentParser(description="Compare a checkpoint/safetensors file to a JSON metadata template.")
parser.add_argument("--checkpoint", "--in", type=Path, help="Path to the input checkpoint/safetensors file")
parser.add_argument("--template", "--out", type=Path, help="Path to the template .json file to match against")

opt = parser.parse_args()
ckpt = read_checkpoint_meta(opt.checkpoint)
while "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

checkpoint_metadata = {}

for key, tensor in ckpt.items():
    checkpoint_metadata[key] = list(tensor.shape)

with open(opt.template, "r") as f:
    template = json.load(f)

if checkpoint_metadata == template:
    print("True")
    sys.exit(0)
else:
    print("False")
    sys.exit(-1)
