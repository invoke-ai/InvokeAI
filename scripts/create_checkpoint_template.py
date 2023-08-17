#!/usr/bin/env python
"""
Read a checkpoint/safetensors file and write out a template .json file containing
its metadata for use in fast model probing.
"""

import argparse
import json

from pathlib import Path

from invokeai.backend.model_management.models.base import read_checkpoint_meta

parser = argparse.ArgumentParser(description="Create a .json template from checkpoint/safetensors model")
parser.add_argument("--checkpoint", "--in", type=Path, help="Path to the input checkpoint/safetensors file")
parser.add_argument("--template", "--out", type=Path, help="Path to the output .json file")

opt = parser.parse_args()
ckpt = read_checkpoint_meta(opt.checkpoint)
while "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

tmpl = {}

for key, tensor in ckpt.items():
    tmpl[key] = list(tensor.shape)

try:
    with open(opt.template, "w") as f:
        json.dump(tmpl, f)
    print(f"Template written out as {opt.template}")
except Exception as e:
    print(f"An exception occurred while writing template: {str(e)}")
