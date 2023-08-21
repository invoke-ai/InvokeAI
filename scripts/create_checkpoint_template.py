#!/usr/bin/env python
"""
Read a checkpoint/safetensors file and write out a template .json file containing
its metadata for use in fast model probing.
"""

import argparse
import json

from pathlib import Path

from invokeai.backend.model_manager import(
    read_checkpoint_meta,
    ModelType,
    ModelVariantType,
    BaseModelType,
)

parser = argparse.ArgumentParser(
    description="Create a .json template from checkpoint/safetensors model",
)
parser.add_argument('checkpoint', type=Path, help="Path to the input checkpoint/safetensors file")
parser.add_argument("--template", "--out", type=Path, help="Path to the output .json file")
parser.add_argument("--base-type",
                    type=str,
                    choices=[x.value for x in BaseModelType],
                    help="Base model",
                    )
parser.add_argument("--model-type",
                    type=str,
                    choices=[x.value for x in ModelType],
                    default='main',
                    help="Type of the model",
                    )
parser.add_argument("--variant",
                    type=str,
                    choices=[x.value for x in ModelVariantType],
                    default='normal',
                    help="Base type of the model",
                    )

opt = parser.parse_args()
ckpt = read_checkpoint_meta(opt.checkpoint)
while "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

tmpl = {}

for key, tensor in ckpt.items():
    tmpl[key] = list(tensor.shape)

meta = {
    'base_type': opt.base_type,
    'model_type': opt.model_type,
    'variant': opt.variant,
    'template': tmpl
}

try:
    with open(opt.template, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"Template written out as {opt.template}")
except OSError as e:
    print(f"An exception occurred while writing template: {str(e)}")
