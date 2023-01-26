#!/usr/bin/env python

import argparse
import os
import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from ldm.invoke.globals import (Globals, global_cache_dir, global_config_file,
                                global_set_root)
from ldm.invoke.model_manager import ModelManager

parser = argparse.ArgumentParser(description="InvokeAI textual inversion training")
parser.add_argument(
    "--root_dir",
    type=Path,
    default=Globals.root,
    help="Path to the invokeai runtime directory",
)
parser.add_argument(
    "--models",
    required=True,
    type=str,
    nargs="+",
    help="Two to three model names to be merged",
)
parser.add_argument(
    "--merged_model_name",
    "--destination",
    dest="merged_model_name",
    type=str,
    help="Name of the output model. If not specified, will be the concatenation of the input model names.",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="The interpolation parameter, ranging from 0 to 1. It affects the ratio in which the checkpoints are merged. Higher values give more weight to the 2d and 3d models",
)
parser.add_argument(
    "--interpolation",
    dest="interp",
    type=str,
    choices=["weighted_sum", "sigmoid", "inv_sigmoid", "add_difference"],
    default="weighted_sum",
    help='Interpolation method to use. If three models are present, only "add_difference" will work.',
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Try to merge models even if they are incompatible with each other",
)
parser.add_argument(
    "--clobber",
    "--overwrite",
    dest='clobber',
    action="store_true",
    help="Overwrite the merged model if --merged_model_name already exists",
)

args = parser.parse_args()
global_set_root(args.root_dir)

assert args.alpha >= 0 and args.alpha <= 1.0, "alpha must be between 0 and 1"
assert len(args.models) >= 1 and len(args.models) <= 3, "provide 2 or 3 models to merge"

if not args.merged_model_name:
    args.merged_model_name = "+".join(args.models)
    print(
        f'>> No --merged_model_name provided. Defaulting to "{args.merged_model_name}"'
    )

model_manager = ModelManager(OmegaConf.load(global_config_file()))
assert (args.clobber or args.merged_model_name not in model_manager.model_names()), f'A model named "{args.merged_model_name}" already exists. Use --clobber to overwrite.'

# It seems that the merge pipeline is not honoring cache_dir, so we set the
# HF_HOME environment variable here *before* we load diffusers.
cache_dir = str(global_cache_dir("diffusers"))
os.environ["HF_HOME"] = cache_dir
from ldm.invoke.merge_diffusers import merge_diffusion_models

try:
    merge_diffusion_models(**vars(args))
    print(f'>> Models merged into new model: "{args.merged_model_name}".')
except Exception as e:
    print(f"** An error occurred while merging the pipelines: {str(e)}")
    print("** DETAILS:")
    print(traceback.format_exc())
    sys.exit(-1)
