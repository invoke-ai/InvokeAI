"""
Usage:
    strip_models.py <models_input_dir> <stripped_output_dir>

    Strips tensor data from model state_dicts while preserving metadata.
    Used to create lightweight models for testing model classification.

Parameters:
  <models_input_dir>     Directory containing original models.
  <stripped_output_dir>  Directory where stripped models will be saved.

Options:
  -h, --help             Show this help message and exit
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import humanize
import torch

from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.search import ModelSearch


def strip(v):
    match v:
        case torch.Tensor():
            return {"shape": v.shape, "dtype": str(v.dtype), "fakeTensor": True}
        case dict():
            return {k: strip(v) for k, v in v.items()}
        case list() | tuple():
            return [strip(x) for x in v]
        case _:
            return v


STR_TO_DTYPE = {str(dtype): dtype for dtype in torch.__dict__.values() if isinstance(dtype, torch.dtype)}


def dress(v):
    match v:
        case {"shape": shape, "dtype": dtype_str, "fakeTensor": True}:
            dtype = STR_TO_DTYPE[dtype_str]
            return torch.empty(shape, dtype=dtype)
        case dict():
            return {k: dress(v) for k, v in v.items()}
        case list() | tuple():
            return [dress(x) for x in v]
        case _:
            return v


def load_stripped_model(path: Path, *args, **kwargs):
    with open(path, "r") as f:
        contents = json.load(f)
    return dress(contents)


def create_stripped_model(original_model_path: Path, stripped_model_path: Path) -> ModelOnDisk:
    original = ModelOnDisk(original_model_path)
    if original.path.is_file():
        shutil.copy2(original.path, stripped_model_path)
    else:
        shutil.copytree(original.path, stripped_model_path, dirs_exist_ok=True)
    stripped = ModelOnDisk(stripped_model_path)
    print(f"Created clone of {original.name} at {stripped.path}")

    for component_path in stripped.component_paths():
        original_state_dict = stripped.load_state_dict(component_path)
        stripped_state_dict = strip(original_state_dict)  # type: ignore
        with open(component_path, "w") as f:
            json.dump(stripped_state_dict, f, indent=4)

    before_size = humanize.naturalsize(original.size())
    after_size = humanize.naturalsize(stripped.size())
    print(f"{original.name} before: {before_size}, after: {after_size}")

    return stripped


def parse_arguments():
    class Parser(argparse.ArgumentParser):
        def error(self, reason):
            raise ValueError(reason)

    parser = Parser()
    parser.add_argument("models_input_dir", type=Path)
    parser.add_argument("stripped_output_dir", type=Path)

    try:
        args = parser.parse_args()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    if not args.models_input_dir.exists():
        parser.error(f"Error: Input models directory '{args.models_input_dir}' does not exist.")
    if not args.models_input_dir.is_dir():
        parser.error(f"Error: '{args.input_models_dir}' is not a directory.")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    model_paths = sorted(ModelSearch().search(args.models_input_dir))

    for path in model_paths:
        stripped_path = args.stripped_output_dir / path.name
        create_stripped_model(path, stripped_path)
