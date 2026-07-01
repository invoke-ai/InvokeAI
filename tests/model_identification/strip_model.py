"""
Usage:
    strip_model.py <model_path> <output_dir>

    Strips tensor data from model state_dict while preserving metadata.
    Used to create lightweight models for testing model classification.

Parameters:
  <model_path>     The path to the model to be stripped.
  <output_dir>     Directory where stripped models will be saved (e.g. tests/test_model_probe/stripped_models)

Options:
  -h, --help       Show this help message and exit
"""

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import humanize

from invokeai.app.util.misc import uuid_string
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from tests.model_identification.stripped_model_on_disk import StrippedModelOnDisk

TEST_METADATA_FILENAME = "__test_metadata__.json"


TEST_METADATA: dict[str, Any] = {
    "source": "",
    "file_name": "",
    "override_fields": {},
    "expected_config_attrs": {},
    "notes": "",
}


def create_stripped_model(model_path: Path, output_dir: Path):
    """Creates a stripped version of the model at model_path in output_dir. A test metadata file is also created."""

    original_mod = ModelOnDisk(model_path)

    # The stripped model will be stored in a new directory named with a UUID. This mirrors the application's
    # normalized model storage file structure.
    uuid = uuid_string()
    stripped_model_dir = output_dir / uuid
    stripped_model_dir.mkdir(parents=True, exist_ok=True)

    test_metadata_content = deepcopy(TEST_METADATA)

    if original_mod.path.is_file():
        shutil.copy2(original_mod.path, stripped_model_dir / original_mod.path.name)
        test_metadata_content["file_name"] = original_mod.path.name
    else:
        shutil.copytree(original_mod.path, stripped_model_dir, dirs_exist_ok=True)

    stripped_mod = ModelOnDisk(stripped_model_dir)

    print(f"Created clone of {original_mod.name} at {stripped_mod.path}")

    for component_path in stripped_mod.weight_files():
        original_state_dict = stripped_mod.load_state_dict(component_path)

        stripped_state_dict = StrippedModelOnDisk.strip(original_state_dict)
        metadata = stripped_mod.metadata()
        contents = {**stripped_state_dict, StrippedModelOnDisk.METADATA_KEY: metadata}
        component_path.write_text(json.dumps(contents, indent=2))

    test_metadata_path = stripped_model_dir / TEST_METADATA_FILENAME
    test_metadata_path.write_text(json.dumps(test_metadata_content, indent=2))

    before_size = humanize.naturalsize(original_mod.size())
    after_size = humanize.naturalsize(stripped_mod.size())

    print(f"{original_mod.name} before: {before_size}, after: {after_size}")

    return stripped_mod


def parse_arguments():
    class Parser(argparse.ArgumentParser):
        def error(self, message: str):
            raise ValueError(message)

    parser = Parser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_dir", type=Path)

    try:
        args = parser.parse_args()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    if not args.model_path.exists():
        parser.error(f"Error: Input model path '{args.model_path}' does not exist.")

    return args


if __name__ == "__main__":
    args = parse_arguments()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)

    create_stripped_model(model_path, output_dir)
