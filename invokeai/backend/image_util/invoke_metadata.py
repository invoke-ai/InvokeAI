# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

"""Very simple functions to fetch and print metadata from InvokeAI-generated images."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def get_invokeai_metadata(image_path: Path) -> Dict[str, Any]:
    """
    Retrieve "invokeai_metadata" field from png image.

    :param image_path: Path to the image to read metadata from.
    May raise:
      OSError -- image path not found
      KeyError -- image doesn't contain the metadata field
    """
    image: Image = Image.open(image_path)
    return json.loads(image.text["invokeai_metadata"])


def print_invokeai_metadata(image_path: Path):
    """Pretty-print the metadata."""
    try:
        metadata = get_invokeai_metadata(image_path)
        print(f"{image_path}:\n{json.dumps(metadata, sort_keys=True, indent=4)}")
    except OSError:
        print(f"{image_path}:\nNo file found.")
    except KeyError:
        print(f"{image_path}:\nNo metadata found.")
    print()


def main():
    """Run the command-line utility."""
    image_paths = sys.argv[1:]
    if not image_paths:
        print(f"Usage: {Path(sys.argv[0]).name} image1 image2 image3 ...")
        print("\nPretty-print InvokeAI image metadata from the listed png files.")
        sys.exit(-1)
    for img in image_paths:
        print_invokeai_metadata(img)
