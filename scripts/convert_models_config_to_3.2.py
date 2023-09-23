# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

"""
convert_models_config_to_3.2.py.

This script converts a pre-3.2 models.yaml file into the 3.2 format.
The main difference is that each model is identified by a unique hash,
rather than the concatenation of base, type and name used previously.

In addition, there are more metadata fields attached to each model.
These will mostly be empty after conversion, but will be populated
when new models are downloaded from HuggingFace or Civitae.
"""
import argparse
from pathlib import Path

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager.storage import migrate_models_store


def main():
    parser = argparse.ArgumentParser(description="Convert a pre-3.2 models.yaml into the 3.2 version.")
    parser.add_argument("--root", type=Path, help="Alternate root directory containing the models.yaml to convert")
    parser.add_argument(
        "--outfile",
        type=Path,
        default=Path("./models-3.2.yaml"),
        help="File to write to. A file with suffix '.yaml' will use the YAML format. A file with an extension of '.db' will be treated as a SQLite3 database.",
    )
    args = parser.parse_args()
    config_args = ["--root", args.root.as_posix()] if args.root else []

    config = InvokeAIAppConfig.get_config()
    config.parse_args(config_args)
    migrate_models_store(config)


if __name__ == "__main__":
    main()
