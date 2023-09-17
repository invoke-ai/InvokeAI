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

from omegaconf import OmegaConf

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager import DuplicateModelException, InvalidModelException, ModelInstall
from invokeai.backend.model_manager.storage import get_config_store


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
    old_yaml_file = OmegaConf.load(config.model_conf_path)

    store = get_config_store(args.outfile)
    installer = ModelInstall(store=store)

    print(f"Writing 3.2 models configuration into {args.outfile}.")

    for model_key, stanza in old_yaml_file.items():
        if model_key == "__metadata__":
            assert (
                stanza["version"] == "3.0.0"
            ), f"This script works on version 3.0.0 yaml files, but your configuration points to a {stanza['version']} version"
            continue

        try:
            path = config.models_path / stanza["path"]
            new_key = installer.register_path(path)
            model_info = store.get_model(new_key)
            if vae := stanza.get("vae"):
                model_info.vae = (config.models_path / vae).as_posix()
            if model_config := stanza.get("config"):
                model_info.config = (config.root_path / model_config).as_posix()
            model_info.description = stanza.get("description")
            store.update_model(new_key, model_info)

            print(f"{model_key} => {new_key}")
        except (DuplicateModelException, InvalidModelException) as e:
            print(str(e))


if __name__ == "__main__":
    main()
