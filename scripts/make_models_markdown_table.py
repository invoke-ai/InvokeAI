#!/usr/bin/env python

"""
This script is used at release time to generate a markdown table describing the
starter models. This text is then manually copied into 050_INSTALL_MODELS.md.
"""

from omegaconf import OmegaConf
from pathlib import Path


def main():
    initial_models_file = Path(__file__).parent / "../invokeai/configs/INITIAL_MODELS.yaml"
    models = OmegaConf.load(initial_models_file)
    print("|Model Name | HuggingFace Repo ID | Description | URL |")
    print("|---------- | ---------- | ----------- | --- |")
    for model in models:
        repo_id = models[model].repo_id
        url = f"https://huggingface.co/{repo_id}"
        print(f"|{model}|{repo_id}|{models[model].description}|{url} |")


if __name__ == "__main__":
    main()
