#!/usr/bin/env python

import re
from pathlib import Path
import torch
from safetensors.torch import load_file
import argparse
from diffusers import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import create_unet_diffusers_config
from omegaconf import OmegaConf
import requests


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Convert kohya lora to diffusers")
    parser.add_argument(
        "--lora_file",
        type=str,
        default=None,
        required=True,
        help="Lora file to convert",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/lora",
        help="The output directory where converted lora will be saved",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def replace_key_blocks(match_obj):
    k = match_obj.groups()

    return f"{k[0]}.{k[1]}"


def replace_key_out(match_obj):
    return f"to_out"


def replace_key_main(match_obj):
    k = match_obj.groups()
    block = re.sub(r"(.+)_(\d+)", replace_key_blocks, k[0])
    out = re.sub(r"to_out_(\d+)", replace_key_out, k[4])

    return f"{block}.attentions.{k[1]}.transformer_blocks.{k[2]}.attn{k[3]}.processor.{out}_lora.{k[5]}"


def main(args):
    response = requests.get(
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    )
    original_config = OmegaConf.create(response.text)

    new_dict = dict()
    lora_file = Path(args.lora_file)

    if lora_file.suffix == '.safetensors':
        checkpoint = load_file(args.lora_file)
    else:
        checkpoint = torch.load(args.lora_file)

    for idx, key in enumerate(checkpoint):
        check = re.compile(r"lora_unet_(.+)_attentions_(\d+)_transformer_blocks_(\d+)_attn(\d+)_(.+).lora_(.+)")
        if check.match(key):
            new_key = check.sub(replace_key_main, key)
            new_dict[new_key] = checkpoint[key]

    unet_config = create_unet_diffusers_config(original_config, image_size=512)
    unet = UNet2DConditionModel(**unet_config)
    unet.load_attn_procs(new_dict)

    output_dir = Path(args.output_dir, lora_file.name.split('.')[0])
    unet.save_attn_procs(output_dir.absolute().as_posix())


if __name__ == "__main__":
    args = parse_args()
    main(args)
