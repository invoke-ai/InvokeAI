"""Utilities for parsing model files, used mostly by probe.py"""

import json
import torch
from typing import Union
from pathlib import Path
from picklescan.scanner import scan_file_path

def _fast_safetensors_reader(path: str):
    checkpoint = {}
    device = torch.device("meta")
    with open(path, "rb") as f:
        definition_len = int.from_bytes(f.read(8), "little")
        definition_json = f.read(definition_len)
        definition = json.loads(definition_json)

        if "__metadata__" in definition and definition["__metadata__"].get("format", "pt") not in {
            "pt",
            "torch",
            "pytorch",
        }:
            raise Exception("Supported only pytorch safetensors files")
        definition.pop("__metadata__", None)

        for key, info in definition.items():
            dtype = {
                "I8": torch.int8,
                "I16": torch.int16,
                "I32": torch.int32,
                "I64": torch.int64,
                "F16": torch.float16,
                "F32": torch.float32,
                "F64": torch.float64,
            }[info["dtype"]]

            checkpoint[key] = torch.empty(info["shape"], dtype=dtype, device=device)

    return checkpoint

def read_checkpoint_meta(path: Union[str, Path], scan: bool = False):
    if str(path).endswith(".safetensors"):
        try:
            checkpoint = _fast_safetensors_reader(path)
        except Exception:
            # TODO: create issue for support "meta"?
            checkpoint = safetensors.torch.load_file(path, device="cpu")
    else:
        if scan:
            scan_result = scan_file_path(path)
            if scan_result.infected_files != 0:
                raise Exception(f'The model file "{path}" is potentially infected by malware. Aborting import.')
        checkpoint = torch.load(path, map_location=torch.device("meta"))
    return checkpoint

def lora_token_vector_length(checkpoint: dict) -> int:
    """
    Given a checkpoint in memory, return the lora token vector length

    :param checkpoint: The checkpoint
    """

    def _get_shape_1(key: str, tensor, checkpoint) -> int:
        lora_token_vector_length = None

        if "." not in key:
            return lora_token_vector_length  # wrong key format
        model_key, lora_key = key.split(".", 1)

        # check lora/locon
        if lora_key == "lora_down.weight":
            lora_token_vector_length = tensor.shape[1]

        # check loha (don't worry about hada_t1/hada_t2 as it used only in 4d shapes)
        elif lora_key in ["hada_w1_b", "hada_w2_b"]:
            lora_token_vector_length = tensor.shape[1]

        # check lokr (don't worry about lokr_t2 as it used only in 4d shapes)
        elif "lokr_" in lora_key:
            if model_key + ".lokr_w1" in checkpoint:
                _lokr_w1 = checkpoint[model_key + ".lokr_w1"]
            elif model_key + "lokr_w1_b" in checkpoint:
                _lokr_w1 = checkpoint[model_key + ".lokr_w1_b"]
            else:
                return lora_token_vector_length  # unknown format

            if model_key + ".lokr_w2" in checkpoint:
                _lokr_w2 = checkpoint[model_key + ".lokr_w2"]
            elif model_key + "lokr_w2_b" in checkpoint:
                _lokr_w2 = checkpoint[model_key + ".lokr_w2_b"]
            else:
                return lora_token_vector_length  # unknown format

            lora_token_vector_length = _lokr_w1.shape[1] * _lokr_w2.shape[1]

        elif lora_key == "diff":
            lora_token_vector_length = tensor.shape[1]

        # ia3 can be detected only by shape[0] in text encoder
        elif lora_key == "weight" and "lora_unet_" not in model_key:
            lora_token_vector_length = tensor.shape[0]

        return lora_token_vector_length

    lora_token_vector_length = None
    lora_te1_length = None
    lora_te2_length = None
    for key, tensor in checkpoint.items():
        if key.startswith("lora_unet_") and ("_attn2_to_k." in key or "_attn2_to_v." in key):
            lora_token_vector_length = _get_shape_1(key, tensor, checkpoint)
        elif key.startswith("lora_unet_") and (
            "time_emb_proj.lora_down" in key
        ):  # recognizes format at https://civitai.com/models/224641
            lora_token_vector_length = _get_shape_1(key, tensor, checkpoint)
        elif key.startswith("lora_te") and "_self_attn_" in key:
            tmp_length = _get_shape_1(key, tensor, checkpoint)
            if key.startswith("lora_te_"):
                lora_token_vector_length = tmp_length
            elif key.startswith("lora_te1_"):
                lora_te1_length = tmp_length
            elif key.startswith("lora_te2_"):
                lora_te2_length = tmp_length

        if lora_te1_length is not None and lora_te2_length is not None:
            lora_token_vector_length = lora_te1_length + lora_te2_length

        if lora_token_vector_length is not None:
            break

    return lora_token_vector_length
