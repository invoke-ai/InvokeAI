"""Utilities for parsing model files, used mostly by legacy_probe.py"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import safetensors
import torch
from picklescan.scanner import scan_file_path

from invokeai.backend.model_manager.config import ClipVariantType
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader


def _fast_safetensors_reader(path: str) -> Dict[str, torch.Tensor]:
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


def read_checkpoint_meta(path: Union[str, Path], scan: bool = True) -> Dict[str, torch.Tensor]:
    if str(path).endswith(".safetensors"):
        try:
            path_str = path.as_posix() if isinstance(path, Path) else path
            checkpoint = _fast_safetensors_reader(path_str)
        except Exception:
            # TODO: create issue for support "meta"?
            checkpoint = safetensors.torch.load_file(path, device="cpu")
    elif str(path).endswith(".gguf"):
        # The GGUF reader used here uses numpy memmap, so these tensors are not loaded into memory during this function
        checkpoint = gguf_sd_loader(Path(path), compute_dtype=torch.float32)
    else:
        if scan:
            scan_result = scan_file_path(path)
            if scan_result.infected_files != 0 or scan_result.scan_err:
                raise Exception(f'The model file "{path}" is potentially infected by malware. Aborting import.')
        checkpoint = torch.load(path, map_location=torch.device("meta"))
    return checkpoint


def lora_token_vector_length(checkpoint: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Given a checkpoint in memory, return the lora token vector length

    :param checkpoint: The checkpoint
    """

    def _get_shape_1(key: str, tensor: torch.Tensor, checkpoint: Dict[str, torch.Tensor]) -> Optional[int]:
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


def convert_bundle_to_flux_transformer_checkpoint(
    transformer_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    original_state_dict: dict[str, torch.Tensor] = {}
    keys_to_remove: list[str] = []

    for k, v in transformer_state_dict.items():
        if not k.startswith("model.diffusion_model"):
            keys_to_remove.append(k)  # This can be removed in the future if we only want to delete transformer keys
            continue
        if k.endswith("scale"):
            # Scale math must be done at bfloat16 due to our current flux model
            # support limitations at inference time
            v = v.to(dtype=torch.bfloat16)
        new_key = k.replace("model.diffusion_model.", "")
        original_state_dict[new_key] = v
        keys_to_remove.append(k)

    # Remove processed keys from the original dictionary, leaving others in case
    # other model state dicts need to be pulled
    for k in keys_to_remove:
        del transformer_state_dict[k]

    return original_state_dict


def get_clip_variant_type(location: str) -> Optional[ClipVariantType]:
    try:
        path = Path(location)
        config_path = path / "config.json"
        if not config_path.exists():
            config_path = path / "text_encoder" / "config.json"
        if not config_path.exists():
            return ClipVariantType.L
        with open(config_path) as file:
            clip_conf = json.load(file)
            hidden_size = clip_conf.get("hidden_size", -1)
            match hidden_size:
                case 1280:
                    return ClipVariantType.G
                case 768:
                    return ClipVariantType.L
                case _:
                    return ClipVariantType.L
    except Exception:
        return ClipVariantType.L
