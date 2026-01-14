"""SDNQ State Dict Loader - loads SDNQ quantized safetensors files."""

import gc
import json
from pathlib import Path
from typing import Any, Union

import torch
from safetensors.torch import load_file

from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType


def _parse_quantization_config(config_path: Path) -> dict[str, Any]:
    """Parse quantization_config.json for SDNQ parameters."""
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_quantization_type(
    weight_dtype: torch.dtype,
    has_zero_point: bool,
    config: dict[str, Any],
) -> SDNQQuantizationType:
    """Infer quantization type from weight dtype and config."""
    # Check config first
    if "quant_type" in config:
        return SDNQQuantizationType(config["quant_type"])

    # Infer from dtype
    if weight_dtype == torch.int8:
        return SDNQQuantizationType.INT8_ASYM if has_zero_point else SDNQQuantizationType.INT8_SYM
    elif weight_dtype == torch.uint8:
        return SDNQQuantizationType.UINT8_ASYM if has_zero_point else SDNQQuantizationType.UINT8_SYM
    elif weight_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return SDNQQuantizationType.FP8_E4M3
    elif weight_dtype in (torch.float8_e5m2, torch.float8_e5m2fnuz):
        return SDNQQuantizationType.FP8_E5M2

    # Default to symmetric int8
    return SDNQQuantizationType.INT8_SYM


def _get_original_shape(
    weight: torch.Tensor,
    config: dict[str, Any],
    tensor_name: str,
) -> torch.Size:
    """Determine the original tensor shape before quantization."""
    # Check if shape is stored in config
    if "shapes" in config and tensor_name in config["shapes"]:
        return torch.Size(config["shapes"][tensor_name])

    # Quantized tensors usually keep the same shape
    return weight.shape


def sdnq_sd_loader(
    model_path: Path,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Union[SDNQTensor, torch.Tensor]]:
    """Load SDNQ quantized safetensors files.

    SDNQ stores quantized weights with associated scale, zero_point (optional),
    and SVD correction matrices (optional). This loader creates SDNQTensor
    wrappers that provide on-the-fly dequantization.

    Args:
        model_path: Path to safetensors file or directory containing model files.
        compute_dtype: Dtype for dequantized computation (default: bfloat16).

    Returns:
        State dict with SDNQTensor wrappers for quantized weights and
        regular tensors for non-quantized weights.
    """
    # Determine paths
    if model_path.is_dir():
        # Look for main model file
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {model_path}")
        model_file = safetensors_files[0]
        config_path = model_path / "quantization_config.json"
    else:
        model_file = model_path
        config_path = model_path.parent / "quantization_config.json"

    # Load quantization config if available
    quant_config = _parse_quantization_config(config_path)

    # Load safetensors file
    raw_sd = load_file(model_file)

    # Group related tensors (weight, scale, zero_point, svd_up, svd_down)
    sd: dict[str, Union[SDNQTensor, torch.Tensor]] = {}
    processed_keys: set[str] = set()

    for key in raw_sd.keys():
        if key in processed_keys:
            continue

        # Check if this is a base weight tensor
        if key.endswith(".weight"):
            base_key = key[:-7]  # Remove ".weight"

            weight = raw_sd[key]
            scale_key = f"{base_key}.scale"
            zero_point_key = f"{base_key}.zero_point"
            svd_up_key = f"{base_key}.svd_up"
            svd_down_key = f"{base_key}.svd_down"

            # Check if this is a quantized tensor (has scale)
            if scale_key in raw_sd:
                scale = raw_sd[scale_key]
                zero_point = raw_sd.get(zero_point_key)
                svd_up = raw_sd.get(svd_up_key)
                svd_down = raw_sd.get(svd_down_key)

                # Determine quantization type
                quant_type = _infer_quantization_type(
                    weight.dtype,
                    zero_point is not None,
                    quant_config,
                )

                # Determine original shape
                original_shape = _get_original_shape(weight, quant_config, key)

                # Create SDNQTensor
                sd[key] = SDNQTensor(
                    data=weight,
                    quantization_type=quant_type,
                    tensor_shape=original_shape,
                    compute_dtype=compute_dtype,
                    scale=scale,
                    zero_point=zero_point,
                    svd_up=svd_up,
                    svd_down=svd_down,
                )

                # Mark related keys as processed
                processed_keys.add(key)
                processed_keys.add(scale_key)
                if zero_point is not None:
                    processed_keys.add(zero_point_key)
                if svd_up is not None:
                    processed_keys.add(svd_up_key)
                if svd_down is not None:
                    processed_keys.add(svd_down_key)
            else:
                # Regular tensor, not quantized
                sd[key] = weight
                processed_keys.add(key)

        elif key.endswith((".scale", ".zero_point", ".svd_up", ".svd_down")):
            # Skip - these are handled with their parent weight tensor
            continue

        else:
            # Other tensors (biases, norms, etc.) - copy as-is
            sd[key] = raw_sd[key]
            processed_keys.add(key)

    gc.collect()
    return sd


def has_sdnq_tensors(state_dict: dict[str, Any]) -> bool:
    """Check if state dict contains SDNQTensor instances."""
    return any(isinstance(v, SDNQTensor) for v in state_dict.values())


def has_sdnq_keys(state_dict: dict[str, Any]) -> bool:
    """Check if state dict has SDNQ-style keys (weight + scale pairs).

    SDNQ quantized models store weights with associated scale tensors.
    This function detects this pattern to identify SDNQ models.

    Args:
        state_dict: State dict to check.

    Returns:
        True if state dict has SDNQ-style key patterns.
    """
    keys = {k for k in state_dict.keys() if isinstance(k, str)}
    for key in keys:
        if key.endswith(".weight"):
            base = key[:-7]
            if f"{base}.scale" in keys:
                return True
    return False
