"""SDNQ State Dict Loader - loads SDNQ quantized safetensors files."""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Union

import torch
from safetensors.torch import load_file

from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType

logger = logging.getLogger(__name__)


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
    weight: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> SDNQQuantizationType:
    """Infer quantization type from weight dtype and config."""
    # Check config for weights_dtype (SDNQ style)
    weights_dtype = config.get("weights_dtype", "")
    if weights_dtype == "uint4":
        return SDNQQuantizationType.UINT4_ASYM

    # Check config for quant_type
    if "quant_type" in config:
        return SDNQQuantizationType(config["quant_type"])

    # Try to detect uint4 from shapes:
    # uint4 is packed as 2 values per uint8, so if scale suggests a shape that's
    # 2x the weight's last dimension, it's likely uint4
    if weight is not None and scale is not None and weight_dtype == torch.uint8:
        if len(weight.shape) == 2 and len(scale.shape) >= 2:
            # For per-group quantization: scale shape [out, num_groups, 1] or [out, num_groups]
            # weight shape [out, in/2] for uint4
            # Check if scale's out_features matches weight's out_features
            # and if num_groups * group_size would be 2x weight's in_features
            out_features = weight.shape[0]
            packed_in = weight.shape[1]
            scale_out = scale.shape[0]
            num_groups = scale.shape[1]

            if scale_out == out_features and num_groups > 0:
                # If we can infer that unpacked shape would be 2x packed, it's uint4
                # For typical group sizes (32, 64, 128), num_groups * group_size = in_features
                # If packed_in * 2 can be divided by num_groups, it's likely uint4
                unpacked_in = packed_in * 2
                if unpacked_in % num_groups == 0:
                    inferred_group_size = unpacked_in // num_groups
                    if inferred_group_size in (32, 64, 128, 256):
                        return SDNQQuantizationType.UINT4_ASYM

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
    quant_type: SDNQQuantizationType,
    scale: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Size:
    """Determine the original tensor shape before quantization."""
    # Check if shape is stored in config
    if "shapes" in config and tensor_name in config["shapes"]:
        return torch.Size(config["shapes"][tensor_name])

    # For uint4 packed weights with per-group quantization
    if quant_type == SDNQQuantizationType.UINT4_ASYM:
        # For 1D flattened weights, infer shape from scale tensor
        # Scale can have shape [out_features, num_groups, 1] or [out_features, num_groups]
        if scale is not None and len(scale.shape) >= 2:
            out_features = scale.shape[0]
            num_groups = scale.shape[1]
            in_features = num_groups * group_size
            if in_features > 0:
                return torch.Size([out_features, in_features])

        # For 2D packed weights, multiply last dim by 2
        if len(weight.shape) == 2:
            return torch.Size([weight.shape[0], weight.shape[1] * 2])

        # For 1D, infer from packed size and scale's out_features
        if len(weight.shape) == 1 and scale is not None:
            # uint4 packs 2 values per byte
            total_elements = weight.numel() * 2
            out_features = scale.shape[0]
            if out_features > 0 and total_elements % out_features == 0:
                in_features = total_elements // out_features
                return torch.Size([out_features, in_features])

        # Final fallback for 1D
        if len(weight.shape) == 1:
            total_elements = weight.numel() * 2
            return torch.Size([total_elements])

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

    # Get group_size from config (default: 128 for SDNQ)
    # Note: group_size=0 in config means per-tensor quantization or it needs to be inferred
    config_group_size = quant_config.get("group_size", 128)

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
                    weight=weight,
                    scale=scale,
                )

                # Determine group_size for this tensor
                # If config has group_size=0, infer from scale tensor shape
                if config_group_size > 0:
                    tensor_group_size = config_group_size
                elif len(scale.shape) >= 2 and scale.shape[1] > 0:
                    # Scale shape is [out_features, num_groups, ...] or [out_features, num_groups]
                    # We'll compute group_size after determining original_shape
                    tensor_group_size = None  # Will be computed below
                else:
                    tensor_group_size = 128  # Default fallback

                # Determine original shape (need quant_type and scale to handle uint4 packing)
                original_shape = _get_original_shape(
                    weight, quant_config, key, quant_type, scale=scale, group_size=tensor_group_size or 128
                )

                # Compute group_size from original_shape and scale if not set
                if tensor_group_size is None and len(original_shape) == 2 and len(scale.shape) >= 2:
                    out_features, in_features = original_shape
                    num_groups = scale.shape[1]
                    if num_groups > 0:
                        tensor_group_size = in_features // num_groups
                    else:
                        tensor_group_size = 128  # Fallback

                # Log quantization info for debugging
                logger.debug(
                    f"SDNQ: {key} - type={quant_type.value}, weight_shape={weight.shape}, "
                    f"weight_dtype={weight.dtype}, scale_shape={scale.shape}, "
                    f"original_shape={original_shape}, group_size={tensor_group_size}"
                )

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
                    group_size=tensor_group_size if quant_type == SDNQQuantizationType.UINT4_ASYM else None,
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

    # Log summary
    sdnq_count = sum(1 for v in sd.values() if isinstance(v, SDNQTensor))
    regular_count = len(sd) - sdnq_count
    print(f"[SDNQ] Loaded {sdnq_count} quantized tensors, {regular_count} regular tensors from {model_file}")
    logger.info(f"SDNQ loader: {sdnq_count} quantized tensors, {regular_count} regular tensors from {model_file}")

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
