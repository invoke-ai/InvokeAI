from typing import Any, Dict

import torch

from invokeai.backend.flux.ip_adapter.xlabs_ip_adapter_flux import XlabsIpAdapterParams


def is_state_dict_xlabs_ip_adapter(sd: Dict[str, Any]) -> bool:
    """Is the state dict for an XLabs FLUX IP-Adapter model?

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision.
    """
    # If all of the expected keys are present, then this is very likely an XLabs IP-Adapter model.
    expected_keys = {
        "double_blocks.0.processor.ip_adapter_double_stream_k_proj.bias",
        "double_blocks.0.processor.ip_adapter_double_stream_k_proj.weight",
        "double_blocks.0.processor.ip_adapter_double_stream_v_proj.bias",
        "double_blocks.0.processor.ip_adapter_double_stream_v_proj.weight",
        "ip_adapter_proj_model.norm.bias",
        "ip_adapter_proj_model.norm.weight",
        "ip_adapter_proj_model.proj.bias",
        "ip_adapter_proj_model.proj.weight",
    }

    if expected_keys.issubset(sd.keys()):
        return True
    return False


def infer_xlabs_ip_adapter_params_from_state_dict(state_dict: dict[str, torch.Tensor]) -> XlabsIpAdapterParams:
    num_double_blocks = 0
    context_dim = 0
    hidden_dim = 0

    # Count the number of double blocks.
    double_block_index = 0
    while f"double_blocks.{double_block_index}.processor.ip_adapter_double_stream_k_proj.weight" in state_dict:
        double_block_index += 1
    num_double_blocks = double_block_index

    hidden_dim = state_dict["double_blocks.0.processor.ip_adapter_double_stream_k_proj.weight"].shape[0]
    context_dim = state_dict["double_blocks.0.processor.ip_adapter_double_stream_k_proj.weight"].shape[1]
    clip_embeddings_dim = state_dict["ip_adapter_proj_model.proj.weight"].shape[1]
    clip_extra_context_tokens = state_dict["ip_adapter_proj_model.proj.weight"].shape[0] // context_dim

    return XlabsIpAdapterParams(
        num_double_blocks=num_double_blocks,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        clip_embeddings_dim=clip_embeddings_dim,
        clip_extra_context_tokens=clip_extra_context_tokens,
    )
