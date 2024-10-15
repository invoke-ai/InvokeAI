from typing import Any, Dict


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
