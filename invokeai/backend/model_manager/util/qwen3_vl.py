"""Shared helpers for loading Qwen3-VL encoder checkpoints (used by Krea-2 and MiniMax H3)."""

from typing import Any


def normalize_qwen3vl_rope_config(config: Any) -> Any:
    """Mirror Qwen3-VL rope_parameters into rope_scaling for Transformers compatibility.

    Some Qwen3-VL checkpoints store rope settings under ``rope_parameters``, but the installed
    transformers' Qwen3VL rotary embedding reads ``rope_scaling`` (None there) and crashes.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        rope_params = getattr(text_config, "rope_parameters", None)
        if getattr(text_config, "rope_scaling", None) is None and rope_params is not None:
            text_config.rope_scaling = rope_params
    return config
