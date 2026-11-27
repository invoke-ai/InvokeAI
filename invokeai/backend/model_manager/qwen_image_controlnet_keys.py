"""The key signature of InstantX's Qwen-Image ControlNets, as one predicate.

The layout is diffusers' `QwenImageControlNetModel`: a trimmed copy of the Qwen-Image transformer (`img_in`, `txt_in`,
`txt_norm`, `transformer_blocks.N`) plus `controlnet_x_embedder` and one `controlnet_blocks.N` projection per block.
That puts it inside two other probes' nets: the FLUX InstantX ControlNet probe reads the same `controlnet_*` names, and
the Qwen-Image main model probe the same `img_in`/`txt_in`/`txt_norm`. Both have to refuse exactly what the Qwen-Image
ControlNet config claims, and the main checkpoint loader has to recognise a file an earlier version recorded as a main
model, so all four read the signature here rather than restating it.

Imports nothing, like `checkpoint_prefix.py`: identification and the FLUX state-dict utilities both reach it.
"""

from collections.abc import Container

QWEN_IMAGE_INSTANTX_CONTROLNET_KEYS = (
    "controlnet_x_embedder.weight",
    "controlnet_blocks.0.weight",
    # Qwen-Image's dual-stream block and text input norm: a FLUX ControlNet has `norm1`/`ff` blocks and a
    # `context_embedder` instead.
    "transformer_blocks.0.img_mod.1.weight",
    "txt_norm.weight",
)


def is_qwen_image_instantx_controlnet(state_dict: Container[object]) -> bool:
    """Whether the state dict has InstantX's Qwen-Image ControlNet layout, its inpainting variant included."""
    return all(key in state_dict for key in QWEN_IMAGE_INSTANTX_CONTROLNET_KEYS)
