from typing import Any, Dict


def is_state_dict_xlabs_controlnet(sd: Dict[str, Any]) -> bool:
    """Is the state dict for an XLabs ControlNet model?

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision.
    """
    # If all of the expected keys are present, then this is very likely an XLabs ControlNet model.
    expected_keys = {
        "controlnet_blocks.0.bias",
        "controlnet_blocks.0.weight",
        "input_hint_block.0.bias",
        "input_hint_block.0.weight",
        "pos_embed_input.bias",
        "pos_embed_input.weight",
    }

    if expected_keys.issubset(sd.keys()):
        return True
    return False


def is_state_dict_instantx_controlnet(sd: Dict[str, Any]) -> bool:
    """Is the state dict for an InstantX ControlNet model?

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision.
    """
    # If all of the expected keys are present, then this is very likely an InstantX ControlNet model.
    expected_keys = {
        "controlnet_blocks.0.bias",
        "controlnet_blocks.0.weight",
        "controlnet_single_blocks.0.bias",
        "controlnet_single_blocks.0.weight",
        "controlnet_x_embedder.bias",
        "controlnet_x_embedder.weight",
    }

    if expected_keys.issubset(sd.keys()):
        return True
    return False
