"""How a Qwen3-VL encoder variant is named to a user.

Several architectures condition on Qwen3-VL and none can use another size than its own, so their loader
nodes refuse the wrong one by name. The refusal is only useful if the name is one the user recognises
from the picker -- and a third config class (MiniMax H3's truncated Qwen3-VL-32B) shares the model
type while carrying no variant at all, so `None` has to render as something better than "None".
"""

from typing import Any

from invokeai.backend.model_manager.taxonomy import ModelFormat, ModelType, Qwen3VLVariantType

_LABELS = {
    Qwen3VLVariantType.Qwen3VL_4B: "4B",
    Qwen3VLVariantType.Qwen3VL_8B: "8B",
}


_HIDDEN_SIZES = {
    Qwen3VLVariantType.Qwen3VL_4B: 2560,
    Qwen3VLVariantType.Qwen3VL_8B: 4096,
}


def variant_label(variant: object) -> str:
    """A human name for a recorded Qwen3-VL variant, or a description of its absence."""
    if isinstance(variant, Qwen3VLVariantType):
        return _LABELS[variant]
    return "of an unrecorded size" if variant is None else str(getattr(variant, "value", variant))


def raise_for_wrong_qwen3_vl_variant(
    encoder_name: str, variant: object, expected: Qwen3VLVariantType, architecture: str
) -> None:
    """Refuse an encoder of the wrong size before it loads.

    The wrong size would otherwise load in full and then fail on a shape error naming a tensor, which says
    nothing about what the user picked.
    """
    if variant is not expected:
        raise ValueError(
            f"'{encoder_name}' is the Qwen3-VL {variant_label(variant)} encoder. {architecture} conditions on the "
            f"{_LABELS[expected]} one (hidden size {_HIDDEN_SIZES[expected]})."
        )


def raise_for_unfit_vision_tower(
    tower_config: Any, encoder_config: Any | None, expected: Qwen3VLVariantType, architecture: str
) -> None:
    """Refuse a selected mmproj vision tower that cannot serve the architecture's GGUF encoder.

    Refused beside any other encoder rather than ignored: those bring a tower of their own, and a stale
    selection would silently decide nothing while looking as if it did. `encoder_config` is None when no
    standalone encoder was selected.
    """
    if tower_config.type is not ModelType.Qwen3VLVisionTower:
        raise ValueError(f"'{tower_config.name}' is not a Qwen3-VL vision tower (the llama.cpp mmproj file).")
    tower_variant = getattr(tower_config, "variant", None)
    if tower_variant is not expected:
        raise ValueError(
            f"'{tower_config.name}' is the Qwen3-VL {variant_label(tower_variant)} vision tower. "
            f"{architecture} reads references with the {_LABELS[expected]} one (hidden size {_HIDDEN_SIZES[expected]})."
        )
    if encoder_config is None or encoder_config.format is not ModelFormat.GGUFQuantized:
        raise ValueError(
            f"The vision tower '{tower_config.name}' pairs only with a GGUF Qwen3-VL encoder, which has none. "
            "Any other encoder brings its own: clear the vision tower."
        )
