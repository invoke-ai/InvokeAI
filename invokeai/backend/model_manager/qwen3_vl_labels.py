"""How a Qwen3-VL encoder variant is named to a user.

Several architectures condition on Qwen3-VL and none can use another size than its own, so their loader
nodes refuse the wrong one by name. The refusal is only useful if the name is one the user recognises
from the picker -- and a third config class (MiniMax H3's truncated Qwen3-VL-32B) shares the model
type while carrying no variant at all, so `None` has to render as something better than "None".
"""

from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType

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
