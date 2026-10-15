"""How a Qwen3-VL encoder variant is named to a user.

Two architectures condition on Qwen3-VL and neither can use the other's size, so both loader nodes
refuse the wrong one by name. The refusal is only useful if the name is one the user recognises
from the picker -- and a third config class (MiniMax H3's truncated Qwen3-VL-32B) shares the model
type while carrying no variant at all, so `None` has to render as something better than "None".
"""

from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType

_LABELS = {
    Qwen3VLVariantType.Qwen3VL_4B: "4B",
    Qwen3VLVariantType.Qwen3VL_8B: "8B",
}


def variant_label(variant: object) -> str:
    """A human name for a recorded Qwen3-VL variant, or a description of its absence."""
    if isinstance(variant, Qwen3VLVariantType):
        return _LABELS[variant]
    return "of an unrecorded size" if variant is None else str(getattr(variant, "value", variant))
