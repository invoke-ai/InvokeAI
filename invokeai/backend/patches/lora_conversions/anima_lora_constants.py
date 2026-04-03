# Anima LoRA prefix constants
# These prefixes are used for key mapping when applying LoRA patches to Anima models

import re

# Prefix for Anima transformer (Cosmos DiT architecture) LoRA layers
ANIMA_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# Prefix for Qwen3 text encoder LoRA layers
ANIMA_LORA_QWEN3_PREFIX = "lora_qwen3-"

# ---------------------------------------------------------------------------
# Cosmos DiT detection helpers
#
# Shared between ``anima_lora_conversion_utils.is_state_dict_likely_anima_lora``
# and the config probing code in ``configs/lora.py``.  Kept here (rather than
# in ``anima_lora_conversion_utils``) to avoid circular imports.
# ---------------------------------------------------------------------------

# Cosmos DiT subcomponent names unique to the Anima / Cosmos Predict2 architecture.
_COSMOS_DIT_SUBCOMPONENTS_RE = r"(cross_attn|self_attn|mlp|adaln_modulation)"

# Kohya format: lora_unet_[llm_adapter_]blocks_N_<cosmos_subcomponent>
_KOHYA_ANIMA_RE = re.compile(r"lora_unet_(llm_adapter_)?blocks_\d+_" + _COSMOS_DIT_SUBCOMPONENTS_RE)

# PEFT format: <prefix>.blocks.N.<cosmos_subcomponent>
_PEFT_ANIMA_RE = re.compile(
    r"(diffusion_model|transformer|base_model\.model\.transformer)\.blocks\.\d+\." + _COSMOS_DIT_SUBCOMPONENTS_RE
)


def has_cosmos_dit_kohya_keys(str_keys: list[str]) -> bool:
    """Check for Kohya-style keys targeting Cosmos DiT blocks with specific subcomponents.

    Requires both the ``lora_unet_[llm_adapter_]blocks_N_`` prefix **and** a
    Cosmos DiT subcomponent name (cross_attn, self_attn, mlp, adaln_modulation)
    to avoid false-positives on other architectures that might also use bare
    ``blocks`` in their key paths.
    """
    return any(_KOHYA_ANIMA_RE.search(k) is not None for k in str_keys)


def has_cosmos_dit_peft_keys(str_keys: list[str]) -> bool:
    """Check for diffusers PEFT keys targeting Cosmos DiT blocks with specific subcomponents."""
    return any(_PEFT_ANIMA_RE.search(k) is not None for k in str_keys)
