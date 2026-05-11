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

# Cosmos DiT subcomponent names that ALSO appear in Wan (cross_attn, self_attn)
# plus those unique to Cosmos. Used by ``anima_lora_conversion_utils`` to find
# block layers during state-dict conversion, where the architecture is already
# known to be Anima.
_COSMOS_DIT_SUBCOMPONENTS_RE = r"(cross_attn|self_attn|mlp|adaln_modulation)"

# Kohya format: lora_unet_[llm_adapter_]blocks_N_<cosmos_subcomponent>
_KOHYA_ANIMA_RE = re.compile(r"lora_unet_(llm_adapter_)?blocks_\d+_" + _COSMOS_DIT_SUBCOMPONENTS_RE)

# PEFT format: <prefix>.blocks.N.<cosmos_subcomponent>
_PEFT_ANIMA_RE = re.compile(
    r"(diffusion_model|transformer|base_model\.model\.transformer)\.blocks\.\d+\." + _COSMOS_DIT_SUBCOMPONENTS_RE
)


# Subcomponents *uniquely* identifying Anima/Cosmos DiT: ``mlp`` and
# ``adaln_modulation`` (Wan calls those ``ffn`` and ``modulation`` respectively),
# plus the Cosmos attention naming with a ``_proj`` suffix on the projection
# letter (Wan native uses bare ``.q``/``.k``/``.v``/``.o`` — no ``_proj``).
#
# Used by the probe in ``configs/lora.py`` to make Anima-LoRA detection
# *mutually exclusive* with Wan-LoRA detection: a state dict carrying only
# ``cross_attn.q`` / ``ffn.0`` (Wan native) will NOT match here, regardless of
# the order configs are tried.
_COSMOS_DIT_EXCLUSIVE_SUBCOMPONENTS_RE = (
    r"(mlp|adaln_modulation|"
    r"(?:cross|self)_attn[._](?:[qkv]_proj|output_proj))"
)

_KOHYA_ANIMA_STRICT_RE = re.compile(
    r"lora_unet_(llm_adapter_)?blocks_\d+_" + _COSMOS_DIT_EXCLUSIVE_SUBCOMPONENTS_RE
)
_PEFT_ANIMA_STRICT_RE = re.compile(
    r"(diffusion_model|transformer|base_model\.model\.transformer)\.blocks\.\d+\."
    + _COSMOS_DIT_EXCLUSIVE_SUBCOMPONENTS_RE
)


def has_cosmos_dit_kohya_keys(str_keys: list[str]) -> bool:
    """Loose detector — matches any Cosmos-shaped block submodule including
    those whose names collide with Wan (``cross_attn``, ``self_attn``).

    For probe disambiguation between Anima and Wan, prefer
    ``has_cosmos_dit_kohya_keys_strict``. This loose form is still useful
    inside the Anima conversion utility, where the architecture is already
    confirmed to be Anima and we just need to enumerate matching layers.
    """
    return any(_KOHYA_ANIMA_RE.search(k) is not None for k in str_keys)


def has_cosmos_dit_peft_keys(str_keys: list[str]) -> bool:
    """Loose PEFT-format detector — see ``has_cosmos_dit_kohya_keys`` docstring."""
    return any(_PEFT_ANIMA_RE.search(k) is not None for k in str_keys)


def has_cosmos_dit_kohya_keys_strict(str_keys: list[str]) -> bool:
    """Strict Kohya detector requiring an Anima-exclusive submodule (``mlp``,
    ``adaln_modulation``, or Cosmos's ``_proj``-suffixed attention names).

    Mutually exclusive with the Wan LoRA probe — no Wan LoRA can satisfy this.
    """
    return any(_KOHYA_ANIMA_STRICT_RE.search(k) is not None for k in str_keys)


def has_cosmos_dit_peft_keys_strict(str_keys: list[str]) -> bool:
    """Strict PEFT detector. See ``has_cosmos_dit_kohya_keys_strict`` docstring."""
    return any(_PEFT_ANIMA_STRICT_RE.search(k) is not None for k in str_keys)
