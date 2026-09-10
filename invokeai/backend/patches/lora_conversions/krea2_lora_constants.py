"""Krea-2 LoRA prefix constants and kohya key-reconstruction helpers.

The prefixes namespace LoRA patch keys when applying them to Krea-2 models.

The kohya/LyCORIS reconstruction helpers live here rather than in ``krea2_lora_conversion_utils`` so that
``model_manager.configs.lora`` can identify a flattened Krea-2 LoRA without importing the converter: the
converter pulls in the patch layers, which import ``model_manager.load``, closing an import cycle back into
``model_manager.configs``. Same reason ``anima_lora_constants`` exists.
"""

from invokeai.backend.patches.lora_conversions.kohya_key_utils import (
    INDEX_PLACEHOLDER,
    ParsingTree,
    insert_periods_into_kohya_key,
)

# Prefix for Krea-2 transformer (Krea2Transformer2DModel) LoRA layers.
KREA2_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# Prefix for Krea-2 Qwen3-VL text encoder LoRA layers.
KREA2_LORA_QWEN3VL_PREFIX = "lora_qwen3vl-"


# --- Kohya / LyCORIS (flattened) -> native key mapping ---------------------------------------------------------
# sd-scripts and LyCORIS flatten the module path (``path.replace(".", "_")``) and prefix it with
# ``lora_unet_``, e.g. ``lora_unet_blocks_6_attn_wv.lora_down.weight``. Flattening is lossy — nothing in the key
# records where a '_' used to be a '.' — so we reconstruct the dotted path against the native module vocabulary
# below and accept it only if it lands on a leaf. A key we cannot reconstruct with certainty is left untouched
# rather than rewritten into a plausible-looking key that matches no module.
KREA2_KOHYA_PREFIX = "lora_unet_"

# Native Krea-2 transformer/text-fusion block leaves. Only the Linears are listed: the non-Linear natives
# (``mod.lin``, ``prenorm``/``postnorm``, ``attn.qknorm.*``, ``last.norm``/``last.modulation``) have no Linear
# counterpart in the diffusers layout — ``mod.lin`` for instance is folded into the ``scale_shift_table``
# parameter — so an adapter targeting them cannot be applied, and renaming it anyway would turn "unsupported"
# into a silent no-op.
_NATIVE_KREA2_BLOCK_SUBTREE: ParsingTree = {
    "attn": {"wq": {}, "wk": {}, "wv": {}, "wo": {}, "gate": {}},
    "mlp": {"gate": {}, "up": {}, "down": {}},
}

# Parsing tree for the native (ComfyUI) Krea-2 module layout, i.e. the keys the renames in
# ``krea2_lora_conversion_utils`` understand. Walking it resolves the flattened form's only real ambiguity — ``layerwise_blocks`` / ``refiner_blocks`` are
# the native components that themselves contain an underscore.
_KREA2_NATIVE_KOHYA_PARSING_TREE: ParsingTree = {
    "blocks": {INDEX_PLACEHOLDER: _NATIVE_KREA2_BLOCK_SUBTREE},
    "txtfusion": {
        "layerwise_blocks": {INDEX_PLACEHOLDER: _NATIVE_KREA2_BLOCK_SUBTREE},
        "refiner_blocks": {INDEX_PLACEHOLDER: _NATIVE_KREA2_BLOCK_SUBTREE},
        "projector": {},
    },
    "first": {},
    # Literal indices rather than INDEX_PLACEHOLDER: these are ``nn.Sequential`` stages, and only the
    # positions listed in the converter's ``_NATIVE_KREA2_TOP_LEVEL_RENAMES`` hold a Linear — the rest are
    # activations with no weights. Accepting any index would rewrite e.g. ``lora_unet_tmlp_1`` into ``tmlp.1.*``, which the
    # native pass then does not recognize, leaving a half-converted key instead of the untouched original.
    "tmlp": {"0": {}, "2": {}},
    "tproj": {"1": {}},
    "txtmlp": {"1": {}, "3": {}},
    "last": {"linear": {}},
}


def _kohya_module_path_is_leaf(module_path: str, parsing_tree: ParsingTree) -> bool:
    """True if a dotted module path walks the tree all the way to a leaf.

    ``insert_periods_into_kohya_key`` only rejects *leftover* tokens, so a prefix of a real path (e.g.
    ``blocks.0.attn``) parses cleanly without naming a module. Requiring a leaf rejects those.
    """
    subtree = parsing_tree
    for component in module_path.split("."):
        # Mirror ``insert_periods_into_kohya_key``'s precedence: an exact match wins over the index
        # placeholder. Without that, a numeric component would always be looked up as INDEX_PLACEHOLDER and
        # a tree enumerating the specific indices it accepts (``tmlp`` below) could never reach its leaves.
        if component in subtree:
            subtree = subtree[component]
        elif component.isnumeric() and INDEX_PLACEHOLDER in subtree:
            subtree = subtree[INDEX_PLACEHOLDER]
        else:
            return False
    return not subtree


def unflatten_kohya_krea2_module_path(flat_path: str) -> str | None:
    """Reconstruct a dotted native Krea-2 module path from its kohya-flattened form.

    Returns ``None`` when the reconstruction is not a native module path the converter can map, in which case
    the caller must leave the key alone.
    """
    try:
        module_path = insert_periods_into_kohya_key(flat_path, _KREA2_NATIVE_KOHYA_PARSING_TREE)
    except ValueError:
        # Tokens left over: not a native Krea-2 module path.
        return None
    return module_path if _kohya_module_path_is_leaf(module_path, _KREA2_NATIVE_KOHYA_PARSING_TREE) else None


def split_kohya_krea2_key(key: str | int) -> tuple[str, str, str] | None:
    """Split a kohya/LyCORIS key into (flattened module path, separator, weight suffix).

    Returns ``None`` for anything not in the kohya layout, including the non-string keys that ``.pt`` /
    ``.ckpt`` sources can carry. The flattened module path runs up to the first '.'; the weight suffix
    (``lora_down.weight``, ``alpha``, ...) follows it. Some writers emit a doubled separator after the
    prefix, hence the ``lstrip``. Every caller splits through here so the converter's per-module gate, the
    rewrite it guards, and identification can never disagree about which module a key belongs to.
    """
    if not isinstance(key, str) or not key.startswith(KREA2_KOHYA_PREFIX):
        return None
    flat_path, dot, weight_suffix = key[len(KREA2_KOHYA_PREFIX) :].lstrip("_").partition(".")
    return flat_path, dot, weight_suffix


def is_kohya_krea2_lora_key(key: str | int) -> bool:
    """True if ``key`` is a kohya-flattened key naming a Krea-2 module the converter can reconstruct.

    Identification (``model_manager.configs.lora``) uses this instead of matching ``lora_unet_<module>``
    prefixes. That spelling is not Krea-2's alone: Wan writes ``lora_unet_blocks_<idx>_...`` and Anima
    ``lora_unet_[llm_adapter_]blocks_<idx>_...``, so a prefix match sweeps their kohya LoRAs into the
    explicit Krea-2 override, where they install and then silently no-op at generation time. Reconstructing
    the path against the native Krea-2 module vocabulary rejects them - ``self_attn``, ``cross_attn`` and
    ``mlp_layer0`` are not leaves in it - while still accepting the doubled-separator spelling that the
    converter tolerates but no ``lora_unet_`` prefix spells out.
    """
    split = split_kohya_krea2_key(key)
    return split is not None and unflatten_kohya_krea2_module_path(split[0]) is not None
