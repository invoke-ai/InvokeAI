# ERNIE-Image LoRA prefix constants and key-shape detection helpers.
#
# ERNIE-Image LoRAs target diffusers' ``ErnieImageTransformer2DModel`` under its own module names, behind one
# of the usual container prefixes. Measured on published files (ai-toolkit, Civitai's trainer -- which runs
# ai-toolkit -- and a PEFT extraction of the Turbo distillation):
#
#     diffusion_model.layers.0.self_attention.to_q.lora_A.weight
#     diffusion_model.layers.0.mlp.gate_proj.lora_A.weight
#     diffusion_model.layers.0.mlp.linear_fc2.lora_A.weight
#     transformer.layers.0.self_attention.to_out.0.lora_A.weight
#     transformer.adaLN_modulation.1.lora_A.weight
#
# The detection helpers are shared with ``configs/lora.py``, so the ERNIE-Image probe, the Z-Image probe's
# exclusion and the conversion agree on what counts as an ERNIE-Image LoRA. This file imports nothing from the
# config package, which keeps it free of circular imports.

import re

# Prefix for ERNIE-Image transformer LoRA layers in the ModelPatchRaw layer dict; the denoise node patches the
# transformer under it. Same convention as every other transformer family.
ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# The container prefixes the conversion strips, longest first so a nested PEFT prefix is not left with a tail.
ERNIE_IMAGE_LORA_KEY_PREFIXES = (
    "base_model.model.transformer.",
    "model.diffusion_model.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
)

# Anchored to the key start: the conversion strips exactly one of the prefixes above, so detection must not
# admit a key it would then map onto a module path the transformer does not have.
_PREFIX_RE = r"^(?:" + "|".join(re.escape(p) for p in ERNIE_IMAGE_LORA_KEY_PREFIXES) + r")?"

# Block submodules that only ERNIE-Image has among the supported transformers. Z-Image keeps its blocks under
# ``layers.N`` too, but names them ``attention`` and ``feed_forward``; Ideogram 4 uses ``attention.qkv`` and
# ``feed_forward.w1..3``. ``mlp.gate_proj`` / ``up_proj`` are deliberately not markers: Qwen3 and other LLM
# layers use the same names, so they only count once a marker has identified the file.
_ERNIE_IMAGE_MARKER_RE = re.compile(
    _PREFIX_RE + r"layers\.\d+\.(?:self_attention\.(?:to_q|to_k|to_v|to_out\.0)|mlp\.linear_fc2)\."
)

# Every module of ``ErnieImageTransformer2DModel`` a LoRA can patch (its Linear and Conv2d layers). A layer
# outside this set would be skipped at generation with only a log warning, so the probe refuses the file.
_ERNIE_IMAGE_MODULE_RE = re.compile(
    _PREFIX_RE
    + r"(?:layers\.\d+\.(?:self_attention\.(?:to_q|to_k|to_v|to_out\.0)|mlp\.(?:gate_proj|up_proj|linear_fc2))"
    + r"|adaLN_modulation\.1|final_linear|final_norm\.linear|text_proj|time_embedding\.linear_[12]|x_embedder\.proj)$"
)

# Any of these means a different architecture; an ERNIE-Image LoRA never carries them.
_NON_ERNIE_IMAGE_ANTI_RES = (
    re.compile(_PREFIX_RE + r"layers\.\d+\.(?:attention|feed_forward)\."),  # Z-Image / Ideogram 4 blocks
    re.compile(r"(?:^|\.)(?:context_refiner|noise_refiner)\."),  # Z-Image refiners
    re.compile(r"(?:^|\.|_)(?:double_blocks|single_blocks|single_transformer_blocks|transformer_blocks)[\._]\d+"),
    re.compile(r"(?:img_mod|txt_mod|img_mlp|txt_mlp|img_attn|txt_attn)[\._]"),  # Qwen-Image / FLUX
    re.compile(r"(?:^|\.)blocks\.\d+\."),  # Wan / Anima / MiniMax H3
    re.compile(r"(?:^|_)lora_(?:unet|te\d?)_"),  # kohya-flattened SD / FLUX / Z-Image keys
)

# The low-rank and LyCORIS parameter names the conversion builds layers from. DoRA's magnitude vectors are
# deliberately absent: the sidecar wrappers of the quantized builds hand a patch only a placeholder weight,
# and DoRA needs the real one, so a DoRA file would fail half-way into a generation instead of at install.
ERNIE_IMAGE_LORA_PARAM_NAMES = (
    "lora_A.weight",
    "lora_B.weight",
    "lora_down.weight",
    "lora_up.weight",
    "lora_mid.weight",
    "alpha",
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
    "hada_w1_a",
    "hada_w1_b",
    "hada_w2_a",
    "hada_w2_b",
    "hada_t1",
    "hada_t2",
)


def has_ernie_image_markers(str_keys: list[str]) -> bool:
    """True if any key names an ERNIE-Image-only block submodule."""
    return any(_ERNIE_IMAGE_MARKER_RE.search(k) for k in str_keys)


def is_ernie_image_lora(str_keys: list[str]) -> bool:
    """True if a key names an ERNIE-Image-only block submodule and none belongs to another architecture."""
    return has_ernie_image_markers(str_keys) and not any(
        anti.search(k) for k in str_keys for anti in _NON_ERNIE_IMAGE_ANTI_RES
    )


def split_ernie_image_lora_key(key: str) -> tuple[str, str] | None:
    """Split a key into ``(layer path, parameter name)``, or ``None`` if its tail is not a supported parameter."""
    for param_name in ERNIE_IMAGE_LORA_PARAM_NAMES:
        if key.endswith("." + param_name):
            return key[: -len(param_name) - 1], param_name
    return None


def unsupported_ernie_image_lora_keys(str_keys: list[str]) -> list[str]:
    """Keys whose tail is not a parameter the conversion can build a layer from (DoRA, biases, ...)."""
    return [k for k in str_keys if split_ernie_image_lora_key(k) is None]


def ernie_image_lora_refusal(str_keys: list[str]) -> str | None:
    """Why an ERNIE-Image LoRA cannot be applied, or ``None`` if every tensor converts.

    The probe refuses the install with this message and the conversion raises it, so a file is never accepted and
    then half-applied.
    """
    unsupported = unsupported_ernie_image_lora_keys(str_keys)
    if unsupported:
        return (
            f"{len(unsupported)} tensor(s) cannot be applied, e.g. {unsupported[0]!r}. Supported are low-rank "
            "(lora_A/lora_B, lora_down/lora_up), LoKR and LoHA layers; DoRA is not."
        )
    layer_paths = {split[0] for k in str_keys if (split := split_ernie_image_lora_key(k)) is not None}
    unknown = sorted(path for path in layer_paths if not _ERNIE_IMAGE_MODULE_RE.search(path))
    if unknown:
        return f"{len(unknown)} layer(s) target no ERNIE-Image transformer module, e.g. {unknown[0]!r}."
    return None
