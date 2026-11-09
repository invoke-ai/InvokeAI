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

# Module paths that only ERNIE-Image has among the supported transformers, anchored to the key start behind a
# container prefix. Z-Image keeps its blocks under ``layers.N`` too, but names them ``attention`` and
# ``feed_forward``; Ideogram 4 uses ``attention.qkv`` and ``feed_forward.w1..3``. The transformer's top-level
# modules ``adaLN_modulation.1`` and ``final_norm.linear`` exist in no other one either (``time_embedding`` and
# ``text_proj`` are not markers: SD UNets have them), which identifies a file that touches only those and the
# MLP. ``mlp.gate_proj`` / ``up_proj`` alone are deliberately not markers: a PEFT LoRA over a Qwen3-style LLM
# uses the same names under the same ``base_model.model.`` prefix, so they only count once a marker has
# identified the file.
_ERNIE_IMAGE_MARKER_RE = re.compile(
    _PREFIX_RE
    + r"(?:layers\.\d+\.(?:self_attention\.(?:to_q|to_k|to_v|to_out\.0)|mlp\.linear_fc2)\."
    + r"|(?:adaLN_modulation\.1|final_norm\.linear)\.)"
)

# Every Linear of ``ErnieImageTransformer2DModel`` a LoRA can patch. ``x_embedder.proj`` is its only Conv2d: no
# published file touches it and the working-memory estimate prices Linears only, so it counts as unplaceable.
# A layer outside this set would be skipped at generation with only a log warning, so the probe refuses the file.
_ERNIE_IMAGE_MODULE_RE = re.compile(
    _PREFIX_RE
    + r"(?:layers\.\d+\.(?:self_attention\.(?:to_q|to_k|to_v|to_out\.0)|mlp\.(?:gate_proj|up_proj|linear_fc2))"
    + r"|adaLN_modulation\.1|final_linear|final_norm\.linear|text_proj|time_embedding\.linear_[12])$"
)

# The released transformer's geometry (``ERNIE_IMAGE_TRANSFORMER_CONFIG`` in the model loader, which this file
# cannot import): its block count and the (out, in) shape of each patchable weight.
ERNIE_IMAGE_NUM_LAYERS = 36
_LAYER_INDEX_RE = re.compile(_PREFIX_RE + r"layers\.(\d+)\.")
_BLOCK_WEIGHT_SHAPES = {
    "self_attention.to_q": (4096, 4096),
    "self_attention.to_k": (4096, 4096),
    "self_attention.to_v": (4096, 4096),
    "self_attention.to_out.0": (4096, 4096),
    "mlp.gate_proj": (12288, 4096),
    "mlp.up_proj": (12288, 4096),
    "mlp.linear_fc2": (4096, 12288),
}
_TOP_WEIGHT_SHAPES = {
    "adaLN_modulation.1": (24576, 4096),
    "final_linear": (128, 4096),
    "final_norm.linear": (8192, 4096),
    "text_proj": (4096, 3072),
    "time_embedding.linear_1": (4096, 4096),
    "time_embedding.linear_2": (4096, 4096),
}


def ernie_image_weight_shape(module: str) -> tuple[int, int] | None:
    """The (out, in) shape of the transformer weight at ``module`` (container prefix stripped), or ``None``."""
    if block := re.fullmatch(r"layers\.(\d+)\.(.+)", module):
        return _BLOCK_WEIGHT_SHAPES.get(block.group(2)) if int(block.group(1)) < ERNIE_IMAGE_NUM_LAYERS else None
    return _TOP_WEIGHT_SHAPES.get(module)


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
    """Why an ERNIE-Image LoRA cannot be applied from its key names alone, or ``None``.

    Tensor-level defects (incomplete layers, shapes) are found by ``ernie_image_lora_defect``, which starts here.
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
    beyond = sorted(
        path
        for path in layer_paths
        if (block := _LAYER_INDEX_RE.search(path)) and int(block.group(1)) >= ERNIE_IMAGE_NUM_LAYERS
    )
    if beyond:
        return f"{len(beyond)} layer(s) target a block beyond the model's {ERNIE_IMAGE_NUM_LAYERS}, e.g. {beyond[0]!r}."
    return None
