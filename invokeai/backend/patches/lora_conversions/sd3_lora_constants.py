# SD 3 / 3.5 LoRA prefix constants and key-shape detection helpers.
#
# SD 3 LoRAs come in four namings, all measured on published files:
#
# - diffusers / ai-toolkit (`transformer.`) and PEFT adapters (`base_model.model.`): diffusers' own module names --
#       transformer.transformer_blocks.0.attn.to_q.lora_A.weight              (also add_{q,k,v}_proj, to_add_out)
#       transformer.transformer_blocks.0.attn2.to_q.lora_A.weight             (3.5 Medium's dual attention, blocks 0-12)
#       transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight          (ff_context, norm1(_context).linear)
#       transformer.pos_embed.proj.lora_A.weight                              (Conv2d), context_embedder, proj_out, ...
#   plus, with `--train_text_encoder`, `text_encoder.` (CLIP-L) and `text_encoder_2.` (CLIP-G);
# - SimpleTuner's LyCORIS: the same names flattened behind `lycoris_` (`lycoris_transformer_blocks_0_attn_to_q.lokr_w1`);
# - kohya sd-scripts (`networks.lora_sd3`): the original MMDiT names, with fused attention projections --
#       lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight
#       lora_unet_joint_blocks_0_context_block_mlp_fc1.lora_down.weight
#   and `lora_te1_` / `lora_te2_` / `lora_te3_` for CLIP-L, CLIP-G and T5.
#
# Qwen-Image uses the very same attention names as diffusers' SD 3 (`add_q_proj`, `to_add_out`), so names cannot tell
# the two apart; widths can. SD 3 Medium and 3.5 Medium are 1536 wide, 3.5 Large 2432; Qwen-Image is 3072.
#
# The detection helpers are shared with ``configs/lora.py``, so the SD 3 probe, the other probes' exclusions and the
# conversion agree. This file imports nothing from the config package, which keeps it free of cycles.

import re
from typing import Any

import torch

from invokeai.backend.model_manager.taxonomy import SD3VariantType

SD3_LORA_TRANSFORMER_PREFIX = "lora_transformer-"
SD3_LORA_CLIP_L_PREFIX = "lora_clip_l-"
SD3_LORA_CLIP_G_PREFIX = "lora_clip_g-"

# Hidden width of the released transformers, by the variant it identifies.
SD3_MEDIUM_WIDTH = 1536
SD3_LARGE_WIDTH = 2432
SD3_WIDTHS = frozenset({SD3_MEDIUM_WIDTH, SD3_LARGE_WIDTH})

# Every module of diffusers' `SD3Transformer2DModel` a LoRA can patch, `{i}` standing for a block index.
SD3_TRANSFORMER_MODULE_TEMPLATES = (
    "transformer_blocks.{i}.attn.to_q",
    "transformer_blocks.{i}.attn.to_k",
    "transformer_blocks.{i}.attn.to_v",
    "transformer_blocks.{i}.attn.to_out.0",
    "transformer_blocks.{i}.attn.add_q_proj",
    "transformer_blocks.{i}.attn.add_k_proj",
    "transformer_blocks.{i}.attn.add_v_proj",
    "transformer_blocks.{i}.attn.to_add_out",
    "transformer_blocks.{i}.attn2.to_q",
    "transformer_blocks.{i}.attn2.to_k",
    "transformer_blocks.{i}.attn2.to_v",
    "transformer_blocks.{i}.attn2.to_out.0",
    "transformer_blocks.{i}.ff.net.0.proj",
    "transformer_blocks.{i}.ff.net.2",
    "transformer_blocks.{i}.ff_context.net.0.proj",
    "transformer_blocks.{i}.ff_context.net.2",
    "transformer_blocks.{i}.norm1.linear",
    "transformer_blocks.{i}.norm1_context.linear",
    "pos_embed.proj",
    "context_embedder",
    "proj_out",
    "norm_out.linear",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "time_text_embed.text_embedder.linear_1",
    "time_text_embed.text_embedder.linear_2",
)

_TEMPLATE_RE = {
    template: re.compile("^" + re.escape(template).replace(r"\{i\}", r"(\d+)") + "$")
    for template in SD3_TRANSFORMER_MODULE_TEMPLATES
}
# The same templates with every `.` flattened to `_`, as SimpleTuner's LyCORIS (and kohya-style trainers that keep
# diffusers names) write them. Module names contain underscores of their own, so a flattened name is resolved by
# matching it against the whole template rather than by guessing where the dots were.
_FLAT_TEMPLATE_RE = {
    template: re.compile("^" + re.escape(template.replace(".", "_")).replace(r"\{i\}", r"(\d+)") + "$")
    for template in SD3_TRANSFORMER_MODULE_TEMPLATES
}

# Container prefixes in front of diffusers module names, longest first so a nested one is not left with a tail.
SD3_LORA_TRANSFORMER_KEY_PREFIXES = ("base_model.model.transformer.", "base_model.model.", "transformer.")
SD3_LORA_FLAT_TRANSFORMER_PREFIX = "lycoris_"

# Text-encoder prefixes: (diffusers, kohya) per encoder. T5 parts are dropped with a warning.
SD3_LORA_CLIP_L_KEY_PREFIXES = ("text_encoder.", "lora_te1_")
SD3_LORA_CLIP_G_KEY_PREFIXES = ("text_encoder_2.", "lora_te2_")
SD3_LORA_T5_KEY_PREFIXES = ("text_encoder_3.", "lora_te3_")

# Low-rank and LyCORIS parameter names the conversion builds layers from. DoRA is refused: the sidecar wrappers of the
# quantized builds hand a patch only a placeholder weight, and DoRA needs the real one.
SD3_LORA_PARAM_NAMES = (
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

# Names no SD 3 LoRA carries, but the families sharing its attention names do.
_NON_SD3_ANTI_RES = (
    re.compile(r"(?:^|[._])(?:img_mlp|txt_mlp|img_mod|txt_mod)[._]"),  # Qwen-Image
    re.compile(r"(?:^|[._])attn1[._]"),  # CogView4
    re.compile(r"(?:^|[._])(?:single_transformer_blocks|single_blocks|double_blocks)[._]"),  # FLUX
    re.compile(r"(?:^|[._])(?:text_fusion|txtfusion|time_mod_proj)[._]"),  # Krea-2
)


def split_sd3_lora_key(key: str) -> tuple[str, str] | None:
    """Split a key into ``(layer path, parameter name)``, or ``None`` if its tail is not a supported parameter."""
    for param_name in SD3_LORA_PARAM_NAMES:
        if key.endswith("." + param_name):
            return key[: -len(param_name) - 1], param_name
    return None


def resolve_sd3_transformer_module(layer_path: str) -> str | None:
    """The diffusers module path a transformer layer path names, or ``None``.

    Handles the dotted container prefixes and SimpleTuner's flattened `lycoris_` spelling. kohya's MMDiT names are not
    module paths of this model and are mapped by the conversion.
    """
    for prefix in SD3_LORA_TRANSFORMER_KEY_PREFIXES:
        if layer_path.startswith(prefix):
            layer_path = layer_path[len(prefix) :]
            break
    else:
        if layer_path.startswith(SD3_LORA_FLAT_TRANSFORMER_PREFIX):
            flat = layer_path[len(SD3_LORA_FLAT_TRANSFORMER_PREFIX) :]
            for template, pattern in _FLAT_TEMPLATE_RE.items():
                if match := pattern.match(flat):
                    return template.replace("{i}", match.group(1)) if match.groups() else template
            return None
    for pattern in _TEMPLATE_RE.values():
        if pattern.match(layer_path):
            return layer_path
    return None


def group_sd3_lora_layers(state_dict: dict[str | int, Any]) -> dict[str, dict[str, torch.Tensor]]:
    """The tensors of every supported key, grouped by layer path as written in the file."""
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and (split := split_sd3_lora_key(key)) is not None:
            grouped.setdefault(split[0], {})[split[1]] = value
    return grouped


def _factor_shape(values: dict[str, torch.Tensor], name: str) -> tuple[int, int] | None:
    if name in values:
        return int(values[name].shape[0]), int(values[name].shape[1])
    if f"{name}_a" in values and f"{name}_b" in values:
        return int(values[f"{name}_a"].shape[0]), int(values[f"{name}_b"].shape[1])
    return None


def layer_delta_shape(values: dict[str, torch.Tensor]) -> tuple[int, int] | None:
    """The (out, in) shape of the weight a complete layer patches (a Conv2d's kernel folded), or ``None``.

    ``None`` also means incomplete: a low-rank pair in mixed or half spellings, one Kronecker factor, half a LoHA, or
    an `alpha` alone -- nothing a layer can be built from.
    """
    for down_name, up_name in (("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight")):
        if down_name in values and up_name in values:
            return int(values[up_name].shape[0]), int(values[down_name].shape[1])
    w1, w2 = _factor_shape(values, "lokr_w1"), _factor_shape(values, "lokr_w2")
    if w1 is not None and w2 is not None:
        return w1[0] * w2[0], w1[1] * w2[1]
    if {"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"} <= values.keys():
        return int(values["hada_w1_a"].shape[0]), int(values["hada_w1_b"].shape[1])
    return None


# kohya sd-scripts' MMDiT names -> diffusers module paths. A fused `qkv` splits into three. An adaLN projection whose
# output is twice its input -- the last context block, which only modulates, and the final layer -- has its two output
# halves (shift, scale) stored the other way round by diffusers.
_KOHYA_BLOCK_RE = re.compile(r"^lora_unet_joint_blocks_(\d+)_(x|context)_block_(.+)$")
_KOHYA_X_BLOCK = {
    "attn_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "attn_proj": ("attn.to_out.0",),
    "attn2_qkv": ("attn2.to_q", "attn2.to_k", "attn2.to_v"),
    "attn2_proj": ("attn2.to_out.0",),
    "mlp_fc1": ("ff.net.0.proj",),
    "mlp_fc2": ("ff.net.2",),
    "adaLN_modulation_1": ("norm1.linear",),
}
_KOHYA_CONTEXT_BLOCK = {
    "attn_qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
    "attn_proj": ("attn.to_add_out",),
    "mlp_fc1": ("ff_context.net.0.proj",),
    "mlp_fc2": ("ff_context.net.2",),
    "adaLN_modulation_1": ("norm1_context.linear",),
}
_KOHYA_TOP_LEVEL = {
    "lora_unet_x_embedder_proj": "pos_embed.proj",
    "lora_unet_context_embedder": "context_embedder",
    "lora_unet_t_embedder_mlp_0": "time_text_embed.timestep_embedder.linear_1",
    "lora_unet_t_embedder_mlp_2": "time_text_embed.timestep_embedder.linear_2",
    "lora_unet_y_embedder_mlp_0": "time_text_embed.text_embedder.linear_1",
    "lora_unet_y_embedder_mlp_2": "time_text_embed.text_embedder.linear_2",
    "lora_unet_final_layer_linear": "proj_out",
    "lora_unet_final_layer_adaLN_modulation_1": "norm_out.linear",
}


def kohya_mmdit_targets(layer_path: str) -> tuple[str, ...] | None:
    """The diffusers module paths a kohya MMDiT layer patches (three for a fused `qkv`), or ``None``."""
    if layer_path in _KOHYA_TOP_LEVEL:
        return (_KOHYA_TOP_LEVEL[layer_path],)
    match = _KOHYA_BLOCK_RE.match(layer_path)
    if match is None:
        return None
    block, stream, rest = match.groups()
    targets = (_KOHYA_X_BLOCK if stream == "x" else _KOHYA_CONTEXT_BLOCK).get(rest)
    return None if targets is None else tuple(f"transformer_blocks.{block}.{target}" for target in targets)


def kohya_mmdit_swaps_halves(layer_path: str, values: dict[str, torch.Tensor]) -> bool:
    """Whether a kohya layer is an adaLN projection whose output halves diffusers stores swapped."""
    shape = layer_delta_shape(values)
    return layer_path.endswith("_adaLN_modulation_1") and shape is not None and shape[0] == 2 * shape[1]


# CLIP layers, by encoder. kohya flattens the module path; diffusers keeps it behind `text_encoder(_2).`.
_KOHYA_CLIP_RE = re.compile(
    r"^lora_te[12]_text_model_encoder_layers_(\d+)_(mlp|self_attn)_(fc1|fc2|q_proj|k_proj|v_proj|out_proj)$"
)
_CLIP_MODULE_RE = re.compile(
    r"^text_model\.encoder\.layers\.\d+\.(?:mlp\.(?:fc1|fc2)|self_attn\.(?:q_proj|k_proj|v_proj|out_proj))$"
)


def sd3_text_encoder_target(layer_path: str) -> tuple[str, str | None] | None:
    """``(encoder, module path)`` for a text-encoder layer, or ``None`` if it names none.

    ``encoder`` is ``"clip_l"``, ``"clip_g"`` or ``"t5"``; T5 layers are recognized only to be dropped, so their
    module path is ``None``.
    """
    if layer_path.startswith(SD3_LORA_T5_KEY_PREFIXES):
        return "t5", None
    for encoder, prefixes in (("clip_l", SD3_LORA_CLIP_L_KEY_PREFIXES), ("clip_g", SD3_LORA_CLIP_G_KEY_PREFIXES)):
        for prefix in prefixes:
            if not layer_path.startswith(prefix):
                continue
            if prefix.endswith("_"):
                match = _KOHYA_CLIP_RE.match(layer_path)
                return (
                    (encoder, f"text_model.encoder.layers.{match.group(1)}.{match.group(2)}.{match.group(3)}")
                    if match
                    else None
                )
            module = layer_path[len(prefix) :]
            return (encoder, module) if _CLIP_MODULE_RE.match(module) else None
    return None


def _is_transformer_layer(layer_path: str) -> bool:
    return kohya_mmdit_targets(layer_path) is not None or resolve_sd3_transformer_module(layer_path) is not None


def sd3_variant_from_width(width: int | None) -> SD3VariantType | None:
    """The SD 3 size a transformer (or the LoRA trained for it) has, from its width; ``None`` for any other width."""
    return {SD3_MEDIUM_WIDTH: SD3VariantType.Medium, SD3_LARGE_WIDTH: SD3VariantType.Large}.get(width or 0)


def _attention_width(grouped: dict[str, dict[str, torch.Tensor]]) -> int | None:
    """The input width of the first attention projection a LoRA trains: the model's width, in either naming (kohya's
    fused `qkv` included)."""
    for layer_path, values in grouped.items():
        module = resolve_sd3_transformer_module(layer_path)
        kohya = kohya_mmdit_targets(layer_path)
        is_attention = (module is not None and ".attn" in module) or (kohya is not None and ".attn" in kohya[0])
        if is_attention and (shape := layer_delta_shape(values)) is not None:
            return shape[1]
    return None


def is_sd3_lora(state_dict: dict[str | int, Any]) -> bool:
    """True if the state dict trains an SD 3 attention projection at an SD 3 width, and nothing of another family.

    The width decides: diffusers' names are shared with Qwen-Image (3072 wide). A LoRA that trains no attention at all
    is not recognized.
    """
    str_keys = [k for k in state_dict if isinstance(k, str)]
    if any(anti.search(k) for k in str_keys for anti in _NON_SD3_ANTI_RES):
        return False
    return _attention_width(group_sd3_lora_layers(state_dict)) in SD3_WIDTHS


def sd3_lora_variant(state_dict: dict[str | int, Any]) -> SD3VariantType | None:
    """The SD 3 size a LoRA was trained for; ``None`` if no attention layer reveals it."""
    return sd3_variant_from_width(_attention_width(group_sd3_lora_layers(state_dict)))


def sd3_lora_refusal(state_dict: dict[str | int, Any]) -> str | None:
    """Why an SD 3 LoRA cannot be applied, or ``None`` if every tensor converts.

    The probe refuses the install with this message and the conversion raises it, so a file is never accepted and
    then half-applied -- layers the conversion could not place would otherwise be dropped with only a log warning.
    """
    str_keys = [k for k in state_dict if isinstance(k, str)]
    unsupported = [k for k in str_keys if split_sd3_lora_key(k) is None]
    if unsupported:
        return (
            f"{len(unsupported)} tensor(s) are not low-rank, LoKR or LoHA parameters, e.g. {unsupported[0]!r}. "
            "DoRA is not supported."
        )
    grouped = group_sd3_lora_layers(state_dict)
    unknown = [p for p in grouped if not (_is_transformer_layer(p) or sd3_text_encoder_target(p) is not None)]
    if unknown:
        return f"{len(unknown)} layer(s) target no SD 3 transformer or text encoder module, e.g. {unknown[0]!r}."
    incomplete = [path for path, values in grouped.items() if layer_delta_shape(values) is None]
    if incomplete:
        return (
            f"layer {incomplete[0]!r} is incomplete (has {sorted(grouped[incomplete[0]])}); the file is truncated or "
            "was edited."
        )
    # A fused qkv is split by rows and a swapped adaLN reorders them; both only work on a low-rank pair's `up`.
    needs_low_rank = [
        path
        for path, values in grouped.items()
        if (targets := kohya_mmdit_targets(path)) is not None
        and (len(targets) > 1 or kohya_mmdit_swaps_halves(path, values))
        and not ({"lora_A.weight", "lora_down.weight"} & values.keys())
    ]
    if needs_low_rank:
        return (
            f"layer {needs_low_rank[0]!r} is not low-rank. kohya's fused attention and its swapped modulation layers "
            "are rearranged onto diffusers' modules, which only works for low-rank pairs."
        )
    return None
