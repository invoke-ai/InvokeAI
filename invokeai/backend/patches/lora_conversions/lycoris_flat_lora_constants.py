# SimpleTuner's LyCORIS flat-key LoRA naming (`lycoris_` prefix): key parsing, per-family module names and the
# family detection shared by the probes (`configs/lora.py`) and the conversion.
#
# SimpleTuner's LyCORIS output names each trained module by flattening the model's own diffusers module path with
# underscores behind a `lycoris_` prefix -- `lycoris_single_transformer_blocks_0_attn_to_k.lokr_w1` -- or behind
# `lycoris__` (the wrapper's separator plus the name), as in `lycoris__context_embedder.lokr_w1`. Module names
# contain underscores of their own (`to_k`, `add_k_proj`, `feed_forward.w1`), so a flattened name is resolvable
# only against the whole module list of a family, the way `sd3_lora_constants` already resolves SD 3's spelling.
#
# Family detection rests on names only this family's module list holds (`single_transformer_blocks.N.attn.to_q`
# for FLUX.1, `to_qkv_mlp_proj` for FLUX.2 Klein, the UNet's `down_blocks` for SDXL, `layers.N.attention` for
# Z-Image, `blocks.N.attn1` for Wan), plus, for SDXL, the transformer-block depth that separates it from SD 1/2
# (`configs/lora.py`'s `_state_dict_looks_like_sdxl_unet_lora` documents the same rule), and, for Z-Image, the
# width that separates it from Ideogram 4. A file with no such name claims nothing: the flattening alone is no
# evidence of any base. SD 3's own flat spelling (`sd3_lokr_simpletuner_format`) is left to `sd3_lora_constants`,
# whose module names a family here must not resolve -- checked per name at detection time.
#
# This file imports nothing from the config package, which keeps it free of cycles.

import re
from collections.abc import Mapping
from typing import Any, Literal

import torch

from invokeai.backend.patches.lora_conversions.ideogram4_lora_constants import IDEOGRAM4_WIDTHS, layer_delta_shape
from invokeai.backend.patches.lora_conversions.sd3_lora_constants import resolve_sd3_transformer_module

LycorisFlatFamily = Literal["flux1", "flux2", "sdxl", "z_image", "wan"]

LYCORIS_FLAT_PREFIX = "lycoris_"

# The parameters a layer can carry, as `any_lora_layer_from_state_dict` reads them (plus the DoRA magnitudes it
# dispatches on and the `diff` of LyCORIS's full algorithm). A key whose tail is none of these is not a layer
# parameter and keeps a file out of this layout entirely.
LYCORIS_LORA_PARAM_NAMES = (
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
    "dora_scale",
    "dora_magnitude",
    "diff",
)


def split_lycoris_flat_key(key: str) -> tuple[str, str] | None:
    """``(flattened module name, parameter name)`` for a `lycoris_` key, or ``None``.

    The `lycoris__` spelling (the wrapper's separator in front of the name) drops the second underscore with the
    first: `lycoris__context_embedder` names the module `context_embedder`, like `lycoris_context_embedder` would.
    """
    if not key.startswith(LYCORIS_FLAT_PREFIX):
        return None
    remainder = key[len(LYCORIS_FLAT_PREFIX) :]
    for param_name in LYCORIS_LORA_PARAM_NAMES:
        if remainder.endswith("." + param_name):
            flat_name = remainder[: -len(param_name) - 1]
            return flat_name.removeprefix("_"), param_name
    return None


def group_lycoris_flat_layers(state_dict: Mapping[str | int, Any]) -> dict[str, dict[str, torch.Tensor]] | None:
    """The tensors of a SimpleTuner LyCORIS file grouped by flattened module name, or ``None``.

    ``None`` means the state dict is not this layout: some key is not a `lycoris_` layer parameter. Callers fall
    back to the other formats, so nothing that was identified before this layout was recognized changes.
    """
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            return None
        split = split_lycoris_flat_key(key)
        if split is None:
            return None
        grouped.setdefault(split[0], {})[split[1]] = value
    return grouped or None


# Every module of each family's transformer a LoRA can patch, as diffusers names them, `{i}`/`{j}`/`{k}` standing
# for block indices. The flattened spelling of each template is what the file holds; a name is resolved by matching
# it against the whole template rather than by guessing where the dots were.

FLUX1_TRANSFORMER_MODULE_TEMPLATES = (
    "context_embedder",
    "x_embedder",
    "proj_out",
    "norm_out.linear",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "time_text_embed.text_embedder.linear_1",
    "time_text_embed.text_embedder.linear_2",
    "time_text_embed.guidance_embedder.linear_1",
    "time_text_embed.guidance_embedder.linear_2",
    "transformer_blocks.{i}.attn.to_q",
    "transformer_blocks.{i}.attn.to_k",
    "transformer_blocks.{i}.attn.to_v",
    "transformer_blocks.{i}.attn.to_out.0",
    "transformer_blocks.{i}.attn.add_q_proj",
    "transformer_blocks.{i}.attn.add_k_proj",
    "transformer_blocks.{i}.attn.add_v_proj",
    "transformer_blocks.{i}.attn.to_add_out",
    "transformer_blocks.{i}.ff.net.0.proj",
    "transformer_blocks.{i}.ff.net.2",
    "transformer_blocks.{i}.ff_context.net.0.proj",
    "transformer_blocks.{i}.ff_context.net.2",
    "transformer_blocks.{i}.norm1.linear",
    "transformer_blocks.{i}.norm1_context.linear",
    "single_transformer_blocks.{i}.attn.to_q",
    "single_transformer_blocks.{i}.attn.to_k",
    "single_transformer_blocks.{i}.attn.to_v",
    "single_transformer_blocks.{i}.proj_mlp",
    "single_transformer_blocks.{i}.proj_out",
    "single_transformer_blocks.{i}.norm.linear",
)

FLUX2_TRANSFORMER_MODULE_TEMPLATES = (
    "transformer_blocks.{i}.attn.to_q",
    "transformer_blocks.{i}.attn.to_k",
    "transformer_blocks.{i}.attn.to_v",
    "transformer_blocks.{i}.attn.to_out.0",
    "transformer_blocks.{i}.attn.add_q_proj",
    "transformer_blocks.{i}.attn.add_k_proj",
    "transformer_blocks.{i}.attn.add_v_proj",
    "transformer_blocks.{i}.attn.to_add_out",
    "transformer_blocks.{i}.ff.linear_in",
    "transformer_blocks.{i}.ff.linear_out",
    "transformer_blocks.{i}.ff_context.linear_in",
    "transformer_blocks.{i}.ff_context.linear_out",
    "single_transformer_blocks.{i}.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.{i}.attn.to_out",
)

# Z-Image's S3-DiT keeps its blocks under `layers`, `context_refiner` and `noise_refiner` (`z_image_lora_conversion_utils`).
_Z_IMAGE_BLOCK_GROUPS = ("layers", "context_refiner", "noise_refiner")
Z_IMAGE_TRANSFORMER_MODULE_TEMPLATES = tuple(
    f"{group}.{{i}}.{tail}"
    for group in _Z_IMAGE_BLOCK_GROUPS
    for tail in (
        "attention.to_q",
        "attention.to_k",
        "attention.to_v",
        "attention.to_out.0",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "adaLN_modulation.0",
    )
)

WAN_TRANSFORMER_MODULE_TEMPLATES = (
    "blocks.{i}.attn1.to_q",
    "blocks.{i}.attn1.to_k",
    "blocks.{i}.attn1.to_v",
    "blocks.{i}.attn1.to_out.0",
    "blocks.{i}.attn1.norm_q",
    "blocks.{i}.attn1.norm_k",
    "blocks.{i}.attn2.to_q",
    "blocks.{i}.attn2.to_k",
    "blocks.{i}.attn2.to_v",
    "blocks.{i}.attn2.to_out.0",
    "blocks.{i}.attn2.norm_q",
    "blocks.{i}.attn2.norm_k",
    "blocks.{i}.ffn.net.0.proj",
    "blocks.{i}.ffn.net.2",
)

# The SDXL UNet's module names. The first down block and the last up block carry no attention (`DownBlock2D`,
# `UpBlock2D` in `stabilityai/stable-diffusion-xl-base-1.0`'s unet config), which is why the `attentions` templates
# sit on blocks `{i}` = 1, 2 only by data, not by rule: the indices are free and depth is what identifies the family.
_SDXL_ATTENTION_TAILS = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "norm1",
    "norm2",
)
_SDXL_RESNET_TAILS = ("norm1", "conv1", "norm2", "conv2", "time_emb_proj", "conv_shortcut")
SDXL_UNET_MODULE_TEMPLATES = (
    tuple(
        f"{stage}_blocks.{{i}}.attentions.{{j}}.transformer_blocks.{{k}}.{tail}"
        for stage in ("down", "up")
        for tail in _SDXL_ATTENTION_TAILS
    )
    + tuple(f"mid_block.attentions.{{j}}.transformer_blocks.{{k}}.{tail}" for tail in _SDXL_ATTENTION_TAILS)
    + tuple(f"{stage}_blocks.{{i}}.resnets.{{j}}.{tail}" for stage in ("down", "up") for tail in _SDXL_RESNET_TAILS)
    + tuple(f"mid_block.resnets.{{j}}.{tail}" for tail in _SDXL_RESNET_TAILS)
    + (
        "down_blocks.{i}.downsamplers.{j}.conv",
        "up_blocks.{i}.upsamplers.{j}.conv",
        "up_blocks.{i}.upsamplers.{j}.conv_shortcut",
        "conv_in",
        "conv_norm_out",
        "conv_out",
        "time_embedding.linear_{i}",
        "add_embedding.linear_{i}",
    )
)

_FAMILY_MODULE_TEMPLATES: dict[LycorisFlatFamily, tuple[str, ...]] = {
    "flux1": FLUX1_TRANSFORMER_MODULE_TEMPLATES,
    "flux2": FLUX2_TRANSFORMER_MODULE_TEMPLATES,
    "sdxl": SDXL_UNET_MODULE_TEMPLATES,
    "z_image": Z_IMAGE_TRANSFORMER_MODULE_TEMPLATES,
    "wan": WAN_TRANSFORMER_MODULE_TEMPLATES,
}

_INDEX_PLACEHOLDERS = ("{i}", "{j}", "{k}")


def _template_regexes(templates: tuple[str, ...]) -> dict[str, re.Pattern[str]]:
    """One regex per template, matching the flattened spelling with the indices captured in order."""
    compiled: dict[str, re.Pattern[str]] = {}
    for template in templates:
        flat = template.replace(".", "_")
        pattern = re.escape(flat)
        for placeholder in _INDEX_PLACEHOLDERS:
            pattern = pattern.replace(re.escape(placeholder), r"(\d+)")
        compiled[template] = re.compile(f"^{pattern}$")
    return compiled


_FLAT_TEMPLATE_RE: dict[LycorisFlatFamily, dict[str, re.Pattern[str]]] = {
    family: _template_regexes(templates) for family, templates in _FAMILY_MODULE_TEMPLATES.items()
}

# A transformer block past the single stack of SD 1.x/SD 2.x UNet attentions: the depth that identifies SDXL, the
# same rule `_state_dict_looks_like_sdxl_unet_lora` applies to kohya's `lora_unet_` spelling.
_SDXL_DEEP_ATTENTION_RE = re.compile(
    r"^(?:(?:down|up)_blocks_\d+_attentions_\d+|mid_block_attentions_\d+)_transformer_blocks_(\d+)_"
)


def resolve_lycoris_flat_module(family: LycorisFlatFamily, flat_name: str) -> str | None:
    """The diffusers module path a flattened name names in ``family``, or ``None``."""
    for template, pattern in _FLAT_TEMPLATE_RE[family].items():
        if match := pattern.match(flat_name):
            resolved = template
            for group in match.groups():
                for placeholder in _INDEX_PLACEHOLDERS:
                    if placeholder in resolved:
                        resolved = resolved.replace(placeholder, group, 1)
                        break
            return resolved
    return None


def _flattened_name_resolves_only_in(family: LycorisFlatFamily, flat_name: str) -> bool:
    """True for names only ``family``'s module list holds — the evidence a file is of that family.

    SD 3's flat spelling counts as a neighbour here: FLUX.1's double blocks share SD 3's module names, so those
    names identify neither (`sd3_lokr_simpletuner_format` is an SD 3 file holding exactly them).
    """
    if resolve_lycoris_flat_module(family, flat_name) is None:
        return False
    if resolve_sd3_transformer_module(LYCORIS_FLAT_PREFIX + flat_name) is not None:
        return False
    return all(
        other is family or resolve_lycoris_flat_module(other, flat_name) is None for other in _FAMILY_MODULE_TEMPLATES
    )


def _candidate_family(layers: dict[str, dict[str, torch.Tensor]]) -> LycorisFlatFamily | None:
    matches: list[LycorisFlatFamily] = [
        family
        for family in _FAMILY_MODULE_TEMPLATES
        if any(_flattened_name_resolves_only_in(family, flat_name) for flat_name in layers)
    ]
    if len(matches) != 1:
        return None
    family = matches[0]

    if family == "sdxl" and not any(_SDXL_DEEP_ATTENTION_RE.match(flat_name + "_") for flat_name in layers):
        # A LoRA confined to the shallow attention blocks is indistinguishable from SD 1/2 by structure alone
        # (the limitation `_state_dict_looks_like_sdxl_unet_lora` documents).
        return None
    if family == "z_image" and any(
        (shape := layer_delta_shape(values)) is not None
        and (shape[0] in IDEOGRAM4_WIDTHS or shape[1] in IDEOGRAM4_WIDTHS)
        for values in layers.values()
    ):
        # Ideogram 4 names its blocks exactly like Z-Image; its widths (4608) are what tell the two apart, the
        # same check `LoRA_LyCORIS_ZImage_Config` applies to the other spellings.
        return None
    return family


def lycoris_flat_candidate_family(state_dict: Mapping[str | int, Any]) -> LycorisFlatFamily | None:
    """The family a state dict's exclusive names point to, whatever else it holds.

    The conversion resolves every name against this family and names any it cannot place, so a refused file says
    which layer is the problem. The probes use `lycoris_flat_family`, which additionally requires every name to
    resolve: a file with a key of no family is not identified at all.
    """
    layers = group_lycoris_flat_layers(state_dict)
    return None if layers is None else _candidate_family(layers)


def lycoris_flat_family(state_dict: Mapping[str | int, Any]) -> LycorisFlatFamily | None:
    """Which family's transformer a SimpleTuner LyCORIS flat-key state dict was trained for, or ``None``.

    ``None`` covers everything this layout does not settle: state dicts of other layouts, files whose names no
    family holds, files with a name of no family, and files whose names several families share without one
    only-this-family name to break the tie.
    """
    layers = group_lycoris_flat_layers(state_dict)
    if layers is None:
        return None
    family = _candidate_family(layers)
    if family is None or not all(resolve_lycoris_flat_module(family, flat_name) is not None for flat_name in layers):
        return None
    return family
