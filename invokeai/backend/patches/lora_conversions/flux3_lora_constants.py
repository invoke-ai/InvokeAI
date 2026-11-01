"""Which keys a FLUX.3 Image LoRA may hold, shared by its identification and its conversion.

Published FLUX.3 LoRAs come in these key layouts, all onto the same transformer:

- diffusers/PEFT: `transformer.` + diffusers' module path + `lora_A`/`lora_B` (diffusers' trainers, SimpleTuner), with
  the PEFT config in the header's `lora_adapter_metadata`;
- Black Forest Labs' native module names under `diffusion_model.` (`single_blocks.N.q_proj`,
  `content_mode_blocks.<mode>.N.mlp_in`, `emb_in.image`, `final_layer.image.adaLN_modulation.1`), with PEFT's
  `lora_A`/`lora_B` (ai-toolkit) or `lora_down`/`lora_up` and a per-layer `alpha` (ComfyUI conversions);
- diffusers' module paths without a prefix, as diffusers' `save_lora_adapter` saves the transformer's own adapter
  (its PEFT config unprefixed in the header) and PEFT a model's state dict: with its adapter name in each key
  (`lora_A.default.weight`), which `normalize_peft_adapter_names` strips before anything here reads the keys, or under
  `base_model.model.`, as a PEFT model's `save_pretrained` writes them.

Native names reach diffusers' module paths through the rename the base checkpoint takes
(`flux3_native_key_to_diffusers`), so a LoRA lands on exactly the modules the native weights land on; a second table
would drift from it the first time either side gains a layer. One native module stores its output differently: BFL's
`final_layer.image.adaLN_modulation.1` writes [shift, scale] rows where diffusers' `norm_out.linear` reads [scale,
shift], so a native layer there is applied with its halves swapped. A layer that names `norm_out.linear` itself is
already in diffusers' order.

Every tensor of a file is a LoRA parameter or a reason to refuse the file, never silently left out: anything else
could be meant for the model -- a full weight, a norm-scale delta such as a distillation extract's `delta.*` tensors, a
`set_weight` -- and applying the rest would make a partial LoRA that passes for a whole one. One exception is known
and skipped with a warning: the bank of per-step output heads PDD few-step LoRAs carry (`_is_pdd_head_bank`), which
PDD's own sampler fuses per jump in place of `proj_out`. Without the heads such a LoRA still runs, as a plain few-step
LoRA.

Identification also holds every layer to the shape of the weight it updates in the released transformer, which Klein
and Klein Base share (`misshapen_flux3_layers`). The conversion maps names only: it applies to whichever transformer it
is given.

The module imports no patch layers: the model identification imports it, and the layers import the model manager's
loaders, which import the identification.
"""

import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from invokeai.backend.flux3.conversion import flux3_native_key_to_diffusers

FLUX3_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# The prefixes the producers above write, longest first, so `base_model.model.` is not read as part of a module name.
_PREFIXES = ("base_model.model.", "diffusion_model.", "transformer.")

# Each recognized parameter name and the value key the layer classes read. DoRA magnitudes come in two orientations
# (see `DoRALayer`): LyCORIS's `dora_scale` indexes the input dim, PEFT's and ai-toolkit's the output dim.
_PARAM_VALUE_KEYS = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
    "lora_down.weight": "lora_down.weight",
    "lora_up.weight": "lora_up.weight",
    "alpha": "alpha",
    "dora_scale": "dora_scale",
    # A PEFT model's own state dict, its adapter name stripped.
    "lora_magnitude_vector.weight": "dora_magnitude",
    # What PEFT's `get_peft_model_state_dict` saves: it drops the `.weight`.
    "lora_magnitude_vector": "dora_magnitude",
    # ai-toolkit's DoRA.
    "magnitude": "dora_magnitude",
    "diff": "diff",
    **{name: name for name in ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_t2")},
    **{name: name for name in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2")},
}

# The released transformer (`transformer/config.json` of black-forest-labs/flux-3-image-klein and -klein-base): 24
# heads of 128 make the width, `mlp_ratio` 3 the feed-forward width (its gated input projects to twice that), 128
# latent channels in and out, a 20480-wide caption embedding, 256 timestep channels, 5 blocks per mode and 28 joint
# blocks.
_HIDDEN = 3072
_MLP = 3 * _HIDDEN
_LATENT_CHANNELS = 128
_CONTEXT = 20480
_TIMESTEP_CHANNELS = 256

# Every Linear of the transformer in diffusers' names, with its weight's (out_features, in_features).
_BLOCK_LINEARS = {
    "attn.to_q": (_HIDDEN, _HIDDEN),
    "attn.to_k": (_HIDDEN, _HIDDEN),
    "attn.to_v": (_HIDDEN, _HIDDEN),
    "attn.to_out.0": (_HIDDEN, _HIDDEN),
    "ff.linear_in": (2 * _MLP, _HIDDEN),
    "ff.linear_out": (_HIDDEN, _MLP),
}
_STACK_DEPTHS = {
    "mode_blocks.image": 5,
    "mode_blocks.image_cond": 5,
    "txt_mode_blocks": 5,
    "single_transformer_blocks": 28,
}
_TOP_LINEARS = {
    "x_embedders.image": (_HIDDEN, _LATENT_CHANNELS),
    "x_embedders.image_cond": (_HIDDEN, _LATENT_CHANNELS),
    "context_embedder": (_HIDDEN, _CONTEXT),
    "timestep_embedder.linear_1": (_HIDDEN, _TIMESTEP_CHANNELS),
    "timestep_embedder.linear_2": (_HIDDEN, _HIDDEN),
    # Klein's alone: the embedder of the timestep each step jumps to. Klein Base has no such module, and the patcher
    # skips a layer on it there, with a warning.
    "target_timestep_embedder.linear_1": (_HIDDEN, _TIMESTEP_CHANNELS),
    "target_timestep_embedder.linear_2": (_HIDDEN, _HIDDEN),
    **{
        f"{stage}.{mode}.linear": (3 * _HIDDEN, _HIDDEN)
        for stage in ("early_stream_modulations", "single_stream_modulations")
        for mode in ("image", "image_cond", "txt")
    },
    "norm_out.linear": (2 * _HIDDEN, _HIDDEN),
    "proj_out": (_LATENT_CHANNELS, _HIDDEN),
}
# A block index as the transformer spells it, without leading zeros: `01` would be block 1 under a second name.
_BLOCK = re.compile(rf"({'|'.join(re.escape(stack) for stack in _STACK_DEPTHS)})\.(0|[1-9]\d*)\.(.+)")
# BFL's output modulation, whose rows diffusers' `norm_out.linear` holds in swapped halves.
_NATIVE_OUTPUT_MODULATION = "final_layer.image.adaLN_modulation.1"

# Module names only FLUX.3 has, BFL's and diffusers', compared as whole key segments: FLUX.1 and FLUX.2 name theirs
# `x_embedder` and `single_stream_modulation`, so a substring test would claim them.
_FLUX3_SEGMENTS = frozenset(
    {
        "content_mode_blocks",
        "mode_blocks",
        "txt_mode_blocks",
        "early_stream_modulations",
        "single_stream_modulations",
        "x_embedders",
        "emb_in",
        "ref_time_in",
        "target_timestep_embedder",
    }
)
# FLUX.3's single blocks project q, k, v and the MLP separately; FLUX.1's and FLUX.2's native ones fuse them in
# `linear1`/`linear2`. In diffusers' names only their gated MLP is FLUX.3's own: FLUX.1's and FLUX.2's single blocks
# have no `ff`, AuraFlow's an `ff.linear_1`, and AuraFlow's attention is named like FLUX.3's (`attn.to_out.0`).
_FLUX3_NATIVE_SINGLE_LEAVES = frozenset({"q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out"})
_FLUX3_DIFFUSERS_SINGLE_LEAVES = (("ff", "linear_in"), ("ff", "linear_out"))


def _layer_value_sets() -> frozenset[frozenset[str]]:
    """Every set of value keys that builds one layer reading all of them: nothing missing, nothing it would ignore.

    The kinds `any_lora_layer_from_state_dict` builds: low-rank, DoRA with either magnitude, LoKR with each operand
    whole or factored, LoHA, and a full diff, which has no rank for an alpha to scale. LyCORIS writes Tucker cores
    (`lokr_t2`, `hada_t1`/`hada_t2`) only for convolution kernels, and this transformer is all Linears, so a layer
    with one is refused. No Linear here has a bias, so no bias key is recognized.
    """
    low_rank = frozenset({"lora_down.weight", "lora_up.weight"})
    lokr_w1 = (frozenset({"lokr_w1"}), frozenset({"lokr_w1_a", "lokr_w1_b"}))
    lokr_w2 = (frozenset({"lokr_w2"}), frozenset({"lokr_w2_a", "lokr_w2_b"}))
    ranked = [
        low_rank,
        low_rank | {"dora_scale"},
        low_rank | {"dora_magnitude"},
        *(w1 | w2 for w1, w2 in itertools.product(lokr_w1, lokr_w2)),
        frozenset({"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}),
    ]
    return frozenset([*ranked, *(layer | {"alpha"} for layer in ranked), frozenset({"diff"})])


_LAYERS = _layer_value_sets()


def _weight_shape(module: str) -> tuple[int, int] | None:
    """The released (out_features, in_features) of a transformer Linear by its diffusers path, None for no Linear."""
    if block := _BLOCK.fullmatch(module):
        stack, index, leaf = block.groups()
        return _BLOCK_LINEARS.get(leaf) if int(index) < _STACK_DEPTHS[stack] else None
    return _TOP_LINEARS.get(module)


def _resolve_module(name: str) -> tuple[str, bool] | None:
    """`(diffusers module path, halves swapped)` for a LoRA's module name, prefix stripped; None for no Linear."""
    if _weight_shape(name) is not None:
        return name, False
    module = flux3_native_key_to_diffusers(f"{name}.weight")[: -len(".weight")]
    if module != name and _weight_shape(module) is not None:
        return module, name == _NATIVE_OUTPUT_MODULATION
    return None


def _split_param(key: str) -> tuple[str, str] | None:
    """`(module name, value key)` of a key with a recognized LoRA parameter name, the prefix stripped."""
    for param, value_key in _PARAM_VALUE_KEYS.items():
        if key.endswith("." + param):
            name = key[: -len(param) - 1]
            for prefix in _PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            return name, value_key
    return None


def _names_a_flux3_module(key: str) -> bool:
    segments = key.split(".")
    if not _FLUX3_SEGMENTS.isdisjoint(segments):
        return True
    for i, segment in enumerate(segments[:-2]):
        if not segments[i + 1].isdigit():
            continue
        leaf = tuple(segments[i + 2 :])
        if segment == "single_blocks" and leaf[0] in _FLUX3_NATIVE_SINGLE_LEAVES:
            return True
        if segment == "single_transformer_blocks" and any(
            leaf[: len(marker)] == marker for marker in _FLUX3_DIFFUSERS_SINGLE_LEAVES
        ):
            return True
    return False


def has_flux3_lora_keys(keys: Iterable[str | int]) -> bool:
    """Whether a state dict is a FLUX.3 LoRA's: a key names a module only FLUX.3's transformer has, and a key is a LoRA
    parameter. The second keeps a FLUX.3 checkpoint, whose weights carry the same module names, out.

    Says which family a file belongs to, not whether it applies: `read_flux3_lora_keys` decides that.
    """
    str_keys = [key for key in keys if isinstance(key, str)]
    return any(_names_a_flux3_module(key) for key in str_keys) and any(_split_param(key) for key in str_keys)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _is_pdd_head_bank(key: str | int, value: Any) -> bool:
    """Whether a tensor is a PDD few-step LoRA's bank of per-step output heads: one `proj_out` weight per step of PDD's
    trajectory, (steps, 128, 3072). NoiseToNova's PEFT export stores it as a root `proj_out.weight`, its ComfyUI
    export under `pdd.`. No parameter of the transformer has three dimensions."""
    if not isinstance(key, str) or (key != "proj_out.weight" and not key.startswith("pdd.")):
        return False
    shape = _shape(value)
    return len(shape) == 3 and shape[1:] == _TOP_LINEARS["proj_out"]


# Why a key keeps a file from applying in full. Shown in the install queue, so plain text.
NOT_A_LORA_PARAMETER = "are not LoRA parameters InvokeAI applies (a full weight, for instance)"
DISTILLATION_EXTRACT = (
    "are a distillation extract's norm-scale deltas or weights to set (delta.*, set_weight), which no LoRA carries"
)
NO_FLUX3_LAYER = "name no Linear layer of FLUX.3's transformer (another model's, or a text encoder's)"
NAMED_TWICE = "name one layer more than once (under two prefixes, or under both BFL's and diffusers' names)"
NOT_ONE_LAYER = "do not make exactly one complete layer (half a pair, a lone alpha or magnitude, two kinds at once)"
WRONG_SHAPE = "do not update a weight of the shape the FLUX.3 layer they name has"
# What to install instead, after the reasons, for a file refused for one of these.
_NEXT_STEPS = {DISTILLATION_EXTRACT: "To sample with distilled weights, install FLUX.3 Klein itself."}


def _refusal_without_layer(key: str | int) -> str:
    if isinstance(key, str) and (key.startswith("delta.") or key.endswith(".set_weight")):
        return DISTILLATION_EXTRACT
    return NOT_A_LORA_PARAMETER


@dataclass(frozen=True)
class Flux3LoRAKeys:
    """A FLUX.3 LoRA's keys sorted out (`read_flux3_lora_keys`).

    `layers` maps each diffusers module path to its layer's source key per value key; `swapped` holds the paths whose
    layer is in BFL's [shift, scale] row order; `pdd_heads` the keys of a PDD head bank; `refused` each key that keeps
    the file from applying in full, with why.
    """

    layers: Mapping[str, Mapping[str, str]]
    swapped: frozenset[str]
    pdd_heads: tuple[str, ...]
    refused: Mapping[str, str]


def read_flux3_lora_keys(state_dict: Mapping[str | int, Any]) -> Flux3LoRAKeys:
    """Sort a FLUX.3 LoRA's keys into layers, a PDD head bank and the keys that keep it from applying in full.

    A key is refused when it is no LoRA parameter (see the module docstring), names a module this transformer has no
    Linear for, names its layer as another key does under another prefix or name, or belongs to a layer that is not
    exactly one complete layer (`_layer_value_sets`). Shapes are `misshapen_flux3_layers`' to check.
    """
    refused: dict[str, str] = {}
    pdd_heads: list[str] = []
    modules: dict[str, dict[str, list[str]]] = {}
    # The names each module is reached by. One module under two names (BFL's and diffusers') is refused even when the
    # two give different values: the output modulation's two names order its rows differently.
    names: dict[str, set[str]] = {}
    swapped: set[str] = set()
    for key, value in state_dict.items():
        split = _split_param(key) if isinstance(key, str) else None
        if split is None:
            if _is_pdd_head_bank(key, value):
                pdd_heads.append(str(key))
            else:
                refused[str(key)] = _refusal_without_layer(key)
            continue
        name, value_key = split
        resolved = _resolve_module(name)
        if resolved is None:
            refused[str(key)] = NO_FLUX3_LAYER
            continue
        module, is_swapped = resolved
        modules.setdefault(module, {}).setdefault(value_key, []).append(str(key))
        names.setdefault(module, set()).add(name)
        if is_swapped:
            swapped.add(module)

    layers: dict[str, dict[str, str]] = {}
    for module, values in modules.items():
        keys = [key for spellings in values.values() for key in spellings]
        if len(names[module]) > 1 or any(len(spellings) > 1 for spellings in values.values()):
            refused.update(dict.fromkeys(keys, NAMED_TWICE))
        elif frozenset(values) not in _LAYERS:
            refused.update(dict.fromkeys(keys, NOT_ONE_LAYER))
        else:
            layers[module] = {value_key: spellings[0] for value_key, spellings in values.items()}
    return Flux3LoRAKeys(
        layers=layers, swapped=frozenset(swapped & layers.keys()), pdd_heads=tuple(pdd_heads), refused=refused
    )


def _product(a: Any, b: Any) -> tuple[int, ...] | None:
    """The shape of `a @ b`, or None unless both are matrices whose inner sizes agree."""
    a_shape, b_shape = _shape(a), _shape(b)
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[1] != b_shape[0]:
        return None
    return a_shape[0], b_shape[1]


def _update_shape(layer: Mapping[str, Any]) -> tuple[int, ...] | None:
    """The shape of the weight update one layer's values make, or None if they make none.

    A full diff is the update itself; a low-rank (or DoRA) layer's is up @ down; LoHA's the elementwise product of
    two low-rank products, which must agree; LoKR's the Kronecker product of two matrices, each given whole or as a
    low-rank product.
    """
    if "diff" in layer:
        return _shape(layer["diff"])
    if "lora_up.weight" in layer:
        return _product(layer["lora_up.weight"], layer["lora_down.weight"])
    if "hada_w1_a" in layer:
        w1 = _product(layer["hada_w1_a"], layer["hada_w1_b"])
        return w1 if w1 == _product(layer["hada_w2_a"], layer["hada_w2_b"]) else None
    w1 = _shape(layer["lokr_w1"]) if "lokr_w1" in layer else _product(layer["lokr_w1_a"], layer["lokr_w1_b"])
    w2 = _shape(layer["lokr_w2"]) if "lokr_w2" in layer else _product(layer["lokr_w2_a"], layer["lokr_w2_b"])
    if w1 is None or w2 is None or len(w1) != 2 or len(w2) != 2:
        return None
    return w1[0] * w2[0], w1[1] * w2[1]


def _magnitude_fits(layer: Mapping[str, Any], out_features: int, in_features: int) -> bool:
    """Whether a DoRA layer's magnitude, if it has one, has an entry per row or column of its weight in the
    orientation `DoRALayer` reads it in: PEFT's and ai-toolkit's one per output row, `(out,)` or `(out, 1)`;
    LyCORIS's `dora_scale` one per output row as `(out, 1)`, otherwise one per input column, `(1, in)` or `(in,)`."""
    if "dora_magnitude" in layer:
        shape = _shape(layer["dora_magnitude"])
        return len(shape) > 0 and shape[0] == out_features and all(size == 1 for size in shape[1:])
    if "dora_scale" in layer:
        return _shape(layer["dora_scale"]) in {(out_features, 1), (1, in_features), (in_features,)}
    return True


def misshapen_flux3_layers(state_dict: Mapping[str | int, Any], keys: Flux3LoRAKeys) -> dict[str, str]:
    """Each key of a layer whose update does not have the shape of the weight it changes in the released transformer,
    or whose DoRA magnitude does not fit that weight, with why (`WRONG_SHAPE`)."""
    misshapen: dict[str, str] = {}
    for module, sources in keys.layers.items():
        weight_shape = _weight_shape(module)
        assert weight_shape is not None  # `read_flux3_lora_keys` resolved it to a Linear
        values = {value_key: state_dict[key] for value_key, key in sources.items()}
        if _update_shape(values) != weight_shape or not _magnitude_fits(values, *weight_shape):
            misshapen.update(dict.fromkeys(sources.values(), WRONG_SHAPE))
    return misshapen


def describe_refusal(refused: Mapping[str, str]) -> str:
    """Why a FLUX.3 LoRA does not apply in full: each reason with its count of keys and one of them, then what to
    install instead where a reason has an answer."""
    by_reason: dict[str, list[str]] = {}
    for key, reason in refused.items():
        by_reason.setdefault(reason, []).append(key)
    parts = [f"{len(keys)} key(s) {reason}, e.g. '{sorted(keys)[0]}'" for reason, keys in by_reason.items()]
    next_steps = [_NEXT_STEPS[reason] for reason in by_reason if reason in _NEXT_STEPS]
    return " ".join(["This FLUX.3 LoRA does not apply in full: " + "; ".join(parts) + ".", *next_steps])


LORA_ADAPTER_METADATA_KEY = "lora_adapter_metadata"
"""The header entry diffusers saves PEFT `LoraConfig`s in: a pipeline's save keys each component's fields
`<component>.<field>`, a model's `save_lora_adapter` writes the model's own unprefixed."""


def peft_scale(metadata: Mapping[str, str] | None) -> float | None:
    """The scale PEFT gives every low-rank layer of the transformer LoRA a diffusers header describes, or None for a
    header that describes none.

    The transformer's config is the header's `transformer.` fields, as a pipeline's save writes them and
    `load_lora_adapter` reads them; a header without any holds it unprefixed, as the transformer's own
    `save_lora_adapter` writes it and `load_lora_adapter(prefix=None)` reads it. A header with both is read by its
    `transformer.` fields, a pipeline's. diffusers rebuilds the config from those (`_create_lora_config`, PEFT's
    defaults for the fields it lacks: r 8, lora_alpha 8), and PEFT scales each layer by `lora_alpha / r`, or
    `lora_alpha / sqrt(r)` with rsLoRA. Raises ValueError for a header this cannot apply as PEFT would: not a JSON
    object, an adapter other than a LoRA (how PEFT scales a LoKr or LoHa layer is not checked against InvokeAI's
    layers), a rank or alpha set per module (`rank_pattern`, `alpha_pattern`), or no usable rank.
    """
    raw = (metadata or {}).get(LORA_ADAPTER_METADATA_KEY)
    if not raw:
        return None
    try:
        fields = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"This FLUX.3 LoRA's lora_adapter_metadata header is not JSON ({e}).") from None
    if not isinstance(fields, dict):
        raise ValueError("This FLUX.3 LoRA's lora_adapter_metadata header is not a JSON object of PEFT settings.")
    prefixed = {
        key.removeprefix("transformer."): value for key, value in fields.items() if key.startswith("transformer.")
    }
    # Another component's fields (`text_encoder.r`) are not the transformer's.
    config = prefixed or {key: value for key, value in fields.items() if "." not in key}
    if not config:
        return None
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError(
            f"This FLUX.3 LoRA's lora_adapter_metadata header describes a PEFT {config['peft_type']} adapter; "
            "InvokeAI applies PEFT LoRA headers only."
        )
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(
            "This FLUX.3 LoRA sets its rank or alpha per module (rank_pattern or alpha_pattern in its "
            "lora_adapter_metadata header), which InvokeAI does not apply."
        )
    rank, alpha = config.get("r", 8), config.get("lora_alpha", 8)
    if not isinstance(rank, int | float) or not isinstance(alpha, int | float) or rank <= 0:
        raise ValueError("This FLUX.3 LoRA's lora_adapter_metadata header gives no usable r and lora_alpha.")
    return alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank
