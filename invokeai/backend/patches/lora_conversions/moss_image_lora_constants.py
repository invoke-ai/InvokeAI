"""Which keys a MOSS-Image LoRA may hold, and the shapes their layers must have, shared by its identification and its
conversion.

MOSS-Image's modules are named the same in the checkpoint and at runtime, so a LoRA's key is a container prefix, the
module path and a parameter name. The prefixes the published files use:

- `diffusion_model.` (ai-toolkit, ComfyUI) and `transformer.` (diffusers' `save_lora_weights`), or none (a bare PEFT
  file, DiffSynth), with `lora_A`/`lora_B` or `lora_down`/`lora_up`, a per-layer `alpha` or none;
- `lora_unet_` and the module path with its dots flattened to underscores (kohya, LyCORIS). The flattening is lossy
  (`ffn_up_gate` has underscores of its own), so a flattened name is mapped through the table of modules, not re-dotted.

Raw PEFT's `base_model.model.` prefix is recognized, so such a file is told from other families' (Z-Image's probe
admits it too), but refused: its alpha lives in an `adapter_config.json` next to the file, not in it. Install the
ComfyUI or diffusers export instead.
Bare `lora_A`/`lora_B` keys without `.weight` are MOSS-Image's edit LoRAs, which condition on reference images and are
not applied here.

ComfyUI's export and ai-toolkit fuse each block's `proj.q`, `proj.k` and `proj.v` into `layers.N.proj.qkv` (rows q, k,
v in that order), which the transformer here holds split: its layers are cut into thirds on conversion. The text
refiner's `text_embedder.blocks.N.qkv` is fused at runtime too and stays whole.

A file is held to what the patcher applies in full: a layer it cannot apply or that names no module of the
transformer is refused at identification instead of being skipped with a log line when a generation runs. Norm
weights are `_RMSNorm`s, which a LoRA sidecar cannot wrap, so a file that patches them is refused as well, whatever the
transformer's format.

The module imports no patch layers: the model identification imports it, and the layers import the model manager's
loaders, which import the identification.
"""

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from invokeai.backend.patches.lora_conversions.dora_magnitude_keys import DORA_MAGNITUDE_SUFFIX_TO_VALUE_KEY
from invokeai.backend.patches.lora_conversions.flux3_lora_constants import lora_layer_value_sets, update_shape

MOSS_IMAGE_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# Longest first, so `base_model.model.transformer.` wins over `base_model.model.`.
_PREFIXES = ("base_model.model.transformer.", "base_model.model.", "diffusion_model.", "transformer.")
_RAW_PEFT_PREFIX = "base_model.model."
_KOHYA_PREFIX = "lora_unet_"

# Each recognized parameter name and the value key the layer classes read. DoRA magnitudes come in two orientations
# (see `DoRALayer`): LyCORIS's `dora_scale` may index either axis, PEFT's and ai-toolkit's the output dim.
_PARAM_VALUE_KEYS = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
    "lora_down.weight": "lora_down.weight",
    "lora_up.weight": "lora_up.weight",
    "alpha": "alpha",
    "diff": "diff",
    **{suffix.removeprefix("."): value_key for suffix, value_key in DORA_MAGNITUDE_SUFFIX_TO_VALUE_KEY.items()},
    **{name: name for name in ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b")},
    **{name: name for name in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")},
}

# The released transformer's dimensions: `MOSSImageTransformer`'s constructor defaults (`pipeline_moss_image.py`),
# which are the `transformer/config.json` of every released checkpoint. 24 heads divide the 3072-wide hidden state, the
# feed-forward is 8192 wide (its gated input projects to twice that), the text encoder is 4096 wide, the timestep
# embedding takes 256 frequencies, the latent has 128 channels, and there are 42 blocks and 2 text-refiner blocks.
_DIM = 3072
_HEAD_DIM = _DIM // 24
_FFN = 8192
_TEXT_DIM = 4096
_FREQ_DIM = 256
_LATENT_CHANNELS = 128
_NUM_BLOCKS = 42
_NUM_TEXT_BLOCKS = 2

# Each Linear and RMSNorm weight's shape: (out_features, in_features) and (width,). Block modules are relative to their
# stack's `<stack>.<index>.`.
_NORMS = {"norm1": (_DIM,), "norm2": (_DIM,), "q_norm": (_HEAD_DIM,), "k_norm": (_HEAD_DIM,)}
_BLOCK_LINEARS = {
    "proj.q": (_DIM, _DIM),
    "proj.k": (_DIM, _DIM),
    "proj.v": (_DIM, _DIM),
    "proj.o": (_DIM, _DIM),
    "proj.qkv": (3 * _DIM, _DIM),
    "proj.ffn_up_gate": (2 * _FFN, _DIM),
    "proj.ffn_down": (_DIM, _FFN),
}
_TEXT_BLOCK_LINEARS = {
    "qkv": (3 * _DIM, _DIM),
    "proj": (_DIM, _DIM),
    "ffn_gate": (_FFN, _DIM),
    "ffn_up": (_FFN, _DIM),
    "ffn_down": (_DIM, _FFN),
}
_STACKS = {
    "layers": (_NUM_BLOCKS, _BLOCK_LINEARS, _NORMS),
    "text_embedder.blocks": (_NUM_TEXT_BLOCKS, _TEXT_BLOCK_LINEARS, _NORMS),
}
_TOP_LINEARS = {
    "video_embedder": (_DIM, _LATENT_CHANNELS),
    "text_embedder.proj": (_DIM, _TEXT_DIM),
    "time_embedding.0": (_DIM, _FREQ_DIM),
    "time_embedding.2": (_DIM, _DIM),
    "shared_block_modulation": (6 * _DIM, _DIM),
    "video_head_modulation": (2 * _DIM, _DIM),
    "final_linear_video": (_LATENT_CHANNELS, _DIM),
}
_TOP_NORMS = {"final_norm_video": (_DIM,)}

# A block index as the transformer spells it, without leading zeros: `01` would be block 1 under a second name.
_BLOCK = re.compile(r"(layers|text_embedder\.blocks)\.(0|[1-9]\d*)\.(.+)")
_KOHYA_BLOCK = re.compile(r"(layers|text_embedder_blocks)_(0|[1-9]\d*)_(.+)")
_KOHYA_STACKS = {"layers": "layers", "text_embedder_blocks": "text_embedder.blocks"}
_KOHYA_LEAVES = {
    stack: {leaf.replace(".", "_"): leaf for leaf in (*linears, *norms)}
    for stack, (_, linears, norms) in _STACKS.items()
}
_KOHYA_TOP = {module.replace(".", "_"): module for module in (*_TOP_LINEARS, *_TOP_NORMS)}


def _module_path(name: str) -> str | None:
    """The module path a key's module part names, container prefix stripped, or None for a module the transformer has
    no kind of. A block past the transformer's depth resolves; `_weight_shape` tells it has no weight."""
    if name.startswith(_KOHYA_PREFIX):
        flat = name[len(_KOHYA_PREFIX) :]
        if block := _KOHYA_BLOCK.fullmatch(flat):
            stack = _KOHYA_STACKS[block.group(1)]
            leaf = _KOHYA_LEAVES[stack].get(block.group(3))
            return f"{stack}.{block.group(2)}.{leaf}" if leaf is not None else None
        return _KOHYA_TOP.get(flat)
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if block := _BLOCK.fullmatch(name):
        stack, _, leaf = block.groups()
        return name if leaf in _STACKS[stack][1] or leaf in _STACKS[stack][2] else None
    return name if name in _TOP_LINEARS or name in _TOP_NORMS else None


def _weight_shape(module: str) -> tuple[int, ...] | None:
    """The released shape of the weight a module path from `_module_path` names, None past the transformer's depth."""
    if block := _BLOCK.fullmatch(module):
        stack, index, leaf = block.groups()
        depth, linears, norms = _STACKS[stack]
        return (linears | norms)[leaf] if int(index) < depth else None
    return (_TOP_LINEARS | _TOP_NORMS)[module]


def _is_marker_module(module: str) -> bool:
    """Whether a module is a Linear that says the file is a MOSS-Image one. `time_embedding` is a timestep MLP other
    transformers have under that name."""
    if block := _BLOCK.fullmatch(module):
        return block.group(3) in _STACKS[block.group(1)][1]
    return module in _TOP_LINEARS and not module.startswith("time_embedding.")


def _split_param(key: str) -> tuple[str, str] | None:
    """`(module name, value key)` of a key with a recognized LoRA parameter name; the module name is as written."""
    for param, value_key in _PARAM_VALUE_KEYS.items():
        if key.endswith("." + param):
            return key[: -len(param) - 1], value_key
    return None


def has_moss_image_lora_keys(keys: Iterable[str | int]) -> bool:
    """Whether a state dict is a MOSS-Image LoRA's: a LoRA parameter on a Linear that only MOSS-Image's transformer
    has (a block's `proj.q`..`ffn_down`, its text refiner, its embedders and modulations). A checkpoint carries the
    same module names, but under `.weight`/`.bias`.

    Says which family a file belongs to, not whether it applies: `unsupported_keys` decides that.
    """
    for key in keys:
        split = _split_param(key) if isinstance(key, str) else None
        module = _module_path(split[0]) if split is not None else None
        if module is not None and _is_marker_module(module):
            return True
    return False


_LAYERS = lora_layer_value_sets()
_LOW_RANK = frozenset(
    {frozenset({"lora_down.weight", "lora_up.weight"}), frozenset({"lora_down.weight", "lora_up.weight", "alpha"})}
)


def is_fused_qkv(module: str) -> bool:
    """Whether a module path from the keys' layers is a block's fused `proj.qkv`, which the transformer holds as
    `proj.q`, `proj.k` and `proj.v`."""
    return module.startswith("layers.") and module.endswith(".proj.qkv")


def split_qkv_modules(fused: str) -> tuple[str, str, str]:
    """The `proj.q`, `proj.k` and `proj.v` module paths a block's fused `proj.qkv` is held as."""
    block = fused[: -len("qkv")]
    return block + "q", block + "k", block + "v"


# Why a key keeps a file from applying in full. Shown in the install queue, so plain text.
NOT_A_LORA_PARAMETER = (
    "are not LoRA parameters InvokeAI applies (another adapter kind such as OFT, a full weight, a bias or norm delta, "
    "or a replacement weight)"
)
RAW_PEFT = (
    "are raw PEFT keys (base_model.model.), whose alpha is in an adapter_config.json that is not in the file: "
    "install the ComfyUI or diffusers export instead"
)
NO_MOSS_IMAGE_LAYER = (
    "name no Linear layer of MOSS-Image's transformer (another model's, a text encoder's, or a block it does not have)"
)
NORM_WEIGHT = "patch a norm weight, which a LoRA cannot change on MOSS-Image's transformer"
NAMED_TWICE = "name one layer more than once (under two prefixes, or as both lora_A/B and lora_down/up)"
NOT_ONE_LAYER = "do not make exactly one complete layer (half a pair, a lone alpha or magnitude, two kinds at once)"
WRONG_SHAPE = "do not update a weight of the shape the MOSS-Image layer they name has"
FUSED_AND_SPLIT = "name one block's attention both fused (proj.qkv) and split (proj.q, proj.k, proj.v)"


@dataclass(frozen=True)
class MossImageLoRAKeys:
    """A MOSS-Image LoRA's keys sorted out (`read_moss_image_lora_keys`).

    `layers` maps each module path to its source key per value key. A block's fused `proj.qkv` is a module of its own
    here; every entry is one complete layer. `leftovers` holds the two incomplete entries ai-toolkit writes, which a
    neighbouring layer completes: a lone `dora_scale` on `proj.qkv`, the magnitude of the DoRA layers its split
    `proj.q/k/v` LoRA make, and a lone `alpha` on `proj.q`/`k`/`v`, the scale of the LoKr on the fused `proj.qkv`.
    `refused` holds each key that keeps the file from applying in full, with why.
    """

    layers: Mapping[str, Mapping[str, str]]
    leftovers: Mapping[str, Mapping[str, str]]
    refused: Mapping[str, str]


def _is_lokr(values: Mapping[str, Any]) -> bool:
    return "lokr_w1" in values or "lokr_w1_a" in values


def read_moss_image_lora_keys(state_dict: Mapping[str | int, Any]) -> MossImageLoRAKeys:
    """Sort a MOSS-Image LoRA's keys into layers and the keys that keep it from applying in full.

    A key is refused when it is no LoRA parameter, names a module this transformer has no Linear for (a norm among
    them), names its layer as another key does under another prefix or spelling, or belongs to a layer that is not
    exactly one complete layer (`lora_layer_value_sets`) -- the `leftovers` of `MossImageLoRAKeys` aside. A block that
    names its attention fused and split is refused too. Shapes are `misshapen_moss_image_layers`' to check.
    """
    refused: dict[str, str] = {}
    modules: dict[str, dict[str, list[str]]] = {}
    names: dict[str, set[str]] = {}
    for key in state_dict:
        if isinstance(key, str) and key.startswith("bundle_emb"):
            # Textual-inversion embeddings some trainers bundle; the LoRA loader drops them (`LoRALoader._load_model`).
            continue
        split = _split_param(key) if isinstance(key, str) else None
        if split is None:
            refused[str(key)] = NOT_A_LORA_PARAMETER
            continue
        name, value_key = split
        if name.startswith(_RAW_PEFT_PREFIX):
            refused[str(key)] = RAW_PEFT
            continue
        module = _module_path(name)
        if module is None or _weight_shape(module) is None:
            refused[str(key)] = NO_MOSS_IMAGE_LAYER
            continue
        modules.setdefault(module, {}).setdefault(value_key, []).append(str(key))
        names.setdefault(module, set()).add(name)

    layers: dict[str, dict[str, str]] = {}
    incomplete: dict[str, dict[str, str]] = {}
    leftovers: dict[str, dict[str, str]] = {}
    for module, values in modules.items():
        keys = [key for spellings in values.values() for key in spellings]
        if len(_weight_shape(module) or ()) == 1:
            refused.update(dict.fromkeys(keys, NORM_WEIGHT))
        elif len(names[module]) > 1 or any(len(spellings) > 1 for spellings in values.values()):
            refused.update(dict.fromkeys(keys, NAMED_TWICE))
        else:
            sources = {value_key: spellings[0] for value_key, spellings in values.items()}
            (layers if frozenset(values) in _LAYERS else incomplete)[module] = sources

    for fused in [module for module in (*layers, *incomplete) if is_fused_qkv(module)]:
        split_modules = split_qkv_modules(fused)
        fused_values = layers.get(fused) or incomplete[fused]
        lone_alphas = [m for m in split_modules if set(incomplete.get(m, ())) == {"alpha"}]
        weighted = [m for m in split_modules if m in layers or (m in incomplete and m not in lone_alphas)]
        if fused in layers:
            if weighted:
                for module in (fused, *weighted):
                    refused.update(
                        dict.fromkeys((layers.pop(module, None) or incomplete.pop(module)).values(), FUSED_AND_SPLIT)
                    )
            elif lone_alphas and _is_lokr(fused_values) and "alpha" not in fused_values:
                leftovers.update({module: incomplete.pop(module) for module in lone_alphas})
        elif set(fused_values) == {"dora_scale"} and all(
            m in layers and frozenset(layers[m]) in _LOW_RANK for m in split_modules
        ):
            leftovers[fused] = incomplete.pop(fused)
    for sources in incomplete.values():
        refused.update(dict.fromkeys(sources.values(), NOT_ONE_LAYER))
    return MossImageLoRAKeys(layers=layers, leftovers=leftovers, refused=refused)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _magnitude_fits(layer: Mapping[str, Any], out_features: int, in_features: int, *, fused: bool) -> bool:
    """Whether a DoRA layer's magnitude, if it has one, has an entry per row or column of its weight in the
    orientation `DoRALayer` reads it in: PEFT's and ai-toolkit's one per output row, `(out,)` or `(out, 1)`;
    LyCORIS's `dora_scale` one per output row as `(out, 1)`, otherwise one per input column, `(1, in)` or `(in,)`.
    A fused `proj.qkv` is cut into rows, which only a per-row magnitude survives."""
    if "dora_magnitude" in layer:
        shape = _shape(layer["dora_magnitude"])
        return len(shape) > 0 and shape[0] == out_features and all(size == 1 for size in shape[1:])
    if "dora_scale" in layer:
        allowed = {(out_features, 1)} if fused else {(out_features, 1), (1, in_features), (in_features,)}
        return _shape(layer["dora_scale"]) in allowed
    return True


def misshapen_moss_image_layers(state_dict: Mapping[str | int, Any], keys: MossImageLoRAKeys) -> dict[str, str]:
    """Each key of a layer whose update does not have the shape of the weight it changes in the released transformer,
    or whose DoRA magnitude does not fit that weight, or whose alpha holds more than one value, with why. The
    `leftovers` are held to what their neighbours need of them: a one-value alpha, a magnitude of one entry per row."""
    misshapen: dict[str, str] = {}
    for module, sources in keys.layers.items():
        weight_shape = _weight_shape(module)
        assert weight_shape is not None and len(weight_shape) == 2  # `read_moss_image_lora_keys` resolved a Linear
        values = {value_key: state_dict[key] for value_key, key in sources.items()}
        # Loading reads alpha with `.item()`, which only a single value survives.
        alpha_is_scalar = "alpha" not in values or math.prod(_shape(values["alpha"])) == 1
        fits = (
            update_shape(values) == weight_shape
            and _magnitude_fits(values, *weight_shape, fused=is_fused_qkv(module))
            and alpha_is_scalar
        )
        if not fits:
            misshapen.update(dict.fromkeys(sources.values(), WRONG_SHAPE))
    for module, sources in keys.leftovers.items():
        weight_shape = _weight_shape(module)
        assert weight_shape is not None
        values = {value_key: state_dict[key] for value_key, key in sources.items()}
        if "alpha" in values:
            fits = math.prod(_shape(values["alpha"])) == 1
        else:
            fits = _shape(values["dora_scale"]) == (weight_shape[0], 1)
        if not fits:
            misshapen.update(dict.fromkeys(sources.values(), WRONG_SHAPE))
    return misshapen


def unsupported_keys(state_dict: Mapping[str | int, Any]) -> dict[str, str]:
    """Each key that keeps this state dict from being applied to MOSS-Image as a whole, with why; empty for a LoRA that
    applies in full. Reads shapes only, so identification can run it on meta tensors."""
    keys = read_moss_image_lora_keys(state_dict)
    return {**keys.refused, **misshapen_moss_image_layers(state_dict, keys)}


def describe_refusal(refused: Mapping[str, str]) -> str:
    """Why a MOSS-Image LoRA does not apply in full: each reason with its count of keys and one of them."""
    by_reason: dict[str, list[str]] = {}
    for key, reason in refused.items():
        by_reason.setdefault(reason, []).append(key)
    parts = [f"{len(keys)} key(s) {reason}, e.g. '{sorted(keys)[0]}'" for reason, keys in by_reason.items()]
    return "This MOSS-Image LoRA does not apply in full: " + "; ".join(parts) + "."
