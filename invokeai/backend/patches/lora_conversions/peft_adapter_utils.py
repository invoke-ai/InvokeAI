"""Utilities for handling PEFT named-adapter LoRA state dicts.

PEFT (HuggingFace Parameter-Efficient Fine-Tuning) supports multiple named adapters per model.
When saved, the adapter name is encoded in the weight key:

    Standard PEFT:           foo.bar.lora_A.weight
    Named-adapter PEFT:      foo.bar.lora_A.<adapter_name>.weight

A DoRA adapter's magnitude carries the name the same way (`foo.bar.lora_magnitude_vector.<adapter_name>.weight`).

The most common adapter name is "default", produced automatically by `model.add_adapter()`
without an explicit name. Some training tools (e.g. Diffusers' PEFT integration with
multi-adapter support, certain LoRA fine-tuning scripts) save in this format even with a
single adapter.

InvokeAI's downstream LoRA detection and conversion code expects the standard PEFT suffix
(`lora_A.weight` / `lora_B.weight` / `lora_magnitude_vector.weight`). This module normalizes named-adapter state
dicts to that form so the rest of the pipeline can handle them transparently.
"""

import json
import math
import re
from collections.abc import Mapping
from typing import Any

# Match a named-adapter PEFT key ending: .lora_A.<adapter_name>.weight, .lora_B.<adapter_name>.weight or a DoRA
# magnitude's .lora_magnitude_vector.<adapter_name>.weight. The adapter name is a single dot-free component (PEFT
# identifiers do not contain dots).
_NAMED_ADAPTER_RE = re.compile(r"\.(lora_A|lora_B|lora_magnitude_vector)\.([^.]+)\.weight$")


def _extract_adapter_names(state_dict: dict[str | int, Any]) -> set[str]:
    """Return the set of distinct PEFT adapter names found in the state dict.

    A "named adapter" key is one matching `.lora_A.<name>.weight`, `.lora_B.<name>.weight` or
    `.lora_magnitude_vector.<name>.weight`. Keys in the standard PEFT form (`.lora_A.weight` / `.lora_B.weight`) do
    not contribute.
    """
    names: set[str] = set()
    for key in state_dict:
        if not isinstance(key, str):
            continue
        m = _NAMED_ADAPTER_RE.search(key)
        if m:
            names.add(m.group(2))
    return names


def has_peft_named_adapter_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check whether the state dict contains any PEFT named-adapter keys."""
    return bool(_extract_adapter_names(state_dict))


def normalize_peft_adapter_names(state_dict: dict[str | int, Any]) -> dict[str | int, Any]:
    """Return a state dict with PEFT named-adapter suffixes stripped to the standard form.

    Transforms:
        foo.bar.lora_A.<adapter_name>.weight  →  foo.bar.lora_A.weight
        foo.bar.lora_B.<adapter_name>.weight  →  foo.bar.lora_B.weight
        foo.bar.lora_magnitude_vector.<adapter_name>.weight  →  foo.bar.lora_magnitude_vector.weight

    Only applied when the state dict contains exactly one distinct adapter name. If the
    file holds multiple adapters, the keys are left untouched (renaming would collide and
    multi-adapter LoRAs are not currently supported by InvokeAI).

    If no named-adapter keys are present, the input dict is returned unchanged.
    """
    adapter_names = _extract_adapter_names(state_dict)
    if len(adapter_names) != 1:
        return state_dict

    normalized: dict[str | int, Any] = {}
    for key, value in state_dict.items():
        if isinstance(key, str):
            new_key = _NAMED_ADAPTER_RE.sub(r".\1.weight", key)
            normalized[new_key] = value
        else:
            normalized[key] = value
    return normalized


LORA_ADAPTER_METADATA_KEY = "lora_adapter_metadata"
"""The header entry diffusers saves PEFT `LoraConfig`s in: a pipeline's save keys each component's fields
`<component>.<field>`, a model's `save_lora_adapter` writes the model's own unprefixed."""


def peft_scale(metadata: Mapping[str, str] | None, *, family: str) -> float | None:
    """The scale PEFT gives every low-rank layer of the transformer LoRA a diffusers header describes, or None for a
    header that describes none. `family` names the model in the messages of the ValueErrors.

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
        raise ValueError(f"This {family} LoRA's lora_adapter_metadata header is not JSON ({e}).") from None
    if not isinstance(fields, dict):
        raise ValueError(f"This {family} LoRA's lora_adapter_metadata header is not a JSON object of PEFT settings.")
    prefixed = {
        key.removeprefix("transformer."): value for key, value in fields.items() if key.startswith("transformer.")
    }
    # Another component's fields (`text_encoder.r`) are not the transformer's.
    config = prefixed or {key: value for key, value in fields.items() if "." not in key}
    if not config:
        return None
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError(
            f"This {family} LoRA's lora_adapter_metadata header describes a PEFT {config['peft_type']} adapter; "
            "InvokeAI applies PEFT LoRA headers only."
        )
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(
            f"This {family} LoRA sets its rank or alpha per module (rank_pattern or alpha_pattern in its "
            "lora_adapter_metadata header), which InvokeAI does not apply."
        )
    rank, alpha = config.get("r", 8), config.get("lora_alpha", 8)
    if not isinstance(rank, int | float) or not isinstance(alpha, int | float) or rank <= 0:
        raise ValueError(f"This {family} LoRA's lora_adapter_metadata header gives no usable r and lora_alpha.")
    return alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank
