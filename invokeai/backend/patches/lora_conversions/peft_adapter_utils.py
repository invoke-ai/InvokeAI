"""Utilities for handling PEFT named-adapter LoRA state dicts.

PEFT (HuggingFace Parameter-Efficient Fine-Tuning) supports multiple named adapters per model.
When saved, the adapter name is encoded in the weight key:

    Standard PEFT:           foo.bar.lora_A.weight
    Named-adapter PEFT:      foo.bar.lora_A.<adapter_name>.weight

The most common adapter name is "default", produced automatically by `model.add_adapter()`
without an explicit name. Some training tools (e.g. Diffusers' PEFT integration with
multi-adapter support, certain LoRA fine-tuning scripts) save in this format even with a
single adapter.

InvokeAI's downstream LoRA detection and conversion code expects the standard PEFT suffix
(`lora_A.weight` / `lora_B.weight`). This module normalizes named-adapter state dicts to
that form so the rest of the pipeline can handle them transparently.
"""

import re
from typing import Any

# Match a named-adapter PEFT key ending: .lora_A.<adapter_name>.weight or .lora_B.<adapter_name>.weight.
# The adapter name is a single dot-free component (PEFT identifiers do not contain dots).
_NAMED_ADAPTER_RE = re.compile(r"\.lora_([AB])\.([^.]+)\.weight$")


def _extract_adapter_names(state_dict: dict[str | int, Any]) -> set[str]:
    """Return the set of distinct PEFT adapter names found in the state dict.

    A "named adapter" key is one matching `.lora_A.<name>.weight` or `.lora_B.<name>.weight`.
    Keys in the standard PEFT form (`.lora_A.weight` / `.lora_B.weight`) do not contribute.
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
            new_key = _NAMED_ADAPTER_RE.sub(r".lora_\1.weight", key)
            normalized[new_key] = value
        else:
            normalized[key] = value
    return normalized
