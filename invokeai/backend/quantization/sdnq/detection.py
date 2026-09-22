"""Deciding whether a folder holds SDNQ-quantized weights.

This is one question with one answer, but it used to have four near-identical implementations — in
`configs/main.py`, the FLUX and Z-Image loaders, and `vae.py` — and only the last of them looked past
the `quantization_config.json` marker. That divergence is a real defect rather than untidiness:
identification and loading consult the *same* folder and must reach the *same* verdict. When they
disagree on a markerless export, identification hands the folder to a plain-diffusers config and the
loader then calls `from_pretrained()` on packed SDNQ weights, which either fails outright or
misreads them.
"""

import json
from pathlib import Path

from safetensors import safe_open

_QUANTIZATION_CONFIG_FILENAME = "quantization_config.json"


def folder_has_sdnq_keys(folder_path: Path) -> bool:
    """True if the safetensors in `folder_path` carry an SDNQ ``<name>.weight`` / ``<name>.scale`` pair.

    The pair is resolved across the union of every shard, never within a single file: sharding splits
    a checkpoint by tensor order, so a weight and its scale routinely land in different files.

    Only safetensors are inspected, which is the same set `sdnq_sd_loader` reads — a `.bin` holding
    SDNQ-shaped tensors is not something we could load anyway, so calling it SDNQ would only move the
    failure around.
    """
    if not folder_path.is_dir():
        return False

    keys: set[str] = set()
    for shard in sorted(folder_path.glob("*.safetensors")):
        try:
            with safe_open(shard, framework="pt", device="cpu") as f:
                keys.update(f.keys())
        except Exception:
            continue

    return any(key.endswith(".weight") and f"{key[: -len('.weight')]}.scale" in keys for key in keys)


def is_sdnq_folder(folder_path: Path) -> bool:
    """True if `folder_path` holds SDNQ-quantized weights.

    Checks the `quantization_config.json` marker first because it is definitive and free, then falls
    back to the key shape. The fallback is what covers exports that ship no marker — without it such
    a folder reads as plain diffusers to identification and as SDNQ to nothing at all.
    """
    marker = folder_path / _QUANTIZATION_CONFIG_FILENAME
    if marker.is_file():
        try:
            with open(marker, "r", encoding="utf-8") as f:
                if json.load(f).get("quant_method") == "sdnq":
                    return True
        except (json.JSONDecodeError, OSError):
            pass
        # A marker that exists but names another method is not evidence *against* SDNQ keys, so fall
        # through rather than returning False here.

    return folder_has_sdnq_keys(folder_path)
