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
from collections.abc import Iterable
from pathlib import Path

from safetensors import safe_open

_QUANTIZATION_CONFIG_FILENAME = "quantization_config.json"


def safetensors_tensor_names(files: Iterable[Path]) -> set[str]:
    """Union of the tensor names declared by `files`, read from their headers.

    Identification only ever needs the *names* — "is there a `<name>.scale` next to this
    `<name>.weight`", "is there a visual tower" — never the tensor data, and the header carries them
    per file. That is what makes this shard-safe, which `ModelOnDisk.load_state_dict()` is not: it
    refuses to pick a file when a folder holds more than one weight file and raises `ValueError` —
    not a `NotAMatchError` — so it aborts that config's probe instead of declining the model. Every
    sharded encoder folder (the FLUX.2 Klein and Z-Image `text_encoder` downloads are all 2-4 shards)
    hit that and fell back to `unknown`.

    Non-safetensors entries are skipped: they are the set `sdnq_sd_loader` reads, and a file whose
    header cannot be read contributes nothing rather than failing the whole check.
    """
    names: set[str] = set()
    for file in sorted(files):
        if file.suffix != ".safetensors":
            continue
        try:
            with safe_open(file, framework="pt", device="cpu") as f:
                names.update(f.keys())
        except Exception:
            continue
    return names


def safetensors_have_sdnq_keys(files: Iterable[Path]) -> bool:
    """True if `files` carry an SDNQ ``<name>.weight`` / ``<name>.scale`` pair.

    The pair is resolved across the union of every shard, never within a single file: sharding splits
    a checkpoint by tensor order, so a weight and its scale routinely land in different files.
    """
    names = safetensors_tensor_names(files)
    return any(name.endswith(".weight") and f"{name[: -len('.weight')]}.scale" in names for name in names)


def folder_has_sdnq_keys(folder_path: Path) -> bool:
    """True if the safetensors directly in `folder_path` carry an SDNQ weight/scale pair.

    Only safetensors are inspected, which is the same set `sdnq_sd_loader` reads — a `.bin` holding
    SDNQ-shaped tensors is not something we could load anyway, so calling it SDNQ would only move the
    failure around.
    """
    if not folder_path.is_dir():
        return False

    return safetensors_have_sdnq_keys(folder_path.glob("*.safetensors"))


def folder_has_sdnq_marker(folder_path: Path) -> bool:
    """True if `folder_path` holds a `quantization_config.json` naming SDNQ as the quant method.

    A marker that is missing, unreadable or names another method is not evidence *against* SDNQ
    weights — callers fall through to the key shape rather than treating False as "not SDNQ".
    """
    marker = folder_path / _QUANTIZATION_CONFIG_FILENAME
    if not marker.is_file():
        return False
    try:
        with open(marker, "r", encoding="utf-8") as f:
            return json.load(f).get("quant_method") == "sdnq"
    except (json.JSONDecodeError, OSError):
        return False


def is_sdnq_folder(folder_path: Path) -> bool:
    """True if `folder_path` holds SDNQ-quantized weights.

    Checks the `quantization_config.json` marker first because it is definitive and free, then falls
    back to the key shape. The fallback is what covers exports that ship no marker — without it such
    a folder reads as plain diffusers to identification and as SDNQ to nothing at all.
    """
    return folder_has_sdnq_marker(folder_path) or folder_has_sdnq_keys(folder_path)
