# Copyright (c) 2024 Lincoln Stein and the InvokeAI Development Team
"""
This module exports the function has_baked_in_sdxl_vae().
It returns True if an SDXL checkpoint model has the original SDXL 1.0 VAE,
which doesn't work properly in fp16 mode.
"""

import hashlib
from pathlib import Path

from safetensors.torch import load_file

SDXL_1_0_VAE_HASH = "bc40b16c3a0fa4625abdfc01c04ffc21bf3cefa6af6c7768ec61eb1f1ac0da51"


def has_baked_in_sdxl_vae(checkpoint_path: Path) -> bool:
    """Return true if the checkpoint contains a custom (non SDXL-1.0) VAE."""
    hash = _vae_hash(checkpoint_path)
    return hash != SDXL_1_0_VAE_HASH


def _vae_hash(checkpoint_path: Path) -> str:
    checkpoint = load_file(checkpoint_path, device="cpu")
    vae_keys = [x for x in checkpoint.keys() if x.startswith("first_stage_model.")]
    hash = hashlib.new("sha256")
    for key in vae_keys:
        value = checkpoint[key]
        hash.update(bytes(key, "UTF-8"))
        hash.update(bytes(str(value), "UTF-8"))

    return hash.hexdigest()
