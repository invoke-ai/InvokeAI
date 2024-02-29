# Copyright (c) 2023 Lincoln Stein and the InvokeAI Team
"""
Utility routine used for autodetection of optimal slice size
for attention mechanism.
"""

import psutil
import torch


def auto_detect_slice_size(latents: torch.Tensor) -> str:
    bytes_per_element_needed_for_baddbmm_duplication = latents.element_size() + 4
    max_size_required_for_baddbmm = (
        16
        * latents.size(dim=2)
        * latents.size(dim=3)
        * latents.size(dim=2)
        * latents.size(dim=3)
        * bytes_per_element_needed_for_baddbmm_duplication
    )
    if latents.device.type in {"cpu", "mps"}:
        mem_free = psutil.virtual_memory().free
    elif latents.device.type == "cuda":
        mem_free, _ = torch.cuda.mem_get_info(latents.device)
    else:
        raise ValueError(f"unrecognized device {latents.device}")

    if max_size_required_for_baddbmm > (mem_free * 3.0 / 4.0):
        return "max"
    elif torch.backends.mps.is_available():
        return "max"
    else:
        return "balanced"
