"""Tiny real Spandrel upscalers, for tests that identify or re-identify one from disk."""

from pathlib import Path

from safetensors.torch import save_file
from spandrel.architectures.Compact import Compact


def write_compact_upscaler(path: Path, scale: int) -> Path:
    """Save a minimal Compact (Real-ESRGAN SRVGG) model with the given native scale to `path`."""
    model = Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=scale)
    save_file(model.state_dict(), path)
    return path
