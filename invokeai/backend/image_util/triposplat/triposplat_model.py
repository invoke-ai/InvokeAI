"""InvokeAI's model-manager wrapper for the vendored TripoSplat pipeline (this file is NOT vendored)."""

from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.raw_model import RawModel


class TripoSplatModel(RawModel):
    """Wraps the vendored TripoSplatPipeline so the model manager's memory cache can move it between CPU
    and GPU.

    The pipeline holds five sub-modules with mixed dtypes (DINOv3 + Flux2 VAE are bfloat16; rmbg / flow /
    decoder are float16), so we move device only and never re-cast dtype.
    """

    def __init__(self, pipe: object):
        self._pipe = pipe

    @property
    def _submodules(self) -> tuple:
        p = self._pipe
        return (p.dinov3, p.vae_encoder, p.rmbg, p.flow_model, p.decoder)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        # Only CPU/CUDA are supported (mirrors GroundingDinoPipeline's MPS guard). dtype is intentionally
        # ignored: the sub-modules require their original mixed precision.
        if device is None or device.type not in {"cpu", "cuda"}:
            return
        for module in self._submodules:
            module.to(device=device)
        self._pipe._device = torch.device(device)

    def calc_size(self) -> int:
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return sum(calc_module_size(module) for module in self._submodules)

    @staticmethod
    def load_model(model_path: Path) -> "TripoSplatModel":
        from invokeai.backend.image_util.triposplat.triposplat import TripoSplatPipeline

        def _find(filename: str) -> str:
            # The download cache may hand us the repo dir or a parent; locate each checkpoint by name so
            # we are robust to the exact directory layout returned by download_and_cache_model.
            matches = list(model_path.rglob(filename))
            if not matches:
                raise FileNotFoundError(f"TripoSplat checkpoint '{filename}' not found under {model_path}")
            return str(matches[0])

        # Construct on CPU; the model cache moves it to GPU on lock via .to().
        pipe = TripoSplatPipeline(
            ckpt_path=_find("triposplat_fp16.safetensors"),
            decoder_path=_find("triposplat_vae_decoder_fp16.safetensors"),
            dinov3_path=_find("dino_v3_vit_h.safetensors"),
            flux2_vae_encoder_path=_find("flux2-vae.safetensors"),
            rmbg_path=_find("birefnet.safetensors"),
            device="cpu",
        )
        return TripoSplatModel(pipe)
