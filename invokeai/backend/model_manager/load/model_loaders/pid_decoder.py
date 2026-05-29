"""Loader for PiD (Pixel Diffusion Decoder) checkpoints.

Returns the raw `state_dict` keyed by the underlying PidNet module layout
(i.e. with the upstream `net.` prefix stripped). The downstream PiDDecoder
wrapper (invokeai/backend/pid/decode.py, Phase C) is responsible for
instantiating a `PidNet` of the matching backbone and loading the state dict
into it. Returning a state dict here keeps the Phase B model-manager wiring
independent from Phase C's decode pipeline.
"""

from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as safetensors_load_file

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType

# NVIDIA's official PiD `.pth` checkpoints store the student under the `net.`
# prefix (see `PidDistillModel.state_dict(prefix="net.")` in the vendored
# upstream). We strip it on load so the Phase C wrapper can call
# `PidNet.load_state_dict()` directly.
_NET_PREFIX = "net."


def _load_raw_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return safetensors_load_file(str(path))
    if suffix in {".pth", ".pt", ".ckpt", ".bin"}:
        # NVIDIA's official PiD `.pth` checkpoints are plain tensor dicts
        # (verified against the released res2k_sr4x_official_flux checkpoint),
        # so weights_only=True is sufficient and avoids the arbitrary-code
        # execution risk of full unpickling.
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        return sd  # type: ignore[return-value]
    raise ValueError(f"Unrecognised PiD decoder checkpoint extension: {suffix!r}")


def _strip_net_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith(_NET_PREFIX) for k in state_dict if isinstance(k, str)):
        return state_dict
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(_NET_PREFIX):
            out[k[len(_NET_PREFIX) :]] = v
        elif isinstance(k, str) and (
            k.startswith("net_ema.") or k.startswith("fake_score.") or k.startswith("discriminator.")
        ):
            continue
        else:
            out[k] = v
    return out


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.PiDDecoder, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.PiDDecoder, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusion3, type=ModelType.PiDDecoder, format=ModelFormat.Checkpoint
)
class PiDDecoderLoader(ModelLoader):
    """Loads a PiD decoder checkpoint (.pth / .safetensors) as a raw state dict."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("Unexpected submodel requested for PiD decoder.")

        state_dict = _strip_net_prefix(_load_raw_checkpoint(Path(config.path)))

        if self._torch_dtype is not None:
            for k, v in state_dict.items():
                if v.is_floating_point():
                    state_dict[k] = v.to(self._torch_dtype)

        return state_dict
