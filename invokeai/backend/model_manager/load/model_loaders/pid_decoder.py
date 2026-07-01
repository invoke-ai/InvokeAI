"""Loader for PiD (Pixel Diffusion Decoder) checkpoints.

Returns a fully-constructed `PidNet` so the model cache can size it
correctly and apply its standard sequential-offload / partial-load
policies. We instantiate the architecture (per backbone) here and pour the
checkpoint's tensors directly into it, then discard the intermediate state
dict — avoiding the 2x VRAM peak you would get from holding both a `dict`
and the live module at the same time.
"""

from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as safetensors_load_file

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.pid.decode import load_pid_decoder

# NVIDIA's official PiD `.pth` checkpoints store the student under the `net.`
# prefix (see `PidDistillModel.state_dict(prefix="net.")` in the vendored
# upstream). We strip it on load so PidNet.load_state_dict() can consume the
# dict directly.
_NET_PREFIX = "net."


def _load_raw_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return safetensors_load_file(str(path))
    if suffix in {".pth", ".pt", ".ckpt", ".bin"}:
        # NVIDIA's PiD `.pth` checkpoints are plain tensor dicts (verified
        # against the released res2k_sr4x_official_flux checkpoint).
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
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusionXL, type=ModelType.PiDDecoder, format=ModelFormat.Checkpoint
)
class PiDDecoderLoader(ModelLoader):
    """Loads a PiD checkpoint into a fully-constructed PidNet of the matching backbone."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("Unexpected submodel requested for PiD decoder.")

        # Backbone is encoded in the config's `base` field — populated by
        # PiDDecoder_Checkpoint_*_Config when the user added the model.
        backbone: BaseModelType = config.base

        raw_sd = _strip_net_prefix(_load_raw_checkpoint(Path(config.path)))

        # Build the live PidNet on CPU and pour the checkpoint in — then drop
        # the dict so we don't hold two copies in RAM at once.
        pid_net = load_pid_decoder(raw_sd, backbone)
        del raw_sd

        # We deliberately keep PidNet's parameters in float32 here. PiD
        # consumes Gemma-2 hidden states that contain large outliers
        # (per-token max well past 100) and the in-network RMSNorm
        # (`variance = hidden_states.pow(2).mean(-1, keepdim=True)`) loses
        # precision badly in bf16, producing all-NaN outputs. The decode
        # wrapper runs the forward pass under `torch.autocast(bf16)` so the
        # bulk of the matmuls still execute in bf16 — only the precision-
        # critical reductions stay fp32. This roughly doubles VRAM for the
        # weights (~5 GB instead of ~2.5 GB) but is the only configuration
        # we have measured to be numerically stable end-to-end.

        pid_net.eval()
        pid_net.requires_grad_(False)
        return pid_net
