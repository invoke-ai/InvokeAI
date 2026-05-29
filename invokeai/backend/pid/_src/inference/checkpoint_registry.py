# Shared official PID checkpoint registry.
#
# Single source of truth for the (experiment_name, checkpoint_path) pair used by
# every pixel-decoder demo in `pid/_src/inference/`. The registry is keyed by
# (backbone, ckpt_type):
#
#   ckpt_type = "2k"      Original 2048px-trained decoders, used as
#                         512→2048 (4×) decoder for diffusers-style backbones,
#                         or 256→2048 (8×) for Scale-RAE.
#   ckpt_type = "2kto4k"  Multi-resolution-trained decoders (data bucketing
#                         2048→3840 + SD3-style dynamic shift). Designed for
#                         1024 LDM → 4K (3840) decoding. Only registered for
#                         the diffusers backbones (flux/flux2/sd3/zimage);
#                         rae / scale_rae have no 2kto4k variant.
#
# Backbone-tag namespace:
#   flux       Flux1-dev (16-ch VAE)                     LDM + from_clean   (2k + 2kto4k)
#   flux2      Flux2-dev (128-ch BN VAE)                 LDM + from_clean   (2k + 2kto4k)
#   sd3        SD3 medium (16-ch VAE)                    LDM + from_clean   (2k + 2kto4k)
#   zimage     ZImage (Flux1's 16-ch VAE)                LDM only — reuses Flux1 model (2k + 2kto4k)
#   zimage_turbo  ZImage-Turbo (same 16-ch VAE)          LDM only — reuses Flux1 model (2k + 2kto4k)
#   rae        DINOv2-B + RAE ViT-XL (768-ch RAE)        LDM + from_clean   (2k only, sr4x)
#   scale_rae  SigLIP-2 So400M + Scale-RAE ViT-XL        LDM + from_clean   (2k only, sr8x)
#
# `pid_scale` is the spatial upscaling factor baked into the PID network
# (sr4x → 4, sr8x → 8) and is forwarded to the demo's --scale argument.

from dataclasses import dataclass


@dataclass(frozen=True)
class PIDCheckpoint:
    experiment: str
    checkpoint_path: str
    pid_scale: int


_CKPT_ROOT = "checkpoints"

VALID_CKPT_TYPES = ("2k", "2kto4k")


PID_CHECKPOINT_REGISTRY: dict[tuple[str, str], PIDCheckpoint] = {
    # ---- 2k (the original 2048-trained release) ----
    ("flux", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("flux2", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux2_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("sd3", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_sd3_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("zimage", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("rae", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_dinov2_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_dinov2_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("scale_rae", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr8x_official_siglip_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr8x_official_siglip_distill_4step/model_ema_bf16.pth",
        pid_scale=8,
    ),
    # ---- 2kto4k (multi-res-trained, dynamic_shift-aware) ----
    ("flux", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("flux2", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_flux2_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("sd3", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_sd3_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
}
# ZImage and ZImage-Turbo use Flux1's 16-ch VAE for both ckpt types → alias to
# the flux entries. Keep explicit aliases (vs. duplicating) so updating "flux"
# updates these backbones too.
PID_CHECKPOINT_REGISTRY[("zimage_turbo", "2k")] = PID_CHECKPOINT_REGISTRY[("flux", "2k")]
PID_CHECKPOINT_REGISTRY[("zimage", "2kto4k")] = PID_CHECKPOINT_REGISTRY[("flux", "2kto4k")]
PID_CHECKPOINT_REGISTRY[("zimage_turbo", "2kto4k")] = PID_CHECKPOINT_REGISTRY[("flux", "2kto4k")]


def get_pid_checkpoint(backbone: str, ckpt_type: str = "2k") -> PIDCheckpoint:
    """Return the registered official PID checkpoint for `(backbone, ckpt_type)`.

    `ckpt_type` defaults to `"2k"` so existing call sites keep their pre-2kto4k
    behavior. Raises KeyError with the list of valid keys when the pair is
    unknown — typical cause is asking for a `2kto4k` variant of a backbone
    that doesn't ship one (rae / scale_rae).
    """
    if ckpt_type not in VALID_CKPT_TYPES:
        raise KeyError(f"Unknown ckpt_type {ckpt_type!r}. Valid: {VALID_CKPT_TYPES}")
    try:
        return PID_CHECKPOINT_REGISTRY[(backbone, ckpt_type)]
    except KeyError as exc:
        valid = ", ".join(sorted(f"{b}+{t}" for b, t in PID_CHECKPOINT_REGISTRY))
        raise KeyError(f"Unknown (backbone, ckpt_type)=({backbone!r}, {ckpt_type!r}). Valid: {valid}") from exc


__all__ = [
    "PIDCheckpoint",
    "PID_CHECKPOINT_REGISTRY",
    "VALID_CKPT_TYPES",
    "get_pid_checkpoint",
]
