# SPDX-License-Identifier: Apache-2.0
"""Decode pipeline for the vendored PiD (Pixel Diffusion Decoder).

This module bridges between InvokeAI's model-manager-loaded PiD checkpoints
(state dicts produced by `model_loaders/pid_decoder.py`) and the underlying
`PidNet` super-resolution network. It deliberately reimplements the small
sampling loop from `PidDistillModel.generate_samples_from_batch` (vendored
in `_src/models/pid_distill_model.py`) so the wrapper stays free of the
upstream's CUDA-only, distributed-training-flavoured init paths and can be
driven entirely by InvokeAI's per-call device / dtype choices.

Hyperparameters were extracted from PiD's `pid_sr4x` base net config and
the per-backbone experiment overrides (NVIDIA's upstream `pid/_src/configs/`,
not vendored here — only the values needed at inference). See
`shared_config.py` and `experiment/{flux,flux2,sd3}.py` in the upstream
repository for the source of truth.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid._src.networks.pid_net import PidNet

# ---------------------------------------------------------------------------
# Network hyperparameters per backbone
# ---------------------------------------------------------------------------

# `pid_sr4x` base config (defaults/model_pid.py upstream) plus the shared
# `_common_model_overrides` net dict (experiment/shared_config.py upstream).
_PID_SR4X_BASE: dict = {
    # T2I backbone (PixDiT_T2I args)
    "in_channels": 3,
    "num_groups": 24,
    "hidden_size": 1536,
    "pixel_hidden_size": 16,
    "pixel_attn_hidden_size": 1152,
    "pixel_num_groups": 16,
    "patch_depth": 14,
    "pixel_depth": 2,
    "patch_size": 16,
    "txt_embed_dim": 2304,  # Gemma-2-2b-it hidden size
    "txt_max_length": 300,
    "use_text_rope": True,
    "text_rope_theta": 10000.0,
    "rope_mode": "ntk_aware",
    "rope_ref_h": 1024,
    "rope_ref_w": 1024,
    "repa_encoder_index": -1,  # REPA disabled at inference
    # SR / LQ branch
    "lq_inject_mode": "controlnet",
    "lq_in_channels": 0,
    "lq_hidden_dim": 512,
    "lq_gate_type": "sigma_aware_per_token_per_dim",
    "lq_interval": 2,  # overridden by shared_config
    "zero_init_lq": True,
    "train_lq_proj_only": False,
    "sr_scale": 4,
    "pit_lq_inject": False,
    "pit_lq_gate_type": "sigma_aware_per_token_per_dim",
}

# Per-backbone net deltas (mirrors upstream experiment/{name}.py).
_PER_BACKBONE: dict[BaseModelType, dict] = {
    BaseModelType.Flux: {
        "lq_latent_channels": 16,
        "latent_spatial_down_factor": 8,
    },
    BaseModelType.Flux2: {
        "lq_latent_channels": 128,
        "latent_spatial_down_factor": 16,
    },
    BaseModelType.StableDiffusion3: {
        "lq_latent_channels": 16,
        "latent_spatial_down_factor": 8,
    },
}

# Distilled-student schedule (`student_t_list` from shared_config).
_STUDENT_T_LIST: list[float] = [0.999, 0.866, 0.634, 0.342, 0.0]

# Flow-matching timescale that maps the [0,1] schedule to the network's
# expected timestep range.
_FM_TIMESCALE: float = 1000.0

# Caption pre-processing constants from PiD's `shared_config.py`. The model
# was trained with these strings prepended; using anything else degrades
# quality. See `_encode_text_raw` in the upstream pixeldit_model.py.
PID_CHI_PROMPT: str = "\n".join(
    [
        'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]
)
PID_NEGATIVE_PROMPT: str = (
    "low quality, worst quality, over-saturated, three legs, six fingers, cartoon, anime, "
    "cgi, low res, blurry, deformed, distortion, duplicated limbs, plastic skin, jpeg artifacts, "
    "watermark"
)
PID_MODEL_MAX_LENGTH: int = 300


# Working-memory (activation) estimate for the PiD decode, mirroring `estimate_vae_working_memory_*` (see #8414).
# PiD runs a multi-step pixel-diffusion at the full super-resolved output resolution, so its peak activation
# memory scales with the OUTPUT pixel count. This constant is an experimentally-tunable starting value: it must
# stay small enough not to over-reserve VRAM on setups that already run PiD, while still reserving enough that the
# model cache offloads the (large) main transformer before the decode runs. Calibrate against measured peak VRAM.
_PID_DECODE_WORKING_MEMORY_SCALING_CONSTANT = 160


def estimate_pid_decode_working_memory(latent: Tensor, backbone: BaseModelType) -> int:
    """Estimate the working (activation) memory in bytes for a PiD decode of *latent*.

    The decoded image is ``latent_spatial * sr_scale * latent_spatial_down_factor`` pixels per side. PidNet runs
    in float32 (see ``model_loaders/pid_decoder.py``), so the element size is 4 bytes. Returns 0 for unsupported
    backbones so callers fall back to the cache's default working-memory reservation.
    """
    per_backbone = _PER_BACKBONE.get(backbone)
    if per_backbone is None:
        return 0
    total_up = int(_PID_SR4X_BASE["sr_scale"]) * int(per_backbone["latent_spatial_down_factor"])
    out_h = int(latent.shape[-2]) * total_up
    out_w = int(latent.shape[-1]) * total_up
    element_size = 4  # PidNet runs in float32 (see model_loaders/pid_decoder.py)
    return int(out_h * out_w * element_size * _PID_DECODE_WORKING_MEMORY_SCALING_CONSTANT)


def build_pid_net(backbone: BaseModelType) -> PidNet:
    """Build an uninitialised PidNet of the right shape for *backbone*.

    The returned network is on CPU and in float32; the caller is responsible
    for casting it to the desired dtype/device before loading weights.
    """
    if backbone not in _PER_BACKBONE:
        raise ValueError(
            f"PiD decoder backbone {backbone!r} is not supported. Expected one of: {list(_PER_BACKBONE.keys())}."
        )
    kwargs = {**_PID_SR4X_BASE, **_PER_BACKBONE[backbone]}
    return PidNet(**kwargs)


def load_pid_decoder(state_dict: dict[str, Tensor], backbone: BaseModelType) -> PidNet:
    """Instantiate a PidNet for *backbone* and populate it with *state_dict*.

    The state dict is expected to be the model-manager loader's output, i.e.
    already stripped of the `net.` prefix used by NVIDIA's distill model
    serialisation. The caller still owns dtype/device placement of the
    returned net.
    """
    net = build_pid_net(backbone)
    # strict=False keeps parity with the upstream loader: missing LQ-projection
    # keys are tolerated when reloading PixDiT_T2I weights into PidNet, and
    # extra keys (e.g. legacy EMA artefacts) are dropped.
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"PiD checkpoint has unexpected keys not present in PidNet: {unexpected[:5]}"
            + (f" (+ {len(unexpected) - 5} more)" if len(unexpected) > 5 else "")
        )
    if missing:
        # We tolerate missing `lq_proj.*` (e.g. if the user accidentally
        # passed a vanilla PixDiT_T2I checkpoint), but anything else points
        # to a real architecture mismatch.
        non_lq = [k for k in missing if "lq_proj" not in k]
        if non_lq:
            raise RuntimeError(
                f"PiD checkpoint is missing non-LQ keys required by PidNet: {non_lq[:5]}"
                + (f" (+ {len(non_lq) - 5} more)" if len(non_lq) > 5 else "")
            )
    return net


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _get_t_list(device: torch.device, *, num_steps: Optional[int] = None) -> Tensor:
    """Distill-student sigma schedule.

    When *num_steps* differs from the trained 4 steps, linearly sub-sample
    the canonical 5-point list (mirrors `PidDistillModel._get_t_list`).
    """
    full = torch.tensor(_STUDENT_T_LIST, device=device, dtype=torch.float32)
    if num_steps is None or num_steps == 4:
        t = full
    else:
        idx = torch.linspace(0, len(full) - 1, num_steps + 1).round().long()
        t = full[idx]
    assert abs(t[-1].item()) < 1e-6, "t_list must end at 0"
    return t


def _velocity_to_x0(x_t: Tensor, net_output: Tensor, t: Tensor) -> Tensor:
    """Convert the network's velocity prediction back to x0 at time *t*."""
    s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
    t_shaped = t.double().view(*s)
    return (x_t.double() - t_shaped * net_output.double()).to(x_t.dtype)


@torch.no_grad()
def _student_sample_loop(
    net: PidNet,
    *,
    noise: Tensor,
    t_list: Tensor,
    caption_embs: Tensor,
    caption_mask: Optional[Tensor],
    lq_latent: Optional[Tensor],
    degrade_sigma: Tensor,
    sample_type: str = "sde",
    autocast_dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Few-step distilled sampler.

    Mirrors `PidDistillModel._student_sample_loop` — the only mode supported
    here is "sde" (the default for the released res2k_sr4x checkpoints).

    ``autocast_dtype`` mirrors PiD's training-time precision config (bf16):
    the parameters can stay in float32 but cosines / RoPE tensors created
    inside the forward must be cast on the fly. Set to ``None`` to disable.
    """
    batch_size = noise.shape[0]
    x = noise
    autocast_ctx = (
        torch.autocast(noise.device.type, dtype=autocast_dtype)
        if autocast_dtype is not None and noise.device.type == "cuda"
        else nullcontext()
    )
    for t_cur, t_next in zip(t_list[:-1], t_list[1:], strict=True):
        t_cur_batch = t_cur.expand(batch_size)
        with autocast_ctx:
            # Do not pass the caption mask through here: upstream PiD's
            # PidDistillModel sampler omits it too, and PidNet forwards the
            # same `mask` argument unchanged to its pixel blocks where the
            # shape (B, T_text) is incompatible with the patch-token K
            # dimension that block expects. We keep `caption_mask` available
            # in the signature so a future patch-block-only path can reuse
            # it without another API change.
            v_pred = net(
                x,
                t_cur_batch * _FM_TIMESCALE,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
            )
        if t_next.item() > 0:
            x0_pred = _velocity_to_x0(x, v_pred, t_cur_batch)
            eps_infer = torch.randn(
                x0_pred.shape,
                device=x0_pred.device,
                dtype=x0_pred.dtype,
                generator=generator,
            )
            broadcast_shape = [batch_size] + [1] * (x.ndim - 1)
            t_next_b = t_next.reshape(1).expand(broadcast_shape)
            if sample_type == "ode":
                # ODE step (kept for symmetry; unused by the 4-step preset).
                dt = t_next - t_cur
                x = x + dt * v_pred
            else:
                x = (1.0 - t_next_b) * x0_pred + t_next_b * eps_infer
        else:
            x = _velocity_to_x0(x, v_pred, t_cur_batch)
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PiDDecodeConfig:
    """Per-call decode knobs.

    The defaults match NVIDIA's released `res2k_sr4x_*_distill_4step`
    presets; callers (i.e. the Phase 6.x invocations) may override them.
    """

    num_inference_steps: int = 4
    scale: int = 4
    sample_type: str = "sde"
    # Caller-supplied per-sample noise levels of the input latent — 0.0 means
    # "the latent is the clean x0 from the LDM" (the from_ldm path); the
    # from_clean upscale path passes the LDM scheduler's per-step sigma here.
    degrade_sigma: float | list[float] | Tensor = 0.0
    seed: int = 0
    student_t_list: list[float] = field(default_factory=lambda: list(_STUDENT_T_LIST))


class PiDDecoder:
    """High-level decoder that hides PidNet construction and sampling.

    Usage::

        net = load_pid_decoder(state_dict, backbone)
        net = net.to(device=..., dtype=...)
        decoder = PiDDecoder(net, backbone=BaseModelType.Flux)
        image = decoder.decode(latent=..., caption_embs=...)
    """

    def __init__(self, net: PidNet, backbone: BaseModelType) -> None:
        if backbone not in _PER_BACKBONE:
            raise ValueError(f"Unsupported PiD backbone: {backbone!r}")
        self.net = net
        self.backbone = backbone

    @property
    def sr_scale(self) -> int:
        return int(self.net.sr_scale)

    @property
    def latent_spatial_down_factor(self) -> int:
        return int(_PER_BACKBONE[self.backbone]["latent_spatial_down_factor"])

    @torch.no_grad()
    def decode(
        self,
        *,
        latent: Tensor,
        caption_embs: Tensor,
        caption_mask: Optional[Tensor] = None,
        config: Optional[PiDDecodeConfig] = None,
    ) -> Tensor:
        """Decode *latent* + *caption_embs* into a pixel tensor in [-1, 1].

        Args:
            latent: ``[B, C_lat, H_lat, W_lat]`` LQ latent (the LDM's x0
                output, scaled per the backbone's VAE convention).
            caption_embs: ``[B, T, 2304]`` Gemma-2-2b-it caption embeddings
                (output of `_encode_text_raw` upstream — InvokeAI callers
                produce this via `Gemma2EncoderLoader`).
            config: per-call sampling overrides; defaults to the released
                `res2k_sr4x_*_distill_4step` preset.

        Returns:
            ``[B, 3, H_lat * sr_scale * latent_spatial_down_factor,
                  W_lat * sr_scale * latent_spatial_down_factor]`` in [-1, 1].
        """
        cfg = config or PiDDecodeConfig()
        device = latent.device
        dtype = next(self.net.parameters()).dtype
        # On CUDA, always run the forward pass under bf16 autocast: matmuls and
        # convolutions execute in bf16 (fast + small activations), while
        # numerically sensitive reductions like RMSNorm stay in the parameter
        # dtype. PidNet is intentionally loaded in fp32 (see the loader) so
        # those reductions actually keep their precision.
        autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
        batch_size = latent.shape[0]

        # Spatial size of the noise tensor — the decoder operates in pixel
        # space at sr_scale * latent_spatial_down_factor times the latent.
        total_up = self.sr_scale * self.latent_spatial_down_factor
        img_h = int(latent.shape[-2] * total_up)
        img_w = int(latent.shape[-1] * total_up)

        gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
        noise = torch.randn(batch_size, 3, img_h, img_w, device=device, generator=gen, dtype=dtype)

        sigma = cfg.degrade_sigma
        if isinstance(sigma, Tensor):
            degrade_sigma_t = sigma.to(device=device, dtype=torch.float32).reshape(-1)
            if degrade_sigma_t.numel() == 1:
                degrade_sigma_t = degrade_sigma_t.expand(batch_size).contiguous()
        elif isinstance(sigma, (list, tuple)):
            degrade_sigma_t = torch.tensor(sigma, device=device, dtype=torch.float32)
        else:
            degrade_sigma_t = torch.full((batch_size,), float(sigma), device=device, dtype=torch.float32)
        if degrade_sigma_t.shape != (batch_size,):
            raise ValueError(
                f"degrade_sigma must broadcast to [B={batch_size}], got shape {tuple(degrade_sigma_t.shape)}"
            )

        caption_embs = caption_embs.to(device=device, dtype=dtype)
        if caption_mask is not None:
            caption_mask = caption_mask.to(device=device)
        lq_latent = latent.to(device=device, dtype=dtype)

        t_list = _get_t_list(device, num_steps=cfg.num_inference_steps)

        self.net.eval()
        x0 = _student_sample_loop(
            self.net,
            noise=noise,
            t_list=t_list,
            caption_embs=caption_embs,
            caption_mask=caption_mask,
            lq_latent=lq_latent,
            degrade_sigma=degrade_sigma_t,
            sample_type=cfg.sample_type,
            autocast_dtype=autocast_dtype,
            generator=gen,
        )
        return x0.clamp(-1, 1)


@torch.no_grad()
def encode_caption_for_pid(
    captions: list[str],
    *,
    tokenizer: "object",  # AutoTokenizer; typed loose to avoid importing transformers at module load
    encoder: "object",  # Gemma2Model
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    chi_prompt: str = PID_CHI_PROMPT,
    model_max_length: int = PID_MODEL_MAX_LENGTH,
) -> tuple[Tensor, Tensor]:
    """Mirror of `PixelDiTModel._encode_text_raw`.

    Prepends the chi-prompt, tokenises with right-padding, runs Gemma's
    `model` (the transformer stack without the LM head), and selects
    ``[CLS] + last (model_max_length - 1)`` tokens to yield a fixed
    ``[B, model_max_length, 2304]`` embedding plus the matching attention
    mask. The mask is critical: PidNet's joint attention zeros padded text
    tokens out via this mask. Without it the decoder treats all ~300 slots
    (including the padding) as valid caption tokens and produces a
    washed-out average image.
    """
    if not captions:
        raise ValueError("encode_caption_for_pid requires at least one caption.")
    n_chi_tokens = len(tokenizer.encode(chi_prompt)) if chi_prompt else 0
    prompts = [chi_prompt + c for c in captions]
    max_len = (n_chi_tokens + model_max_length - 2) if chi_prompt else model_max_length
    # PiD was trained with right-padding (see PixelDiTModel._load_text_encoder
    # upstream). Gemma2's tokenizer defaults to "left" which would push the
    # BOS token away from index 0 and shove pads into the slice the decoder
    # consumes — yielding a garbled caption embedding. We toggle the value
    # for the duration of this call and restore it afterwards so we don't
    # poison the shared cached tokenizer.
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "right"
        toks = tokenizer(
            prompts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
    finally:
        tokenizer.padding_side = old_padding_side
    hidden = encoder(toks.input_ids, toks.attention_mask)[0]
    select_idx = [0] + list(range(-(model_max_length - 1), 0))
    caption_embs = hidden[:, select_idx].to(dtype=dtype)
    # Cast to bool: HF tokenizers emit attention_mask as int64, but PidNet's
    # SDPA call (scaled_dot_product_attention) refuses any int dtype — it
    # requires bool or matching float. Bool also matches the upstream
    # `pad = mask == 0` reduction in pid_net.py.
    caption_mask = toks.attention_mask[:, select_idx].to(torch.bool)
    return caption_embs, caption_mask


__all__ = [
    "PID_CHI_PROMPT",
    "PID_MODEL_MAX_LENGTH",
    "PID_NEGATIVE_PROMPT",
    "PiDDecodeConfig",
    "PiDDecoder",
    "build_pid_net",
    "encode_caption_for_pid",
    "load_pid_decoder",
]
