"""Triage for non-finite VAE decode output ("black image" generations).

On a dual-GPU ROCm rig, Wan-VAE (Anima) image decodes intermittently start returning
all-NaN output while a long video generation runs on the other GPU, and stay broken until
the video finishes. NaN survives the final ``clamp(-1, 1)`` and quantizes to 0, so every
affected generation is a solid black image with no error anywhere in the log.

The failure has at least three candidate mechanisms that ordinary logs cannot tell apart:

1. The latents entering the decode are already non-finite (corruption upstream, in the
   denoise or the tensor transfer) — the VAE is innocent.
2. The VAE's cached weights are corrupt (a bad RAM->VRAM move, or a corrupted canonical CPU
   copy) — every decode fails until the model cache entry is dropped and reloaded.
3. The weights and latents are clean but the decode *compute* produces NaN — an
   allocator-state-dependent kernel failure of the kind already seen on gfx1100 (the ROCm
   fused-SDPA head-dim defect), plausibly surfaced by the multi-GPU policy of skipping
   every ``empty_cache`` while a peer device is mid-session: during an hours-long video
   generation the image GPU's allocator runs in a never-emptied, maximally-grown state.

These helpers gather the discriminating evidence at the moment of failure — latent
finiteness is checked by the caller, weight state and allocator state here — and support a
bounded recovery attempt whose outcome is itself the fingerprint: a decode healed by a
*forced real* ``empty_cache`` (mechanism 3) versus one only healed by a weight reload
(mechanism 2).
"""

from dataclasses import dataclass, field

import torch

from invokeai.backend.util.devices import TorchDevice


@dataclass
class NonFiniteWeightScan:
    """Result of scanning a module's parameters/buffers for NaN/Inf."""

    tensors_scanned: int = 0
    # Fully-qualified names of tensors containing at least one non-finite value, capped by the
    # scanner to keep log lines bounded.
    nonfinite_names: list[str] = field(default_factory=list)
    nonfinite_on_compute: int = 0
    nonfinite_on_cpu: int = 0

    @property
    def clean(self) -> bool:
        return self.nonfinite_on_compute == 0 and self.nonfinite_on_cpu == 0

    def describe(self) -> str:
        if self.clean:
            return f"all {self.tensors_scanned} weight tensors finite"
        names = ", ".join(self.nonfinite_names[:8])
        suffix = ", ..." if len(self.nonfinite_names) > 8 else ""
        return (
            f"{self.nonfinite_on_compute} weight tensor(s) with NaN/Inf on the compute device and "
            f"{self.nonfinite_on_cpu} on the CPU, of {self.tensors_scanned} scanned ({names}{suffix})"
        )


@torch.no_grad()
def scan_module_for_nonfinite_weights(module: torch.nn.Module, compute_device_type: str) -> NonFiniteWeightScan:
    """Scan every parameter and buffer of ``module`` for NaN/Inf values.

    Cheap relative to a decode (one reduction per tensor; a Wan VAE is a few hundred MB), and
    only ever run after a decode has already produced non-finite output.

    Floating tensors only: integer tensors cannot hold NaN/Inf and ``isfinite`` on them is
    wasted work (quantized models keep int8 weights).
    """
    scan = NonFiniteWeightScan()
    seen: set[int] = set()
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        # Shared/tied tensors appear under multiple names; count each storage once. data_ptr
        # catches ties that wrap the same storage in distinct tensor objects; fall back to
        # object identity for tensors without an addressable storage (data_ptr() is 0 on meta
        # and zero-element tensors — one shared key there would hide all but the first).
        try:
            key = tensor.data_ptr() or id(tensor)
        except RuntimeError:
            key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        if not tensor.is_floating_point():
            continue
        scan.tensors_scanned += 1
        try:
            if bool(torch.isfinite(tensor).all()):
                continue
        except Exception:
            # A tensor that cannot even be read (sick context, meta device, a dtype or wrapper
            # subclass without isfinite — those raise TypeError, not RuntimeError) is itself a
            # finding; report it rather than aborting the whole scan.
            pass
        scan.nonfinite_names.append(f"{name}@{tensor.device}")
        if tensor.device.type == compute_device_type:
            scan.nonfinite_on_compute += 1
        else:
            scan.nonfinite_on_cpu += 1
    return scan


def allocator_state_summary(device: torch.device) -> str:
    """One-line caching-allocator + peer-activity summary for ``device``.

    Includes whether another generation device is mid-session — i.e. whether every
    ``empty_cache`` on this device has been getting skipped by the peer-aware policy.
    """
    peer_busy = TorchDevice._another_generation_device_busy()
    if device.type != "cuda" or not torch.cuda.is_available():
        return f"device={device}, peer_generation_device_busy={peer_busy}"
    try:
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        stats = torch.cuda.memory_stats(device)
        inactive_split = int(stats.get("inactive_split_bytes.all.current", 0))
        segments = int(stats.get("segment.all.current", 0))
        mb = 2**20
        return (
            f"device={device}, reserved={reserved / mb:.0f}MB, allocated={allocated / mb:.0f}MB, "
            f"inactive_split={inactive_split / mb:.0f}MB, segments={segments}, "
            f"peer_generation_device_busy={peer_busy}"
        )
    except Exception as e:
        return f"device={device}, allocator stats unavailable ({type(e).__name__}), peer_generation_device_busy={peer_busy}"


def force_real_empty_cache() -> None:
    """Run the REAL ``torch.cuda.empty_cache``, bypassing the peer-aware skip.

    ``install_peer_aware_empty_cache`` rebinds ``torch.cuda.empty_cache`` with a wrapper that
    skips while any other generation device is mid-session. This helper reaches through the
    wrapper (``functools.wraps`` preserves the original as ``__wrapped__``) for the one case
    where the skip is the wrong trade: recovering a decode that is producing garbage, where a
    single peer-convoying cache flush is strictly better than a black image. Callers must log
    that they did this — a peer device's in-flight step will stall until this returns.
    """
    real = getattr(torch.cuda.empty_cache, "__wrapped__", torch.cuda.empty_cache)
    # A real, process-global release satisfies any release a peer-aware skip deferred.
    TorchDevice._empty_cache_deferred.clear()
    real()


def nonfinite_fraction(tensor: torch.Tensor) -> float:
    """Fraction of ``tensor``'s elements that are NaN/Inf."""
    if tensor.numel() == 0:
        return 0.0
    return float((~torch.isfinite(tensor)).sum().item()) / tensor.numel()
