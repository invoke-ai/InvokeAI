"""Windowed-sinc audio resampling for model soundtrack conditioning.

Two families feed a soundtrack to a model at a rate it was trained on -- MiniMax H3's reference
clips and LTX-2's audio conditioning -- and both need the same resampler, so it lives here rather
than under either one.

MiniMax H3's reference implementation resamples soundtracks with
``torchaudio.transforms.Resample`` at its defaults (``lowpass_filter_width=6``,
``rolloff=0.99``, ``sinc_interp_hann``). torchaudio is not an InvokeAI dependency, so this is
a dependency-free port of exactly that path — ``_get_sinc_resample_kernel`` and
``_apply_sinc_resample_kernel`` from ``torchaudio.functional`` (BSD-2-Clause, Copyright (c)
2017 Facebook Inc.; https://github.com/pytorch/audio) — kept operation-for-operation
identical, mixed-dtype quirks included, so the conditioning matches upstream bit for bit.

The linear resampler in :mod:`invokeai.app.util.video_audio` is NOT numerically equivalent
and must not be substituted on this path.
"""

import math

import torch

_LOWPASS_FILTER_WIDTH = 6
_ROLLOFF = 0.99


def _sinc_resample_kernel(orig_freq: int, new_freq: int) -> tuple[torch.Tensor, int]:
    """The torchaudio ``sinc_interp_hann`` kernel at default parameters, float32.

    Mirrors torchaudio's ``dtype=None`` path exactly: the sample index grid is float64, the
    output-phase grid starts as an integer arange whose division lands in float32 before the
    float64 promotion, and the finished kernel is cast to float32.
    """
    base_freq = min(orig_freq, new_freq) * _ROLLOFF
    width = math.ceil(_LOWPASS_FILTER_WIDTH * orig_freq / base_freq)

    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-_LOWPASS_FILTER_WIDTH, _LOWPASS_FILTER_WIDTH)

    window = torch.cos(t * math.pi / _LOWPASS_FILTER_WIDTH / 2) ** 2

    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale
    return kernels.to(dtype=torch.float32), width


def resample_sinc(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resample ``(channels, num_samples)`` float32 audio, matching torchaudio bit for bit.

    Args:
        waveform: The audio, any floating dtype, channels-first.
        orig_freq: The rate the waveform carries its samples at.
        new_freq: The rate to resample to.

    Returns:
        The resampled waveform, same channel count, ``ceil(num_samples * new / orig)`` samples
        (after frequency reduction by their gcd).
    """
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError(f"Sample rates must be positive, got {orig_freq} -> {new_freq}.")
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")
    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig_freq, new_freq = int(orig_freq) // gcd, int(new_freq) // gcd
    kernel, width = _sinc_resample_kernel(orig_freq, new_freq)
    kernel = kernel.to(waveform.dtype)

    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])
    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = int(torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long())
    resampled = resampled[..., :target_length]
    return resampled.view(shape[:-1] + resampled.shape[-1:])
