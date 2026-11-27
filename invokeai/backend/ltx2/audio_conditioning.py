"""Waveform -> LTX-2 audio latents, the inverse of the soundtrack decode.

Generation decodes ``latents -> audio VAE -> log-mel -> vocoder -> waveform``. Conditioning on an
existing soundtrack needs the first half of that run backwards, and the log-mel step is the piece
the release does not ship as a standalone module: the audio VAE carries no filterbank (its file has
no mel, stft or basis tensor), and Lightricks' own front end is not published with the weights.

It does not have to be. The vocoder's bandwidth-extension stage takes the mel of the signal at its
*input* rate and feeds it to a generator built like the vocoder itself, so ``vocoder.mel_stft``
computes exactly the representation the vocoder consumes -- which is what the audio VAE decodes to.
Its ``config.input_sampling_rate`` is the audio VAE's own ``sample_rate`` (16 kHz in the release;
the 48 kHz on the vocoder is its *output*), so the filterbank is fitted for the rate this module
feeds it. Both of its bases are checkpoint-loaded buffers (``CausalSTFT.forward_basis``, the
windowed analysis kernels, and ``MelSTFT.mel_basis``, the filterbank), and neither depends on the
hop, which enters only as a ``conv1d`` stride. So the transform this module needs is the released
one run at the audio VAE's own ``mel_hop_length`` -- the release's learned content, at the rate the
VAE was built for.

Measured against the released weights, and pinned by
``test_a_recording_encodes_to_the_audio_rate_the_transformer_reads_and_survives_the_round_trip``:
at the VAE's hop the result is 25 audio latents per second, which is the rate the transformer's
audio stream uses everywhere else, and the mel this module produces for a vocoder output matches
the mel the VAE decoded to produce it at 0.96 Pearson, in the same shape -- the residual being the
vocoder's own phase reconstruction, not a difference of domain. At the vocoder's own hop the rate
would be 50 per second, which the packing would reject.
"""

import torch

from invokeai.backend.ltx2.constants import LTX2_AUDIO_LATENT_CHANNELS, LTX2_AUDIO_LATENT_MEL_BINS
from invokeai.backend.ltx2.packing import normalize_audio_latents, pack_audio_latents
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.audio_resample import resample_sinc


def build_mel_transform(vocoder: torch.nn.Module, *, hop_length: int) -> torch.nn.Module:
    """The released log-mel transform, re-strided to ``hop_length``.

    A fresh ``MelSTFT`` registers both of its bases as zeros -- they are persistent buffers the
    checkpoint fills -- so one built without copying them produces silence at the log floor for
    every input. They are copied here rather than recomputed: a filterbank fitted to the model is
    not something to re-derive from a formula and hope the conventions match.
    """
    from diffusers.pipelines.ltx2.vocoder import MelSTFT

    source = vocoder.mel_stft
    transform = MelSTFT(
        filter_length=vocoder.config.filter_length,
        hop_length=hop_length,
        window_length=vocoder.config.window_length,
        num_mel_channels=vocoder.config.num_mel_channels,
    )
    with torch.no_grad():
        transform.mel_basis.copy_(source.mel_basis)
        transform.stft_fn.forward_basis.copy_(source.stft_fn.forward_basis)
        transform.stft_fn.inverse_basis.copy_(source.stft_fn.inverse_basis)

    return transform.to(device=source.mel_basis.device, dtype=source.mel_basis.dtype).eval()


@torch.no_grad()
def encode_audio_latents(
    audio_vae: torch.nn.Module,
    vocoder: torch.nn.Module,
    waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> torch.Tensor:
    """A ``(channels, samples)`` waveform -> packed, normalized audio latents ``(1, L, 128)``.

    ``waveform`` is float PCM in [-1, 1] at ``sample_rate``; it is resampled to the VAE's own rate
    with the same windowed-sinc path the other conditioning soundtracks use.
    """
    if waveform.ndim != 2:
        raise ValueError(f"Audio conditioning expects a (channels, samples) waveform; got {tuple(waveform.shape)}.")

    target_rate = int(audio_vae.config.sample_rate)
    hop = int(audio_vae.config.mel_hop_length)
    channels = int(vocoder.config.out_channels)

    audio = waveform.to(dtype=torch.float32)
    if audio.shape[0] == 1 and channels > 1:
        audio = audio.expand(channels, -1)
    elif audio.shape[0] > channels:
        # Down-mixing is the caller's business; refusing here names the mismatch rather than
        # silently conditioning on an arbitrary subset of the channels.
        raise ValueError(f"This soundtrack has {audio.shape[0]} channels; the model takes {channels}.")

    if sample_rate != target_rate:
        audio = resample_sinc(audio, sample_rate, target_rate)

    # The mel frame count is exact only on a whole number of hops, and the VAE's temporal downscale
    # then fixes the latent count -- a ragged tail would shift every latent after it.
    remainder = audio.shape[-1] % hop
    if remainder:
        audio = torch.nn.functional.pad(audio, (0, hop - remainder))

    mel_transform = build_mel_transform(vocoder, hop_length=hop)
    device = mel_transform.mel_basis.device
    mel_dtype = mel_transform.mel_basis.dtype
    log_mel, _, _, _ = mel_transform(audio[None].to(device=device, dtype=mel_dtype).flatten(0, 1))
    log_mel = log_mel.unflatten(0, (1, audio.shape[0])).transpose(2, 3)  # (1, C, frames, mel bins)

    # The audio VAE's own device, not the mel transform's: the two models are locked by independent
    # `model_on_device` calls and a partial load can leave them on different ones.
    vae_device = get_effective_device(audio_vae)
    vae_dtype = next(iter(audio_vae.parameters())).dtype
    latents = audio_vae.encode(log_mel.to(device=vae_device, dtype=vae_dtype)).latent_dist.mode().to(torch.float32)

    expected = (LTX2_AUDIO_LATENT_CHANNELS, LTX2_AUDIO_LATENT_MEL_BINS)
    if (latents.shape[1], latents.shape[3]) != expected:
        raise ValueError(
            f"The audio VAE produced {latents.shape[1]} channels x {latents.shape[3]} mel bins; "
            f"LTX-2's audio stream is {expected[0]} x {expected[1]}."
        )

    packed = pack_audio_latents(latents.cpu())
    return normalize_audio_latents(packed, audio_vae.latents_mean.cpu(), audio_vae.latents_std.cpu())
