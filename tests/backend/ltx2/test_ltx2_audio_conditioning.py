"""Waveform -> LTX-2 audio latents: the log-mel front end the release does not ship separately."""

from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.ltx2.audio_conditioning import build_mel_transform, encode_audio_latents
from invokeai.backend.ltx2.constants import LTX2_AUDIO_LATENT_CHANNELS, LTX2_AUDIO_LATENT_MEL_BINS

# The released shapes: the vocoder analyses at 512/512 into 64 mel bands, and the audio VAE mels
# its own 16 kHz rate at a hop of 160 -- 100 mel frames a second, 25 latents after its downscale.
FILTER_LENGTH = 512
MEL_CHANNELS = 64
VAE_HOP = 160
VAE_RATE = 16000


def _vocoder(hop_length: int = 80, out_channels: int = 2) -> SimpleNamespace:
    """A vocoder whose filters carry recognisable values, as a checkpoint-loaded one would."""
    from diffusers.pipelines.ltx2.vocoder import MelSTFT

    mel_stft = MelSTFT(
        filter_length=FILTER_LENGTH, hop_length=hop_length, window_length=FILTER_LENGTH, num_mel_channels=MEL_CHANNELS
    )
    with torch.no_grad():
        mel_stft.mel_basis.copy_(torch.full_like(mel_stft.mel_basis, 0.25))
        mel_stft.stft_fn.forward_basis.copy_(torch.full_like(mel_stft.stft_fn.forward_basis, 0.5))
        mel_stft.stft_fn.inverse_basis.copy_(torch.full_like(mel_stft.stft_fn.inverse_basis, 0.75))

    return SimpleNamespace(
        config=SimpleNamespace(
            filter_length=FILTER_LENGTH,
            num_mel_channels=MEL_CHANNELS,
            out_channels=out_channels,
            window_length=FILTER_LENGTH,
        ),
        mel_stft=mel_stft,
    )


class _AudioVaeStub:
    """Records the log-mel it was handed and returns a latent of the matching length."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(mel_hop_length=VAE_HOP, sample_rate=VAE_RATE)
        self.latents_mean = torch.full((LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS,), 2.0)
        self.latents_std = torch.full((LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS,), 4.0)
        self.log_mel: torch.Tensor | None = None

    def parameters(self):
        return iter([torch.zeros(1)])

    def buffers(self):
        # `get_effective_device` walks parameters and buffers to find where the model really is.
        return iter([])

    def encode(self, log_mel: torch.Tensor):
        self.log_mel = log_mel
        # The released VAE compresses mel frames 2x in time and the mel bins to 16.
        latents = torch.arange(
            log_mel.shape[2] // 2 * LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS, dtype=torch.float32
        )
        latents = latents.view(1, LTX2_AUDIO_LATENT_CHANNELS, log_mel.shape[2] // 2, LTX2_AUDIO_LATENT_MEL_BINS)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latents))


def test_a_rebuilt_mel_transform_carries_the_checkpoints_filters_rather_than_zeros() -> None:
    """Both bases are persistent buffers a checkpoint fills; a fresh module registers them as
    zeros, which emits the log floor for every input. Re-striding must copy, not reconstruct."""
    from diffusers.pipelines.ltx2.vocoder import MelSTFT

    source = _vocoder(hop_length=320)
    fresh = MelSTFT(
        filter_length=FILTER_LENGTH, hop_length=VAE_HOP, window_length=FILTER_LENGTH, num_mel_channels=MEL_CHANNELS
    )
    assert float(fresh.mel_basis.abs().sum()) == 0.0
    assert float(fresh.stft_fn.forward_basis.abs().sum()) == 0.0

    rebuilt = build_mel_transform(source, hop_length=VAE_HOP)

    assert torch.equal(rebuilt.mel_basis, source.mel_stft.mel_basis)
    assert torch.equal(rebuilt.stft_fn.forward_basis, source.mel_stft.stft_fn.forward_basis)
    assert torch.equal(rebuilt.stft_fn.inverse_basis, source.mel_stft.stft_fn.inverse_basis)


def test_the_rebuilt_transform_runs_at_the_vaes_hop_and_not_the_vocoders() -> None:
    """The hop is the only thing that changes, and it is what sets the latent rate: at the
    vocoder's own hop the result would be twice as many latents as the packing accepts."""
    source = _vocoder(hop_length=VAE_HOP * 2)
    waveform = torch.randn(1, VAE_HOP * 200)

    at_vae_hop = build_mel_transform(source, hop_length=VAE_HOP)(waveform)[0]
    at_vocoder_hop = source.mel_stft(waveform)[0]

    assert at_vae_hop.shape[2] == 200
    assert at_vocoder_hop.shape[2] == 100


def test_a_mono_recording_is_widened_to_the_channels_the_model_takes() -> None:
    vae = _AudioVaeStub()
    encode_audio_latents(vae, _vocoder(), torch.randn(1, VAE_HOP * 50), sample_rate=VAE_RATE)

    assert vae.log_mel is not None
    assert vae.log_mel.shape[1] == 2


def test_a_recording_with_more_channels_than_the_model_takes_is_refused_by_count() -> None:
    """Down-mixing is a decision with an audible result; making it silently would condition the
    generation on a soundtrack the user never heard."""
    with pytest.raises(ValueError, match="6 channels"):
        encode_audio_latents(_AudioVaeStub(), _vocoder(), torch.randn(6, VAE_HOP * 50), sample_rate=VAE_RATE)


def test_a_waveform_at_another_rate_is_resampled_to_the_vaes_own() -> None:
    """The mel hop is in samples, so a 44.1 kHz recording read as if it were 48 kHz would shift
    every latent -- the picture would follow the soundtrack at the wrong speed."""
    vae = _AudioVaeStub()
    seconds = 2.0
    encode_audio_latents(vae, _vocoder(), torch.randn(2, int(44100 * seconds)), sample_rate=44100)

    assert vae.log_mel is not None
    # Frames are the resampled length over the hop, not the source length over the hop.
    assert vae.log_mel.shape[2] == pytest.approx(VAE_RATE * seconds / VAE_HOP, abs=1)


@pytest.mark.parametrize("extra_samples", [0, 1, VAE_HOP - 1])
def test_a_ragged_tail_is_padded_to_a_whole_hop_before_the_mel(extra_samples: int) -> None:
    """The VAE's temporal downscale fixes the latent count off the mel frame count; a partial hop
    would shift every latent after it."""
    vae = _AudioVaeStub()
    encode_audio_latents(vae, _vocoder(), torch.randn(2, VAE_HOP * 50 + extra_samples), sample_rate=VAE_RATE)

    assert vae.log_mel is not None
    assert vae.log_mel.shape[2] == (51 if extra_samples else 50)


def test_the_latents_come_back_packed_and_normalized_on_the_vaes_own_statistics() -> None:
    vae = _AudioVaeStub()

    latents = encode_audio_latents(vae, _vocoder(), torch.randn(2, VAE_HOP * 100), sample_rate=VAE_RATE)

    assert latents.shape == (1, 50, LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS)
    # The statistics are per packed row element, so they apply after packing, not before: the row
    # the model reads is (value - mean) / std with the stub's 2.0 and 4.0.
    raw = vae.encode(vae.log_mel).latent_dist.mode()
    expected = (raw.transpose(1, 2).flatten(2, 3) - 2.0) / 4.0
    assert torch.allclose(latents, expected)


def test_a_waveform_that_is_not_channels_by_samples_is_refused() -> None:
    with pytest.raises(ValueError, match="channels, samples"):
        encode_audio_latents(_AudioVaeStub(), _vocoder(), torch.randn(VAE_HOP * 50), sample_rate=VAE_RATE)


def test_a_vae_that_returns_the_wrong_audio_shape_is_named_rather_than_packed() -> None:
    vae = _AudioVaeStub()
    vae.encode = lambda log_mel: SimpleNamespace(  # type: ignore[method-assign]
        latent_dist=SimpleNamespace(mode=lambda: torch.zeros(1, 4, 50, 8))
    )

    with pytest.raises(ValueError, match="4 channels x 8 mel bins"):
        encode_audio_latents(vae, _vocoder(), torch.randn(2, VAE_HOP * 100), sample_rate=VAE_RATE)
