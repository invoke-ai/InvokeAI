"""Tests for the vendored taeh3 preview decoder (structure, IO contract, loader)."""

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.minimax_h3.taehv_decoder import TAEH3_LATENT_CHANNELS, TAEH3Decoder


@pytest.fixture(scope="module")
def decoder() -> TAEH3Decoder:
    torch.manual_seed(0)
    model = TAEH3Decoder()
    model.eval()
    return model


def test_decode_shapes_and_range(decoder):
    """[N, 24, T, h, w] -> [N, 4T, 3, 16h, 16w], clamped to [0, 1]."""
    latents = torch.randn(1, TAEH3_LATENT_CHANNELS, 2, 4, 6, generator=torch.Generator().manual_seed(1))
    frames = decoder.decode(latents)
    assert frames.shape == (1, 8, 3, 64, 96)
    assert frames.min() >= 0.0 and frames.max() <= 1.0


def test_decode_preview_frame_is_last(decoder):
    latents = torch.randn(1, TAEH3_LATENT_CHANNELS, 2, 4, 4, generator=torch.Generator().manual_seed(2))
    frame = decoder.decode_preview_frame(latents)
    assert frame.shape == (3, 64, 64)
    assert torch.equal(frame, decoder.decode(latents)[0, -1])


def test_decode_rejects_bad_shapes(decoder):
    with pytest.raises(ValueError, match="Expected"):
        decoder.decode(torch.randn(1, 16, 2, 4, 4))
    with pytest.raises(ValueError, match="Expected"):
        decoder.decode(torch.randn(TAEH3_LATENT_CHANNELS, 2, 4, 4))


def test_state_dict_matches_released_checkpoint_layout(decoder):
    """Pin the nn.Sequential indices/shapes against the released taeh3.safetensors layout.

    These keys and shapes were read from the file at the pinned commit; if this test fails, the
    vendored structure has drifted and the real checkpoint will no longer strict-load.
    """
    sd = decoder.state_dict()
    expected = {
        "decoder.1.weight": (256, 24, 3, 3),
        "decoder.3.conv.0.weight": (256, 512, 3, 3),
        "decoder.7.conv.weight": (256, 256, 1, 1),  # TGrow stride 1
        "decoder.8.weight": (128, 256, 3, 3),
        "decoder.13.conv.weight": (256, 128, 1, 1),  # TGrow stride 2
        "decoder.14.weight": (64, 128, 3, 3),
        "decoder.19.conv.weight": (128, 64, 1, 1),  # TGrow stride 2
        "decoder.20.weight": (64, 64, 3, 3),
        "decoder.22.weight": (12, 64, 3, 3),  # 3 RGB x 2x2 pixel-shuffle patch
    }
    for key, shape in expected.items():
        assert key in sd, f"missing {key}"
        assert tuple(sd[key].shape) == shape, f"{key}: {tuple(sd[key].shape)} != {shape}"
    # 4x temporal upscale total: product of TGrow strides (1, 2, 2).
    assert sd["decoder.13.conv.weight"].shape[0] // sd["decoder.13.conv.weight"].shape[1] == 2
    assert sd["decoder.19.conv.weight"].shape[0] // sd["decoder.19.conv.weight"].shape[1] == 2


def test_load_model_ignores_encoder_keys(tmp_path, decoder):
    """The released file bundles an encoder; load_model must strict-load only decoder.* keys."""
    sd = {k: v.to(torch.float16) for k, v in decoder.state_dict().items()}
    sd["encoder.1.weight"] = torch.zeros(64, 12, 3, 3, dtype=torch.float16)
    path = tmp_path / "taeh3.safetensors"
    save_file(sd, str(path))

    loaded = TAEH3Decoder.load_model(path)
    assert not loaded.training
    assert loaded.decoder[1].weight.dtype == torch.float16
    frames = loaded.decode(torch.randn(1, TAEH3_LATENT_CHANNELS, 1, 4, 4))
    assert frames.shape == (1, 4, 3, 64, 64)


def test_load_model_rejects_missing_keys(tmp_path, decoder):
    sd = {k: v.to(torch.float16) for k, v in decoder.state_dict().items()}
    sd.pop("decoder.22.weight")
    path = tmp_path / "incomplete.safetensors"
    save_file(sd, str(path))
    with pytest.raises(RuntimeError, match="Missing"):
        TAEH3Decoder.load_model(path)
