"""Reading a TAE checkpoint: which decoder it holds, in any published layout, and what it is not."""

import pytest
import torch

from invokeai.backend.tiny_autoencoder.checkpoint import (
    TAELTX2_3,
    TAEW2_1,
    TAEW2_2,
    TaehvLayout,
    TaesdLayout,
    build_taesd_decoder,
    build_tiny_autoencoder_decoder,
    detect_taehv_layout,
    detect_taesd_layout,
    detect_tiny_autoencoder_layout,
    is_tinier_taesd,
    taesd_decoder_state_dict,
    tiny_autoencoder_state_dict,
)
from tests.fixtures.tiny_autoencoders import (
    TAEF2,
    TAEQI2_1,
    TAESD,
    TAESD3,
    diffusers_weights,
    native_weights,
    taehv_weights,
    tinier_weights,
)

LAYOUTS = [TAESD, TAESD3, TAEF2, TAEQI2_1]


def _on_meta(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """What `ModelOnDisk.load_state_dict` hands identification: shapes, no data."""
    return {key: torch.empty(value.shape, device="meta") for key, value in sd.items()}


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda layout: f"{layout.arch}-{layout.latent_channels}")
def test_both_layouts_are_detected_from_shapes_alone(layout: TaesdLayout) -> None:
    native = native_weights(layout)

    assert detect_taesd_layout(_on_meta(native)) == layout
    assert detect_taesd_layout(_on_meta(diffusers_weights(native))) == layout


@pytest.mark.parametrize("layout", [TAESD3, TAEF2, TAEQI2_1], ids=["default", "flux_2", "f16"])
def test_diffusers_layout_converts_to_the_native_decoder(layout: TaesdLayout) -> None:
    native = native_weights(layout)

    converted = taesd_decoder_state_dict(diffusers_weights(native))

    assert converted.keys() == native.keys()
    assert all(converted[key] is native[key] for key in native)
    decoder = build_taesd_decoder(layout)
    decoder.load_state_dict(converted, strict=True)


def test_native_layout_is_kept_as_it_is() -> None:
    native = native_weights(TAESD)

    kept = taesd_decoder_state_dict(native)

    assert kept.keys() == native.keys()
    assert all(kept[key] is native[key] for key in native)


def test_tinier_taeqi2_1_is_recognised_but_not_detected_as_a_decoder() -> None:
    tinier = _on_meta(tinier_weights())

    assert detect_taesd_layout(tinier) is None
    assert is_tinier_taesd(tinier)
    assert not is_tinier_taesd(_on_meta(diffusers_weights(native_weights(TAEQI2_1))))


def _vae_keys() -> dict[str, torch.Tensor]:
    return {
        "decoder.conv_in.weight": torch.empty(512, 4, 3, 3, device="meta"),
        "decoder.conv_out.weight": torch.empty(3, 128, 3, 3, device="meta"),
        "encoder.conv_in.weight": torch.empty(128, 3, 3, 3, device="meta"),
    }


def _taehv_keys() -> dict[str, torch.Tensor]:
    """TAEHV (taeh3) keys its decoder Sequential as `decoder.N` with no `layers`."""
    return {
        "decoder.1.weight": torch.empty(256, 24, 3, 3, device="meta"),
        "decoder.3.conv.0.weight": torch.empty(256, 512, 3, 3, device="meta"),
        "decoder.22.weight": torch.empty(12, 64, 3, 3, device="meta"),
    }


def _truncated() -> dict[str, torch.Tensor]:
    sd = native_weights(TAESD)
    del sd["19.bias"]
    return sd


def _extended() -> dict[str, torch.Tensor]:
    sd = native_weights(TAESD)
    sd["20.weight"] = torch.zeros(3, 3, 3, 3)
    return sd


def _fingerprint_only() -> dict[str, torch.Tensor]:
    return {"1.weight": torch.zeros(64, 4, 3, 3), "19.weight": torch.zeros(3, 64, 3, 3)}


@pytest.mark.parametrize("make", [_vae_keys, _taehv_keys, _truncated, _extended, _fingerprint_only, dict])
def test_what_is_not_exactly_a_taesd_decoder_is_not_detected(make) -> None:
    sd = make()

    assert detect_taesd_layout(sd) is None
    assert not is_tinier_taesd(sd)


TAEHV_LAYOUTS = [TAEW2_1, TAEW2_2, TAELTX2_3]


@pytest.mark.parametrize("layout", TAEHV_LAYOUTS, ids=["taew2_1", "taew2_2", "taeltx"])
def test_taehv_files_are_detected_from_shapes_alone_and_not_as_taesd(layout: TaehvLayout) -> None:
    sd = _on_meta(taehv_weights(layout))

    assert detect_taehv_layout(sd) == layout
    assert detect_tiny_autoencoder_layout(sd) == layout
    assert detect_taesd_layout(sd) is None


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda layout: f"{layout.arch}-{layout.latent_channels}")
def test_taesd_files_are_not_detected_as_taehv(layout: TaesdLayout) -> None:
    native = native_weights(layout)

    assert detect_taehv_layout(_on_meta(native)) is None
    assert detect_taehv_layout(_on_meta(diffusers_weights(native))) is None
    assert detect_tiny_autoencoder_layout(_on_meta(native)) == layout


def test_the_taehv_decoder_weights_are_everything_but_the_encoder() -> None:
    sd = taehv_weights(TAEW2_1)

    decoder_sd = tiny_autoencoder_state_dict(sd, TAEW2_1)

    assert not any(key.startswith("encoder.") for key in decoder_sd)
    build_tiny_autoencoder_decoder(TAEW2_1).load_state_dict(decoder_sd, strict=True)


def _taehv_with(**changes) -> dict[str, torch.Tensor]:
    """A TAEHV checkpoint of a network no supported base uses: `TaehvLayout` fields changed from taew2_1's."""
    return taehv_weights(TaehvLayout(**{**vars(TAEW2_1), **changes}))


def _truncated_taehv() -> dict[str, torch.Tensor]:
    sd = taehv_weights(TAEW2_2)
    del sd["decoder.17.conv.4.bias"]
    return sd


def _extended_taehv() -> dict[str, torch.Tensor]:
    sd = taehv_weights(TAEW2_2)
    sd["decoder.23.weight"] = torch.zeros(3, 3, 3, 3)
    return sd


@pytest.mark.parametrize(
    "make",
    [
        # taeh3: MiniMax H3 keeps its own preview path.
        lambda: _taehv_with(latent_channels=24, patch_size=2),
        # taehv1_5: 32 channels at 16x, which no architecture denoises in.
        lambda: _taehv_with(latent_channels=32, patch_size=2),
        # LTX's width with Wan's temporal strides.
        lambda: _taehv_with(latent_channels=128, patch_size=4),
        _truncated_taehv,
        _extended_taehv,
    ],
    ids=["taeh3", "taehv1_5", "ltx-width-wan-strides", "truncated", "extended"],
)
def test_taehv_networks_no_supported_base_uses_are_not_detected(make) -> None:
    assert detect_tiny_autoencoder_layout(_on_meta(make())) is None
