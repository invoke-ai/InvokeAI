"""Reading a TAESD checkpoint: which decoder it holds, in any published layout, and what it is not."""

import pytest
import torch

from invokeai.backend.tiny_autoencoder.checkpoint import (
    TaesdLayout,
    build_taesd_decoder,
    detect_taesd_layout,
    is_tinier_taesd,
    taesd_decoder_state_dict,
)
from tests.fixtures.tiny_autoencoders import (
    TAEF2,
    TAEQI2_1,
    TAESD,
    TAESD3,
    TAEW2_1,
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


def test_a_taehv_checkpoint_is_detected_from_shapes_alone() -> None:
    assert detect_taesd_layout(_on_meta(taehv_weights())) == TAEW2_1


def test_a_taehv_checkpoint_cuts_only_the_tgrow_after_the_last_memblock_to_its_last_timestep() -> None:
    """The image decoder keeps the last frame once no MemBlock follows; earlier TGrows keep their frames."""
    combined = taehv_weights()

    converted = taesd_decoder_state_dict(combined)

    assert not any(key.startswith(("encoder.", "decoder.")) for key in converted)
    decoder = build_taesd_decoder(TAEW2_1)
    decoder.load_state_dict(converted, strict=True)
    torch.testing.assert_close(decoder[19].conv.weight, combined["decoder.19.conv.weight"][64:], rtol=0, atol=0)
    for index in (7, 13):
        weight = combined[f"decoder.{index}.conv.weight"]
        torch.testing.assert_close(decoder[index].conv.weight, weight, rtol=0, atol=0)
    assert decoder[13].conv.weight.shape == (256, 128, 1, 1)


def test_a_converted_taehv_state_dict_is_kept_as_it_is() -> None:
    """The loader converts, then identifies the converted dict: a second pass must not cut the `TGrow`s again."""
    converted = taesd_decoder_state_dict(taehv_weights())

    assert detect_taesd_layout(converted) == TAEW2_1
    assert taesd_decoder_state_dict(converted) == converted


def _taehv_patch_2() -> dict[str, torch.Tensor]:
    """taeh3 / taew2_2 / taehv1_5's 2x2 pixel-shuffle head (12 output channels), at taew2_1's latent width."""
    return _on_meta(taehv_weights(patch_size=2))


def _taehv_truncated() -> dict[str, torch.Tensor]:
    sd = _on_meta(taehv_weights())
    del sd["decoder.17.conv.4.bias"]
    return sd


def _taehv_extended() -> dict[str, torch.Tensor]:
    sd = _on_meta(taehv_weights())
    sd["decoder.23.weight"] = torch.empty(3, 3, 3, 3, device="meta")
    return sd


def _taehv_super() -> dict[str, torch.Tensor]:
    """A `_super` decoder opens with a 1x1 conv into SuperMemBlocks: `decoder.0.weight`, no `decoder.1.weight`."""
    return {
        "decoder.0.weight": torch.empty(512, 16, 1, 1, device="meta"),
        "decoder.1.conv.0.weight": torch.empty(1024, 1, 7, 7, device="meta"),
        "decoder.22.weight": torch.empty(3, 64, 3, 3, device="meta"),
    }


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


@pytest.mark.parametrize(
    "make",
    [
        _vae_keys,
        _taehv_keys,
        _taehv_patch_2,
        _taehv_truncated,
        _taehv_extended,
        _taehv_super,
        _truncated,
        _extended,
        _fingerprint_only,
        dict,
    ],
)
def test_what_is_not_exactly_a_taesd_decoder_is_not_detected(make) -> None:
    sd = make()

    assert detect_taesd_layout(sd) is None
    assert not is_tinier_taesd(sd)
