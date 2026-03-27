import torch

from invokeai.invocation_api import (
    linear_srgb_from_oklab,
    linear_srgb_from_oklch,
    linear_srgb_from_srgb,
    oklab_from_linear_srgb,
    oklab_from_oklch,
    oklch_from_oklab,
    srgb_from_linear_srgb,
)


def test_srgb_oklab_round_trip() -> None:
    srgb = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.1]],
            [[0.0, 1.0], [0.0, 0.6]],
            [[0.0, 1.0], [0.0, 0.9]],
        ],
        dtype=torch.float32,
    )

    round_tripped = srgb_from_linear_srgb(linear_srgb_from_oklab(oklab_from_linear_srgb(linear_srgb_from_srgb(srgb))))

    assert torch.allclose(round_tripped, srgb, atol=1e-5)


def test_oklab_from_pure_srgb_red_matches_reference_value() -> None:
    srgb_red = torch.tensor([[[1.0]], [[0.0]], [[0.0]]], dtype=torch.float32)

    oklab_red = oklab_from_linear_srgb(linear_srgb_from_srgb(srgb_red))

    assert torch.allclose(
        oklab_red[:, 0, 0],
        torch.tensor([0.62795536, 0.22486306, 0.1258463], dtype=torch.float32),
        atol=1e-6,
    )


def test_oklab_oklch_round_trip() -> None:
    oklab = torch.tensor(
        [
            [[0.6, 0.4]],
            [[0.2, -0.1]],
            [[0.1, 0.05]],
        ],
        dtype=torch.float32,
    )

    round_tripped = oklab_from_oklch(oklch_from_oklab(oklab))

    assert torch.allclose(round_tripped, oklab, atol=1e-6)


def test_srgb_oklch_round_trip() -> None:
    srgb = torch.tensor(
        [
            [[0.2, 0.9]],
            [[0.4, 0.3]],
            [[0.8, 0.1]],
        ],
        dtype=torch.float32,
    )

    round_tripped = srgb_from_linear_srgb(
        linear_srgb_from_oklch(oklch_from_oklab(oklab_from_linear_srgb(linear_srgb_from_srgb(srgb))))
    )

    assert torch.allclose(round_tripped, srgb, atol=1e-5)


def test_linear_srgb_from_oklch_matches_oklab_path() -> None:
    oklch = torch.tensor(
        [
            [[0.7, 0.5]],
            [[0.12, 0.04]],
            [[30.0, 210.0]],
        ],
        dtype=torch.float32,
    )

    direct = linear_srgb_from_oklch(oklch)
    via_oklab = linear_srgb_from_oklab(oklab_from_oklch(oklch))

    assert torch.allclose(direct, via_oklab, atol=1e-6)
    assert direct.shape == (3, 1, 2)
