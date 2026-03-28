import pytest
import torch

from invokeai.backend.image_util import color_conversion
from invokeai.invocation_api import (
    hsl_from_linear_srgb,
    hsl_from_srgb,
    lab_from_linear_srgb,
    lab_from_srgb,
    lab_from_xyz,
    linear_srgb_from_hsl,
    linear_srgb_from_lab,
    linear_srgb_from_oklab,
    linear_srgb_from_oklch,
    linear_srgb_from_srgb,
    linear_srgb_from_xyz,
    okhsl_from_srgb,
    okhsv_from_srgb,
    oklab_from_linear_srgb,
    oklab_from_oklch,
    oklab_from_srgb,
    oklab_from_xyz,
    oklch_from_linear_srgb,
    oklch_from_oklab,
    oklch_from_srgb,
    oklch_from_xyz,
    srgb_from_hsl,
    srgb_from_lab,
    srgb_from_linear_srgb,
    srgb_from_okhsl,
    srgb_from_okhsv,
    srgb_from_oklab,
    srgb_from_oklch,
    srgb_from_xyz,
    xyz_d50_to_d65,
    xyz_d65_to_d50,
    xyz_from_lab,
    xyz_from_linear_srgb,
    xyz_from_oklab,
    xyz_from_oklch,
    xyz_from_srgb,
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


def test_oklab_from_srgb_matches_explicit_conversion_path() -> None:
    srgb = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.1]],
            [[0.0, 1.0], [0.0, 0.6]],
            [[0.0, 1.0], [0.0, 0.9]],
        ],
        dtype=torch.float32,
    )

    direct = oklab_from_srgb(srgb)
    via_linear_srgb = oklab_from_linear_srgb(linear_srgb_from_srgb(srgb))

    assert torch.allclose(direct, via_linear_srgb, atol=1e-6)


def test_srgb_from_oklab_matches_explicit_conversion_path() -> None:
    oklab = torch.tensor(
        [
            [[0.6, 0.4]],
            [[0.2, -0.1]],
            [[0.1, 0.05]],
        ],
        dtype=torch.float32,
    )

    direct = srgb_from_oklab(oklab)
    via_linear_srgb = srgb_from_linear_srgb(linear_srgb_from_oklab(oklab))

    assert torch.allclose(direct, via_linear_srgb, atol=1e-6)


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


def test_oklch_from_linear_srgb_matches_explicit_conversion_path() -> None:
    linear_srgb = torch.tensor(
        [
            [[0.1, 0.9]],
            [[0.4, 0.2]],
            [[0.7, 0.3]],
        ],
        dtype=torch.float32,
    )

    direct = oklch_from_linear_srgb(linear_srgb)
    via_oklab = oklch_from_oklab(oklab_from_linear_srgb(linear_srgb))

    assert torch.allclose(direct, via_oklab, atol=1e-6)


def test_oklch_from_srgb_and_back_round_trip() -> None:
    srgb = torch.tensor(
        [
            [[0.2, 0.9]],
            [[0.4, 0.3]],
            [[0.8, 0.1]],
        ],
        dtype=torch.float32,
    )

    direct_round_trip = srgb_from_oklch(oklch_from_srgb(srgb))
    explicit_round_trip = srgb_from_linear_srgb(
        linear_srgb_from_oklch(oklch_from_oklab(oklab_from_linear_srgb(linear_srgb_from_srgb(srgb))))
    )

    assert torch.allclose(direct_round_trip, srgb, atol=1e-5)
    assert torch.allclose(explicit_round_trip, srgb, atol=1e-5)


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


def test_hsl_srgb_round_trip() -> None:
    srgb = torch.tensor(
        [
            [[0.0, 1.0], [0.25, 0.9]],
            [[0.2, 0.8], [0.75, 0.1]],
            [[1.0, 0.1], [0.5, 0.4]],
        ],
        dtype=torch.float32,
    )

    round_tripped = srgb_from_hsl(hsl_from_srgb(srgb))

    assert torch.allclose(round_tripped, srgb, atol=1e-5)


def test_hsl_hue_is_expressed_in_degrees() -> None:
    srgb = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    hsl = hsl_from_srgb(srgb)

    assert torch.allclose(hsl[0, 0, :], torch.tensor([0.0, 120.0, 240.0]), atol=1e-3)
    assert torch.allclose(hsl[1, 0, :], torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(hsl[2, 0, :], torch.tensor([0.5, 0.5, 0.5]), atol=1e-6)


def test_hsl_from_grayscale_has_zero_saturation() -> None:
    srgb = torch.tensor(
        [
            [[0.1, 0.8]],
            [[0.1, 0.8]],
            [[0.1, 0.8]],
        ],
        dtype=torch.float32,
    )

    hsl = hsl_from_srgb(srgb)

    assert torch.allclose(hsl[1, ...], torch.zeros_like(hsl[1, ...]), atol=1e-6)


def test_hsl_from_linear_srgb_matches_explicit_conversion_path() -> None:
    linear_srgb = torch.tensor(
        [
            [[0.0, 1.0], [0.25, 0.8]],
            [[0.2, 0.8], [0.75, 0.1]],
            [[1.0, 0.1], [0.5, 0.4]],
        ],
        dtype=torch.float32,
    )

    direct = hsl_from_linear_srgb(linear_srgb)
    via_srgb = hsl_from_srgb(srgb_from_linear_srgb(linear_srgb))

    assert torch.allclose(direct, via_srgb, atol=1e-6)


def test_linear_srgb_from_hsl_matches_explicit_conversion_path() -> None:
    hsl = torch.tensor(
        [
            [[0.0, 216.0], [90.0, 324.0]],
            [[1.0, 0.25], [0.75, 0.1]],
            [[0.5, 0.4], [0.2, 0.8]],
        ],
        dtype=torch.float32,
    )

    direct = linear_srgb_from_hsl(hsl)
    via_srgb = linear_srgb_from_srgb(srgb_from_hsl(hsl))

    assert torch.allclose(direct, via_srgb, atol=1e-6)


def test_srgb_from_hsl_wraps_degree_hue_values() -> None:
    hsl = torch.tensor(
        [
            [[360.0, -120.0]],
            [[1.0, 1.0]],
            [[0.5, 0.5]],
        ],
        dtype=torch.float32,
    )

    rgb = srgb_from_hsl(hsl)

    assert torch.allclose(rgb[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(rgb[:, 0, 1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)


def test_okhsl_srgb_round_trip() -> None:
    srgb = torch.tensor(
        [
            [[0.05, 0.95], [0.2, 0.8]],
            [[0.4, 0.2], [0.6, 0.1]],
            [[0.9, 0.05], [0.3, 0.7]],
        ],
        dtype=torch.float32,
    )

    round_tripped = srgb_from_okhsl(okhsl_from_srgb(srgb))

    assert torch.allclose(round_tripped, srgb, atol=5e-4)


def test_okhsl_and_okhsv_hue_are_expressed_in_degrees() -> None:
    srgb = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    okhsl = okhsl_from_srgb(srgb)
    okhsv = okhsv_from_srgb(srgb)

    assert torch.allclose(okhsl[0, 0, :], torch.tensor([29.2473, 142.4848, 264.0487]), atol=2e-2)
    assert torch.allclose(okhsv[0, 0, :], torch.tensor([29.2473, 142.4848, 264.0487]), atol=2e-2)
    assert torch.allclose(okhsl[1, 0, :], torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(okhsv[1, 0, :], torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)


def test_okhsl_and_okhsv_srgb_round_trip() -> None:
    srgb = torch.tensor(
        [
            [[0.05, 0.95], [0.2, 0.8]],
            [[0.4, 0.2], [0.6, 0.1]],
            [[0.9, 0.05], [0.3, 0.7]],
        ],
        dtype=torch.float32,
    )

    okhsl_round_tripped = srgb_from_okhsl(okhsl_from_srgb(srgb))
    okhsv_round_tripped = srgb_from_okhsv(okhsv_from_srgb(srgb))

    assert torch.allclose(okhsl_round_tripped, srgb, atol=5e-4)
    assert torch.allclose(okhsv_round_tripped, srgb, atol=5e-4)


def test_okhsl_and_okhsv_outputs_keep_hue_in_degrees_and_other_channels_in_unit_range() -> None:
    srgb = torch.tensor(
        [
            [[0.0, 1.0]],
            [[1.0, 0.0]],
            [[0.5, 0.25]],
        ],
        dtype=torch.float32,
    )

    okhsl = okhsl_from_srgb(srgb)
    okhsv = okhsv_from_srgb(srgb)

    assert torch.all(okhsl[0, ...] >= 0.0)
    assert torch.all(okhsl[0, ...] <= 360.0)
    assert torch.all(okhsl[1:, ...] >= 0.0)
    assert torch.all(okhsl[1:, ...] <= 1.0)
    assert torch.all(okhsv[0, ...] >= 0.0)
    assert torch.all(okhsv[0, ...] <= 360.0)
    assert torch.all(okhsv[1:, ...] >= 0.0)
    assert torch.all(okhsv[1:, ...] <= 1.0)


def test_okhsl_and_okhsv_wrap_degree_hue_values() -> None:
    okhsl = torch.tensor([[[389.2473]], [[1.0]], [[0.5681]]], dtype=torch.float32)
    okhsl_wrapped = torch.tensor([[[29.2473]], [[1.0]], [[0.5681]]], dtype=torch.float32)
    okhsv = torch.tensor([[[389.2473]], [[1.0]], [[1.0]]], dtype=torch.float32)
    okhsv_wrapped = torch.tensor([[[29.2473]], [[1.0]], [[1.0]]], dtype=torch.float32)

    assert torch.allclose(srgb_from_okhsl(okhsl), srgb_from_okhsl(okhsl_wrapped), atol=1e-5)
    assert torch.allclose(srgb_from_okhsv(okhsv), srgb_from_okhsv(okhsv_wrapped), atol=1e-5)


def test_linear_srgb_xyz_round_trip() -> None:
    linear_srgb = torch.tensor(
        [
            [[0.0, 1.0], [0.25, 0.8]],
            [[0.2, 0.8], [0.75, 0.1]],
            [[1.0, 0.1], [0.5, 0.4]],
        ],
        dtype=torch.float32,
    )

    round_tripped = linear_srgb_from_xyz(xyz_from_linear_srgb(linear_srgb))

    assert torch.allclose(round_tripped, linear_srgb, atol=5e-5)


def test_oklab_from_xyz_matches_explicit_conversion_path() -> None:
    xyz = torch.tensor(
        [
            [[0.4124, 0.9505]],
            [[0.2126, 1.0]],
            [[0.0193, 1.0888]],
        ],
        dtype=torch.float32,
    )

    direct = oklab_from_xyz(xyz)
    via_linear_srgb = oklab_from_linear_srgb(linear_srgb_from_xyz(xyz))

    assert torch.allclose(direct, via_linear_srgb, atol=1e-6)


def test_xyz_from_oklab_matches_explicit_conversion_path() -> None:
    oklab = torch.tensor(
        [
            [[0.6, 0.4]],
            [[0.2, -0.1]],
            [[0.1, 0.05]],
        ],
        dtype=torch.float32,
    )

    direct = xyz_from_oklab(oklab)
    via_linear_srgb = xyz_from_linear_srgb(linear_srgb_from_oklab(oklab))

    assert torch.allclose(direct, via_linear_srgb, atol=1e-6)


def test_oklch_from_xyz_matches_explicit_conversion_path() -> None:
    xyz = torch.tensor(
        [
            [[0.4124, 0.9505]],
            [[0.2126, 1.0]],
            [[0.0193, 1.0888]],
        ],
        dtype=torch.float32,
    )

    direct = oklch_from_xyz(xyz)
    via_oklab = oklch_from_oklab(oklab_from_xyz(xyz))

    assert torch.allclose(direct, via_oklab, atol=1e-6)


def test_xyz_from_oklch_matches_explicit_conversion_path() -> None:
    oklch = torch.tensor(
        [
            [[0.7, 0.5]],
            [[0.12, 0.04]],
            [[30.0, 210.0]],
        ],
        dtype=torch.float32,
    )

    direct = xyz_from_oklch(oklch)
    via_oklab = xyz_from_oklab(oklab_from_oklch(oklch))

    assert torch.allclose(direct, via_oklab, atol=1e-6)


def test_lab_from_linear_srgb_matches_explicit_conversion_path() -> None:
    linear_srgb = torch.tensor(
        [
            [[0.0, 1.0], [0.25, 0.8]],
            [[0.2, 0.8], [0.75, 0.1]],
            [[1.0, 0.1], [0.5, 0.4]],
        ],
        dtype=torch.float32,
    )

    direct = lab_from_linear_srgb(linear_srgb)
    via_xyz = lab_from_xyz(xyz_from_linear_srgb(linear_srgb))

    assert torch.allclose(direct, via_xyz, atol=1e-6)


def test_linear_srgb_from_lab_matches_explicit_conversion_path() -> None:
    lab = torch.tensor(
        [
            [[0.0, 100.0], [50.0, 75.0]],
            [[0.0, 0.0], [10.0, -20.0]],
            [[0.0, 0.0], [-5.0, 30.0]],
        ],
        dtype=torch.float32,
    )

    direct = linear_srgb_from_lab(lab)
    via_xyz = linear_srgb_from_xyz(xyz_from_lab(lab))

    assert torch.allclose(direct, via_xyz, atol=1e-6)


def test_srgb_xyz_round_trip() -> None:
    srgb = torch.tensor(
        [
            [[0.0, 1.0], [0.25, 0.8]],
            [[0.2, 0.8], [0.75, 0.1]],
            [[1.0, 0.1], [0.5, 0.4]],
        ],
        dtype=torch.float32,
    )

    round_tripped = srgb_from_xyz(xyz_from_srgb(srgb))

    assert torch.allclose(round_tripped, srgb, atol=5e-4)


def test_lab_from_srgb_and_back_round_trip() -> None:
    srgb = torch.tensor(
        [
            [[0.0, 1.0], [0.25, 0.8]],
            [[0.2, 0.8], [0.75, 0.1]],
            [[1.0, 0.1], [0.5, 0.4]],
        ],
        dtype=torch.float32,
    )

    round_tripped = srgb_from_lab(lab_from_srgb(srgb))

    assert torch.allclose(round_tripped, srgb, atol=5e-4)


def test_xyz_lab_round_trip_for_d65_and_d50() -> None:
    xyz = torch.tensor(
        [
            [[0.4124, 0.9505]],
            [[0.2126, 1.0]],
            [[0.0193, 1.0888]],
        ],
        dtype=torch.float32,
    )

    round_tripped_d65 = xyz_from_lab(lab_from_xyz(xyz, reference_illuminant="D65"), reference_illuminant="D65")
    round_tripped_d50 = xyz_from_lab(lab_from_xyz(xyz, reference_illuminant="D50"), reference_illuminant="D50")

    assert torch.allclose(round_tripped_d65, xyz, atol=1e-4)
    assert torch.allclose(round_tripped_d50, xyz, atol=1e-4)


def test_xyz_d65_to_d50_maps_reference_white() -> None:
    xyz_d65 = torch.tensor([[[0.950489]], [[1.0]], [[1.088840]]], dtype=torch.float32)

    xyz_d50 = xyz_d65_to_d50(xyz_d65)

    assert torch.allclose(xyz_d50[:, 0, 0], torch.tensor([0.964212, 1.0, 0.825188]), atol=5e-4)


def test_xyz_d50_to_d65_round_trip() -> None:
    xyz_d65 = torch.tensor(
        [
            [[0.4124, 0.9505]],
            [[0.2126, 1.0]],
            [[0.0193, 1.0888]],
        ],
        dtype=torch.float32,
    )

    round_tripped = xyz_d50_to_d65(xyz_d65_to_d50(xyz_d65))

    assert torch.allclose(round_tripped, xyz_d65, atol=1e-5)


def test_lab_from_srgb_d50_matches_adapted_xyz_path() -> None:
    srgb = torch.tensor(
        [
            [[0.0, 1.0], [0.25, 0.8]],
            [[0.2, 0.8], [0.75, 0.1]],
            [[1.0, 0.1], [0.5, 0.4]],
        ],
        dtype=torch.float32,
    )

    direct = lab_from_srgb(srgb, reference_illuminant="D50")
    via_xyz = lab_from_xyz(xyz_d65_to_d50(xyz_from_srgb(srgb)), reference_illuminant="D50")

    assert torch.allclose(direct, via_xyz, atol=1e-4)


def test_lab_from_xyz_matches_reference_white_and_black() -> None:
    xyz = torch.tensor(
        [
            [[0.0, 0.950489]],
            [[0.0, 1.0]],
            [[0.0, 1.088840]],
        ],
        dtype=torch.float32,
    )

    lab = lab_from_xyz(xyz, reference_illuminant="D65")

    assert torch.allclose(lab[:, 0, 0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(lab[:, 0, 1], torch.tensor([100.0, 0.0, 0.0]), atol=1e-3)


def test_invalid_tensor_shape_raises_value_error() -> None:
    with pytest.raises(ValueError, match="3xHxW"):
        oklab_from_srgb(torch.zeros((2, 2), dtype=torch.float32))


def test_invalid_reference_illuminant_raises_value_error() -> None:
    xyz = torch.ones((3, 1, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="Unsupported reference_illuminant"):
        lab_from_xyz(xyz, reference_illuminant="E")


def test_okhsl_from_srgb_forwards_steps_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, int] = {}

    def fake_get_cs_tensor(
        l_tensor: torch.Tensor, units_ab_tensor: torch.Tensor, steps: int = 1, steps_outer: int = 1
    ) -> torch.Tensor:
        recorded["steps"] = steps
        recorded["steps_outer"] = steps_outer
        return torch.ones((3, *l_tensor.shape), dtype=l_tensor.dtype, device=l_tensor.device)

    monkeypatch.setattr(color_conversion, "_get_cs_tensor", fake_get_cs_tensor)

    srgb = torch.tensor(
        [
            [[0.2, 0.9]],
            [[0.4, 0.3]],
            [[0.8, 0.1]],
        ],
        dtype=torch.float32,
    )

    color_conversion.okhsl_from_srgb(srgb, steps=3, steps_outer=4)

    assert recorded == {"steps": 3, "steps_outer": 4}


def test_public_hsl_okhsl_okhsv_conversions_preserve_dtype_and_device() -> None:
    srgb = torch.tensor(
        [
            [[0.05, 0.95], [0.2, 0.8]],
            [[0.4, 0.2], [0.6, 0.1]],
            [[0.9, 0.05], [0.3, 0.7]],
        ],
        dtype=torch.float64,
    )

    hsl = hsl_from_srgb(srgb)
    okhsl = okhsl_from_srgb(srgb)
    okhsv = okhsv_from_srgb(srgb)
    srgb_from_plain_hsl = srgb_from_hsl(hsl)
    srgb_from_perceptual_hsl = srgb_from_okhsl(okhsl)
    srgb_from_perceptual_hsv = srgb_from_okhsv(okhsv)

    for output in (hsl, okhsl, okhsv, srgb_from_plain_hsl, srgb_from_perceptual_hsl, srgb_from_perceptual_hsv):
        assert output.dtype == srgb.dtype
        assert output.device == srgb.device
