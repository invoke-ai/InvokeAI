"""A user's calibration lines: how they are fitted, and how an estimate reads them in place of the shipped values."""

from collections.abc import Iterator

import pytest
import torch

from invokeai.backend.util.vae_working_memory import VAE_FLUX_SPEC, VAE_SD_SPEC
from invokeai.backend.working_memory import user_fit
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.fit import shipped_pair, user_pair
from invokeai.backend.working_memory.platform import Platform, platform_for
from invokeai.backend.working_memory.spec import EstimatorSpec
from invokeai.backend.working_memory.user_fit import UserFit, UserPoint, fit_user_lines

MIB = 2**20
DEVICE = torch.device("cuda:0")
PLATFORM = platform_for(DEVICE)


def _features(side: int, dtype: str = "fp16", tile: int | None = None, **extra: object) -> dict:
    return {
        "operation": "decode",
        "height": side,
        "width": side,
        "tile": tile,
        "dtype": dtype,
        "vae_class": "AutoencoderKL",
        "upcast_weight_bytes": 0,
        "score_matrix_bytes": 0,
        **extra,
    }


def _points(slope: int, base: int, sides: tuple[int, ...] = (512, 768, 1024), dtype: str = "fp16") -> list[UserPoint]:
    """Windows on an exact line ``slope * pixels + base``."""
    return [UserPoint("vae.sd", VAE_SD_SPEC.version, _features(side, dtype), slope * side**2 + base) for side in sides]


def _fit(points: list[UserPoint]) -> UserFit:
    lines, problems = fit_user_lines(points, PLATFORM)
    assert lines, problems
    return UserFit(label="user:dc:env", platform=PLATFORM, lines=lines)


@pytest.fixture
def provide() -> Iterator:
    """Register a provider that answers `fit` for every estimate; unregistered after the test."""
    registered: list[user_fit.UserFitProvider] = []

    def install(fit: UserFit | None) -> None:
        def provider(spec: EstimatorSpec, platform: Platform, device: torch.device) -> UserFit | None:
            return fit

        registered.append(provider)
        user_fit.set_user_fit_provider(provider)

    yield install
    for provider in registered:
        user_fit.clear_user_fit_provider(provider)


def _shipped(spec: EstimatorSpec, features: dict) -> int:
    return spec.evaluate(features, shipped_coefficients(spec, PLATFORM)).reservation_bytes


GRID = [
    _features(side, dtype, tile)
    for side in (256, 512, 1024, 1536, 2048, 4096)
    for dtype in ("fp16", "fp32")
    for tile in (None, 512)
]


def test_without_a_user_line_estimates_are_the_shipped_ones_bit_for_bit(provide) -> None:
    before = [VAE_SD_SPEC.estimate(features, DEVICE) for features in GRID]

    provide(None)
    without_fit = [VAE_SD_SPEC.estimate(features, DEVICE) for features in GRID]
    # A fit that measured another line only: an encode.
    encode = [
        UserPoint("vae.sd", VAE_SD_SPEC.version, _features(side) | {"operation": "encode"}, 10**9)
        for side in (512, 1024)
    ]
    provide(_fit(encode))
    other_line = [VAE_SD_SPEC.estimate(features, DEVICE) for features in GRID]

    assert before == without_fit == other_line
    assert [estimate.reservation_bytes for estimate in before] == [_shipped(VAE_SD_SPEC, f) for f in GRID]


def test_within_and_below_its_measured_sizes_the_user_line_replaces_the_shipped_one(provide) -> None:
    # Windows on 900 B/px + 400 MiB; the user line adds 10% and 256 MiB: 990 B/px and 440 + 256 MiB.
    provide(_fit(_points(900, 400 * MIB)))

    for side in (256, 512, 900, 1024):
        estimate = VAE_SD_SPEC.estimate(_features(side), DEVICE)
        assert estimate.reservation_bytes == 990 * side**2 + 696 * MIB
        assert estimate.source.startswith("user:dc:env (decode_fp16 within its measured sizes) + shipped:")

    # The fp32 line was not measured: shipped.
    fp32 = _features(1024, "fp32")
    assert VAE_SD_SPEC.estimate(fp32, DEVICE).reservation_bytes == _shipped(VAE_SD_SPEC, fp32)


def test_above_its_measured_sizes_the_larger_of_the_user_and_the_shipped_estimate_wins(provide) -> None:
    large = _features(2048)
    shipped = _shipped(VAE_SD_SPEC, large)

    provide(_fit(_points(20000, 400 * MIB)))
    high = VAE_SD_SPEC.estimate(large, DEVICE)
    assert high.reservation_bytes == 22000 * 2048**2 + 696 * MIB > shipped
    assert "above its measured sizes, over shipped" in high.source

    provide(_fit(_points(1, 1 * MIB)))
    low = VAE_SD_SPEC.estimate(large, DEVICE)
    assert low.reservation_bytes == shipped
    assert low.source.startswith("shipped:") and "over user:dc:env decode_fp16" in low.source
    # Within the measured sizes the low user line still decides.
    assert VAE_SD_SPEC.estimate(_features(1024), DEVICE).reservation_bytes == 2 * 1024**2 + 258 * MIB


def test_a_tiled_decode_is_placed_by_its_image_and_priced_on_its_tile(provide) -> None:
    provide(_fit(_points(900, 400 * MIB)))
    # 2048px is above the measured 1024px, so the larger estimate wins; both price the 512px tile, 1.25x.
    tiled = _features(2048, tile=512)
    user = 990 * 512 * 512 * 1.25 + 696 * MIB
    assert VAE_SD_SPEC.estimate(tiled, DEVICE).reservation_bytes == max(int(user), _shipped(VAE_SD_SPEC, tiled))


def test_a_line_carried_for_several_estimators_moves_all_of_them(provide) -> None:
    """vae.sd and vae.flux borrow one entry's values, so an SD fp16 decode calibrates FLUX.1's fp16 decode too."""
    assert user_fit.carrier_key("vae.sd", 2, PLATFORM) == user_fit.carrier_key("vae.flux", 2, PLATFORM)
    provide(_fit(_points(900, 400 * MIB)))
    flux = {
        "operation": "decode",
        "height": 1024,
        "width": 1024,
        "dtype": "fp16",
        "tile": None,
        "score_matrix_bytes": 0,
    }
    assert VAE_FLUX_SPEC.estimate(flux, DEVICE).reservation_bytes == 990 * 1024**2 + 696 * MIB


def test_the_user_margin_is_ten_percent_and_256_mib_whatever_the_shipped_margin() -> None:
    points = [(512 * 512, 900 * 512 * 512 + 400 * MIB), (1024 * 1024, 900 * 1024 * 1024 + 400 * MIB)]
    assert user_pair(points) == (990, 696 * MIB)
    assert shipped_pair(points) == (945, 420 * MIB)


def test_the_margin_covers_the_fitted_line_only_and_analytic_terms_stay_as_they_are(provide) -> None:
    """An upcast decode reserves the VAE's growth on top of the line: the user margin must not grow it too."""
    provide(_fit(_points(900, 400 * MIB)))
    upcast = 160 * MIB

    estimate = VAE_SD_SPEC.estimate(_features(1024, upcast_weight_bytes=upcast), DEVICE)

    assert estimate.reservation_bytes == (990 * 1024**2 + 440 * MIB) + 256 * MIB + upcast


def test_only_untiled_single_sample_autoencoder_kl_windows_with_demand_and_a_known_version_fit_a_line() -> None:
    good = _points(900, 400 * MIB, sides=(512, 1024))
    ignored = [
        UserPoint("vae.sd", 2, _features(2048, tile=512), 10**12),
        UserPoint("vae.sd", 2, _features(2048, upcast_weight_bytes=160 * MIB), 10**12),
        UserPoint("vae.sd", 2, _features(2048, vae_class="AutoencoderTiny"), 10**12),
        UserPoint("vae.sd", 2, _features(2048, batch_size=2), 10**12),
        UserPoint("vae.sd", 2, _features(2048), 0),
        UserPoint("vae.sd", 1, _features(2048), 10**12),
        UserPoint("vae.unknown", 2, _features(2048), 10**12),
    ]

    lines, _ = fit_user_lines(good + ignored, PLATFORM)

    [line] = lines.values()
    assert (line.k, line.b, line.x_min, line.x_max, line.points) == (990, 696 * MIB, 512**2, 1024**2, 2)


def test_a_line_measured_at_sizes_too_close_together_fits_nothing_and_says_why() -> None:
    lines, problems = fit_user_lines(_points(900, 400 * MIB, sides=(1024, 1280)), PLATFORM)
    assert not lines
    [problem] = problems.values()
    assert "measure more sizes" in problem


def test_a_provider_that_raises_leaves_estimates_shipped_and_only_its_own_registration_is_cleared(provide) -> None:
    features = _features(1024)
    shipped = VAE_SD_SPEC.estimate(features, DEVICE)

    def broken(spec: EstimatorSpec, platform: Platform, device: torch.device) -> UserFit | None:
        raise RuntimeError("store gone")

    user_fit.set_user_fit_provider(broken)
    try:
        assert VAE_SD_SPEC.estimate(features, DEVICE) == shipped
    finally:
        user_fit.clear_user_fit_provider(broken)

    provide(_fit(_points(900, 400 * MIB)))
    user_fit.clear_user_fit_provider(broken)
    assert VAE_SD_SPEC.estimate(features, DEVICE).source.startswith("user:")
