import pytest

from invokeai.app.invocations.anima_denoise import (
    ANIMA_SHIFT,
    AnimaDenoiseInvocation,
    inverse_loglinear_timestep_shift,
    loglinear_timestep_shift,
)


class TestLoglinearTimestepShift:
    """Test the log-linear timestep shift function used for Anima's noise schedule."""

    def test_shift_1_is_identity(self):
        """With alpha=1.0, shift should be identity."""
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert loglinear_timestep_shift(1.0, t) == t

    def test_shift_at_zero(self):
        """At t=0, shifted sigma should be 0 regardless of alpha."""
        assert loglinear_timestep_shift(3.0, 0.0) == 0.0

    def test_shift_at_one(self):
        """At t=1, shifted sigma should be 1 regardless of alpha."""
        assert loglinear_timestep_shift(3.0, 1.0) == pytest.approx(1.0)

    def test_shift_3_increases_sigma(self):
        """With alpha=3.0, sigma should be larger than t (spends more time at high noise)."""
        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            sigma = loglinear_timestep_shift(3.0, t)
            assert sigma > t, f"At t={t}, sigma={sigma} should be > t"

    def test_shift_monotonic(self):
        """Shifted sigmas should be monotonically increasing with t."""
        prev = 0.0
        for i in range(1, 101):
            t = i / 100.0
            sigma = loglinear_timestep_shift(3.0, t)
            assert sigma > prev, f"Not monotonic at t={t}"
            prev = sigma

    def test_known_value(self):
        """Test a known value: at t=0.5, alpha=3.0, sigma = 3*0.5 / (1 + 2*0.5) = 0.75."""
        assert loglinear_timestep_shift(3.0, 0.5) == pytest.approx(0.75)


class TestInverseLoglinearTimestepShift:
    """Test the inverse log-linear timestep shift (used for inpainting mask correction)."""

    def test_inverse_shift_1_is_identity(self):
        """With alpha=1.0, inverse should be identity."""
        for sigma in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert inverse_loglinear_timestep_shift(1.0, sigma) == sigma

    def test_roundtrip(self):
        """shift(inverse(sigma)) should recover sigma, and inverse(shift(t)) should recover t."""
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            sigma = loglinear_timestep_shift(3.0, t)
            recovered_t = inverse_loglinear_timestep_shift(3.0, sigma)
            assert recovered_t == pytest.approx(t, abs=1e-7), (
                f"Roundtrip failed: t={t} -> sigma={sigma} -> recovered_t={recovered_t}"
            )

    def test_known_value(self):
        """At sigma=0.75, alpha=3.0, t should be 0.5 (inverse of the known shift value)."""
        assert inverse_loglinear_timestep_shift(3.0, 0.75) == pytest.approx(0.5)


class TestGetSigmas:
    """Test the sigma schedule generation."""

    def test_schedule_length(self):
        """Schedule should have num_steps + 1 entries."""
        inv = AnimaDenoiseInvocation(
            positive_conditioning=None,  # type: ignore
            transformer=None,  # type: ignore
        )
        sigmas = inv._get_sigmas(30)
        assert len(sigmas) == 31

    def test_schedule_endpoints(self):
        """Schedule should start near 1.0 and end at 0.0."""
        inv = AnimaDenoiseInvocation(
            positive_conditioning=None,  # type: ignore
            transformer=None,  # type: ignore
        )
        sigmas = inv._get_sigmas(30)
        assert sigmas[0] == pytest.approx(loglinear_timestep_shift(ANIMA_SHIFT, 1.0))
        assert sigmas[-1] == pytest.approx(0.0)

    def test_schedule_monotonically_decreasing(self):
        """Sigmas should decrease from noise to clean."""
        inv = AnimaDenoiseInvocation(
            positive_conditioning=None,  # type: ignore
            transformer=None,  # type: ignore
        )
        sigmas = inv._get_sigmas(30)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], f"Not decreasing at index {i}: {sigmas[i]} <= {sigmas[i + 1]}"

    def test_schedule_uses_shift(self):
        """With shift=3.0, middle sigmas should be larger than the linear midpoint."""
        inv = AnimaDenoiseInvocation(
            positive_conditioning=None,  # type: ignore
            transformer=None,  # type: ignore
        )
        sigmas = inv._get_sigmas(10)
        # At step 5/10, linear t = 0.5, shifted sigma should be 0.75
        assert sigmas[5] == pytest.approx(loglinear_timestep_shift(3.0, 0.5))


class TestGetSigmasEdgeCases:
    """Test edge cases for sigma schedule generation."""

    def test_single_step_produces_valid_schedule(self):
        """_get_sigmas(num_steps=1) should produce a valid 2-element schedule."""
        inv = AnimaDenoiseInvocation(
            positive_conditioning=None,  # type: ignore
            transformer=None,  # type: ignore
        )
        sigmas = inv._get_sigmas(1)
        assert len(sigmas) == 2
        assert sigmas[0] > sigmas[1]
        assert sigmas[0] == pytest.approx(loglinear_timestep_shift(ANIMA_SHIFT, 1.0))
        assert sigmas[-1] == pytest.approx(0.0)


class TestInverseLoglinearEdgeCases:
    """Test edge cases for inverse_loglinear_timestep_shift."""

    def test_alpha_zero_does_not_divide_by_zero(self):
        """inverse_loglinear_timestep_shift with alpha=0 should not raise ZeroDivisionError.

        With alpha=0: denominator = 0 - (0-1)*sigma = sigma.
        At sigma=0, denominator=0 which hits the epsilon guard and returns 1.0.
        At sigma>0, denominator=sigma, result = sigma/sigma = 1.0.
        """
        # Should not raise
        result = inverse_loglinear_timestep_shift(0.0, 0.5)
        assert isinstance(result, float)

        # At sigma=0, denominator would be 0 — should hit the epsilon guard
        result_zero = inverse_loglinear_timestep_shift(0.0, 0.0)
        assert isinstance(result_zero, float)
