"""Qwen-Image-2.1's schedules against what `QwenImage21Pipeline` samples.

The expected sigmas were recorded from the pipeline's own path -- `retrieve_timesteps` with the scheduler configs
shipped in Qwen/Qwen-Image-2.1 and Qwen/Qwen-Image-2.1-Turbo -- not derived from the code under test.
"""

import pytest
import torch

from invokeai.backend.model_manager.taxonomy import QwenImage21VariantType
from invokeai.backend.qwen_image_2_1.sampling import TURBO_SIGMAS, build_sigmas

BASE = QwenImage21VariantType.Base
TURBO = QwenImage21VariantType.Turbo


@pytest.mark.parametrize(
    ("steps", "tokens", "head", "tail"),
    [
        # 1024x1024 and 2048x2048: the shift grows with the image, which keeps more steps at high noise.
        (40, 4096, [1.0, 0.986964, 0.973593], [0.067881, 0.02, 0.0]),
        (40, 16384, [1.0, 0.992646, 0.985013], [0.10223, 0.02, 0.0]),
        (8, 4096, [1.0, 0.916024, 0.820046], [0.244054, 0.02, 0.0]),
    ],
)
def test_the_base_schedule_is_the_pipelines(steps: int, tokens: int, head: list[float], tail: list[float]) -> None:
    sigmas = build_sigmas(BASE, steps, tokens)
    assert sigmas.shape == (steps + 1,)
    torch.testing.assert_close(sigmas[:3], torch.tensor(head), atol=1e-6, rtol=0)
    torch.testing.assert_close(sigmas[-3:], torch.tensor(tail), atol=1e-6, rtol=0)


@pytest.mark.parametrize("tokens", [4096, 16384])
def test_turbo_samples_its_table_unshifted_at_every_size(tokens: int) -> None:
    sigmas = build_sigmas(TURBO, 8, tokens)
    torch.testing.assert_close(sigmas, torch.tensor([*TURBO_SIGMAS, 0.0]), atol=1e-6, rtol=0)


def test_turbo_at_another_step_count_keeps_the_tables_range_and_order() -> None:
    sigmas = build_sigmas(TURBO, 4, 4096).tolist()
    assert len(sigmas) == 5
    assert sigmas[0] == pytest.approx(1.0) and sigmas[3] == pytest.approx(TURBO_SIGMAS[-1]) and sigmas[4] == 0.0
    assert sigmas == sorted(sigmas, reverse=True)


def test_an_unknown_variant_samples_like_the_base_model() -> None:
    torch.testing.assert_close(build_sigmas(None, 40, 4096), build_sigmas(BASE, 40, 4096))


@pytest.mark.parametrize("variant", [BASE, TURBO])
def test_a_single_step_runs_the_whole_range(variant) -> None:
    # The pipeline's stretch to `shift_terminal` divides by zero on one base step and returns NaN.
    assert build_sigmas(variant, 1, 4096).tolist() == [1.0, 0.0]
