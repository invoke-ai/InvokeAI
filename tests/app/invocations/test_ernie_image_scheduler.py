"""Which noise schedule an ERNIE-Image generation actually runs on.

`shift` reshapes every sigma in the schedule, and the driver hands the scheduler raw sigmas
expecting it to apply that shift. Both released pipelines ship `shift: 4.0`; diffusers defaults to
1.0. A single-file checkpoint carries no `scheduler/` directory at all -- `config.path` is the file
-- so the fallback is not an edge case there, it is the only path, and getting it wrong produces a
perfectly coherent image on the wrong schedule. Turbo feels it worst at 8 steps.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.ernie_image.ernie_image_denoise import ErnieImageDenoiseInvocation
from invokeai.backend.flux.schedulers import ERNIE_IMAGE_SHIFT
from invokeai.backend.model_manager.taxonomy import ModelFormat

RELEASED_SHIFT = 4.0


def _node(scheduler: str = "euler") -> ErnieImageDenoiseInvocation:
    return ErnieImageDenoiseInvocation.model_construct(
        scheduler=scheduler, transformer=SimpleNamespace(transformer=object())
    )


def _context(format: ModelFormat, models_path: Path, relative_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(get_config=lambda _model: SimpleNamespace(format=format, path=relative_path)),
        config=SimpleNamespace(get=lambda: SimpleNamespace(models_path=models_path)),
        logger=MagicMock(),
    )


def test_the_released_shift_is_what_we_carry() -> None:
    assert ERNIE_IMAGE_SHIFT == RELEASED_SHIFT


def test_a_single_file_denoises_on_the_released_schedule(tmp_path: Path) -> None:
    """No `scheduler/` exists for a checkpoint, so this is every single-file generation."""
    (tmp_path / "ernie-image.safetensors").touch()
    context = _context(ModelFormat.Checkpoint, tmp_path, "ernie-image.safetensors")

    scheduler = _node()._build_scheduler(context)  # type: ignore[arg-type]

    assert scheduler.config.shift == RELEASED_SHIFT
    # A file having no scheduler directory is the normal case, not something to warn about.
    context.logger.warning.assert_not_called()


def test_a_pipeline_still_reads_its_own_scheduler_config(tmp_path: Path) -> None:
    """The pipeline's own config wins over the constant -- a redistribution may ship another."""
    pipeline = tmp_path / "ernie-pipeline"
    (pipeline / "scheduler").mkdir(parents=True)
    (pipeline / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 3.0}),
        encoding="utf-8",
    )
    context = _context(ModelFormat.Diffusers, tmp_path, "ernie-pipeline")

    scheduler = _node()._build_scheduler(context)  # type: ignore[arg-type]

    assert scheduler.config.shift == 3.0


def test_a_pipeline_without_a_scheduler_config_is_reported_and_still_shifted(tmp_path: Path) -> None:
    """A diffusers directory *should* carry one, so its absence is worth saying out loud -- but the
    generation still has to run on ERNIE's schedule rather than diffusers' unshifted default."""
    (tmp_path / "ernie-pipeline").mkdir()
    context = _context(ModelFormat.Diffusers, tmp_path, "ernie-pipeline")

    scheduler = _node()._build_scheduler(context)  # type: ignore[arg-type]

    assert scheduler.config.shift == RELEASED_SHIFT
    context.logger.warning.assert_called_once()


def test_an_unreadable_scheduler_config_still_shifts(tmp_path: Path) -> None:
    """A config that exists but cannot be parsed is the one remaining way to reach a
    default-constructed scheduler. Diffusers' default is the unshifted schedule, which is wrong for
    ERNIE whether or not we could read the file."""
    pipeline = tmp_path / "ernie-pipeline"
    (pipeline / "scheduler").mkdir(parents=True)
    (pipeline / "scheduler" / "scheduler_config.json").write_text("{ not json", encoding="utf-8")
    context = _context(ModelFormat.Diffusers, tmp_path, "ernie-pipeline")

    scheduler = _node()._build_scheduler(context)  # type: ignore[arg-type]

    assert scheduler.config.shift == RELEASED_SHIFT
    context.logger.warning.assert_called_once()


@pytest.mark.parametrize("scheduler_name", ["euler", "heun"])
def test_every_offered_scheduler_takes_the_shift(tmp_path: Path, scheduler_name: str) -> None:
    """The dropdown offers more than Euler, and a class that ignored `shift` would silently run the
    unshifted schedule for that choice alone."""
    (tmp_path / "ernie-image.safetensors").touch()
    context = _context(ModelFormat.Checkpoint, tmp_path, "ernie-image.safetensors")

    scheduler = _node(scheduler_name)._build_scheduler(context)  # type: ignore[arg-type]

    assert scheduler.config.shift == RELEASED_SHIFT
