"""Tests for the transformers state-dict load report suppression.

The report is built and logged entirely inside transformers, so the cases that matter are
exercised against a real (tiny) checkpoint rather than a hand-written log record: what breaks
the mechanism is transformers moving the emit site, renaming the table, or restructuring it —
none of which a synthetic message would notice.
"""

import logging
import threading
from pathlib import Path
from typing import Iterator

import pytest
from transformers import CLIPConfig, CLIPModel, CLIPVisionConfig, CLIPVisionModelWithProjection

from invokeai.backend.util.load_report import _REPORT_LOGGER_NAME, suppress_load_report

TOWER_KWARGS = {"hidden_size": 32, "intermediate_size": 37, "num_hidden_layers": 1, "num_attention_heads": 2}
VISION_KWARGS = {**TOWER_KWARGS, "image_size": 16, "patch_size": 4, "projection_dim": 8}


@pytest.fixture(scope="module")
def full_clip_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A full CLIP checkpoint: loading either tower out of it leaves the other unexpected."""
    path = tmp_path_factory.mktemp("full-clip")
    config = CLIPConfig(
        text_config={**TOWER_KWARGS, "vocab_size": 99, "max_position_embeddings": 16, "projection_dim": 8},
        vision_config=VISION_KWARGS,
        projection_dim=8,
    )
    CLIPModel(config).save_pretrained(path)
    return path


@pytest.fixture
def reports() -> Iterator[list[logging.LogRecord]]:
    """Load reports that survive the filter, as the log would receive them."""
    logger = logging.getLogger(_REPORT_LOGGER_NAME)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "LOAD REPORT" in record.getMessage():
                records.append(record)

    handler = Capture()
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_expected_report_is_dropped_only_inside_the_context(
    full_clip_checkpoint: Path, reports: list[logging.LogRecord]
) -> None:
    with suppress_load_report():
        CLIPVisionModelWithProjection.from_pretrained(full_clip_checkpoint, local_files_only=True)
    assert reports == []

    CLIPVisionModelWithProjection.from_pretrained(full_clip_checkpoint, local_files_only=True)
    assert len(reports) == 1
    # The text tower is what the dropped report was about.
    assert "text_model" in reports[0].getMessage()


def test_report_naming_missing_weights_survives(full_clip_checkpoint: Path, reports: list[logging.LogRecord]) -> None:
    # A tower the checkpoint cannot fill: transformers initializes the second layer randomly
    # and says so in the same report as the expected keys. Silencing that would leave an
    # encoder embedding on noise with nothing in the log.
    deeper = CLIPVisionConfig(**{**VISION_KWARGS, "num_hidden_layers": 2})

    with suppress_load_report():
        CLIPVisionModelWithProjection.from_pretrained(full_clip_checkpoint, config=deeper, local_files_only=True)

    assert len(reports) == 1
    assert "MISSING" in reports[0].getMessage()
    assert reports[0].levelno == logging.WARNING


def test_other_messages_from_the_same_load_are_kept() -> None:
    logger = logging.getLogger(_REPORT_LOGGER_NAME)
    messages: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = Capture()
    logger.addHandler(handler)
    try:
        with suppress_load_report():
            logger.warning("Some weights of the model checkpoint were not used")
    finally:
        logger.removeHandler(handler)

    assert messages == ["Some weights of the model checkpoint were not used"]


def test_suppression_does_not_reach_other_threads(full_clip_checkpoint: Path, reports: list[logging.LogRecord]) -> None:
    # Loads run concurrently: the image index worker embeds while a generation loads models.
    suppressing = threading.Event()
    loaded = threading.Event()

    def other_thread() -> None:
        suppressing.wait(timeout=10)
        CLIPVisionModelWithProjection.from_pretrained(full_clip_checkpoint, local_files_only=True)
        loaded.set()

    thread = threading.Thread(target=other_thread)
    thread.start()
    try:
        with suppress_load_report():
            suppressing.set()
            assert loaded.wait(timeout=30)
    finally:
        thread.join(timeout=10)

    assert len(reports) == 1


def test_nested_suppression_leaves_the_outer_scope_suppressing(
    full_clip_checkpoint: Path, reports: list[logging.LogRecord]
) -> None:
    with suppress_load_report():
        with suppress_load_report():
            pass
        CLIPVisionModelWithProjection.from_pretrained(full_clip_checkpoint, local_files_only=True)

    assert reports == []
