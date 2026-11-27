import hashlib
import logging
from pathlib import Path

import numpy as np
import pytest

from scripts.family_e2e.records import Record, append_record, load_run_info, write_run_info
from scripts.family_e2e.references import AcceptError, accept, load_reference
from tests.family_e2e.test_records import MB, make_record, make_run_info, write_png

RUN_ID = "20261001-120000-rtx4090"
CASE = "zimage-t2i@8x8"
OTHER_CASE = "zimage-i2i@8x8"


def _run(data_dir: Path, records: list[Record], *, arms: tuple[str, ...] = ("default",)) -> Path:
    run_dir = data_dir / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    write_run_info(
        run_dir, make_run_info(RUN_ID, case_keys=tuple(dict.fromkeys(r.case_key for r in records)), arms=arms)
    )
    for index, record in enumerate(records):
        if record.image_path:
            write_png(run_dir / record.image_path, np.full((8, 8, 3), index * 20, dtype=np.uint8))
        append_record(run_dir, record)
    return run_dir


def test_accept_uses_first_psnr_valid_image_and_medians_of_valid_records(tmp_path: Path) -> None:
    records = [
        make_record(CASE, "warmup", total_s=90.0, peak_bytes=9000 * MB),
        make_record(CASE, rep=0, total_s=10.0, per_step_s=1.0, peak_bytes=4000 * MB, valid_psnr=False),
        make_record(CASE, rep=1, total_s=11.0, per_step_s=1.2, peak_bytes=5000 * MB, valid_memory=False),
        make_record(CASE, rep=2, total_s=40.0, per_step_s=4.0, peak_bytes=6000 * MB, valid_timing=False),
        make_record(CASE, rep=3, status="failed", total_s=1.0, peak_bytes=1 * MB, error="boom"),
    ]
    run_dir = _run(tmp_path, records)

    [reference] = accept(tmp_path, run_dir, None)

    loaded = load_reference(tmp_path, "rtx4090", CASE)
    assert loaded is not None and loaded[0] == reference
    image_bytes = (run_dir / f"images/{CASE}.default.measure.1.0.png").read_bytes()
    assert loaded[1].read_bytes() == image_bytes
    assert reference.image_sha256 == hashlib.sha256(image_bytes).hexdigest()
    assert reference.source_run_id == RUN_ID
    assert reference.total_s_median == 10.5
    assert reference.per_step_s_median == pytest.approx(1.1)
    assert reference.process_peak_bytes_median == 5000 * MB
    assert reference.node_peak_bytes_median == {"denoise": 5000 * MB}


def test_accept_refuses_runs_with_ab_arms(tmp_path: Path) -> None:
    records = [make_record(CASE, rep=0, arm="A"), make_record(CASE, rep=0, arm="B")]
    run_dir = _run(tmp_path, records, arms=("A", "B"))

    with pytest.raises(AcceptError, match="arms A, B"):
        accept(tmp_path, run_dir, None)
    assert not (tmp_path / "references").exists()


def test_accept_refuses_a_calibration_run(tmp_path: Path) -> None:
    """The probe changes time and VRAM, so a calibration run's results never become references."""
    run_dir = _run(tmp_path, [make_record(CASE, rep=0)])
    write_run_info(run_dir, load_run_info(run_dir).model_copy(update={"calibrate": True}))

    with pytest.raises(AcceptError, match="is a calibration run"):
        accept(tmp_path, run_dir, None)
    assert not (tmp_path / "references").exists()


def test_accept_writes_nothing_when_one_requested_case_has_no_valid_image(tmp_path: Path) -> None:
    records = [
        make_record(CASE, rep=0),
        make_record(OTHER_CASE, rep=0, valid_psnr=False, invalid_reasons=["residency 80 %"]),
        make_record(OTHER_CASE, rep=1, status="error", error="CUDA out of memory"),
    ]
    run_dir = _run(tmp_path, records)

    with pytest.raises(AcceptError, match=f"{OTHER_CASE}: no completed measured record"):
        accept(tmp_path, run_dir, [CASE, OTHER_CASE])
    assert not (tmp_path / "references").exists()


def test_accept_refuses_an_image_of_the_wrong_size(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, [make_record(CASE, rep=0, image_size=(8, 4))])

    with pytest.raises(AcceptError, match="8x4 is not the requested 8x8"):
        accept(tmp_path, run_dir, None)


def test_accept_only_the_requested_case(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, [make_record(CASE, rep=0), make_record(OTHER_CASE, rep=0)])

    accepted = accept(tmp_path, run_dir, [OTHER_CASE])

    assert [r.case_key for r in accepted] == [OTHER_CASE]
    assert load_reference(tmp_path, "rtx4090", CASE) is None
    assert load_reference(tmp_path, "rtx4090", OTHER_CASE) is not None


def test_accept_reports_differing_measured_images(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    records = [
        make_record(CASE, rep=0, image_sha256="one"),
        make_record(CASE, rep=1, image_sha256="two"),
    ]
    run_dir = _run(tmp_path, records)

    with caplog.at_level(logging.WARNING):
        accepted = accept(tmp_path, run_dir, None)

    assert len(accepted) == 1
    assert f"{CASE}: the 2 measured images are not identical" in caplog.text


def test_reaccept_replaces_the_reference(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, [make_record(CASE, rep=0, total_s=10.0)])
    accept(tmp_path, run_dir, None)
    (run_dir / "records.jsonl").unlink()
    append_record(run_dir, make_record(CASE, rep=0, total_s=20.0))

    accept(tmp_path, run_dir, None)

    loaded = load_reference(tmp_path, "rtx4090", CASE)
    assert loaded is not None and loaded[0].total_s_median == 20.0
    assert sorted(p.name for p in loaded[1].parent.iterdir()) == ["image.png", "reference.json"]
