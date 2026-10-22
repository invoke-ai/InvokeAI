import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.family_e2e.__main__ import main
from scripts.family_e2e.records import (
    Environment,
    ModelIdentity,
    Record,
    Residency,
    Thresholds,
    append_record,
    write_run_info,
)
from scripts.family_e2e.references import accept
from scripts.family_e2e.report import ReportResult, _residency_text, write_report
from tests.family_e2e.test_records import make_environment, make_record, make_run_info, write_png

PROFILE = "rtx4090"
CASE = "zimage-t2i@8x8"
OTHER_CASE = "zimage-i2i@8x8"
FIRST = "20261001-120000-rtx4090"
SECOND = "20261002-120000-rtx4090"


def _case_records(
    run_id: str, case: str = CASE, totals: tuple[float, ...] = (10.0, 10.0, 10.0), **kw: Any
) -> list[Record]:
    warmup = make_record(case, "warmup", run_id=run_id, total_s=90.0, **kw)
    return [warmup] + [make_record(case, rep=rep, run_id=run_id, total_s=t, **kw) for rep, t in enumerate(totals)]


def _write_run(
    data_dir: Path,
    run_id: str,
    records: list[Record],
    *,
    fills: dict[str, int] | None = None,
    profile: str = PROFILE,
    **run_info: Any,
) -> Path:
    """Writes a run whose images are uniform 8-bit fills (100 unless `fills` maps the record_id elsewhere)."""
    run_dir = data_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    case_keys = tuple(dict.fromkeys(r.case_key for r in records))
    write_run_info(run_dir, make_run_info(run_id, profile=profile, case_keys=case_keys, **run_info))
    for record in records:
        if record.image_path:
            assert record.image_width is not None and record.image_height is not None
            fill = (fills or {}).get(record.record_id, 100)
            write_png(run_dir / record.image_path, np.full((record.image_height, record.image_width, 3), fill))
        append_record(run_dir, record)
    return run_dir


def _summary(result: ReportResult) -> dict[str, list[str]]:
    """Summary table rows by case label: status, min PSNR, time, vs ref, vs prev, peak, vs ref, vs prev."""
    rows = {}
    for line in result.summary_md.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith(("| Case", "| ---")):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


def _attention(result: ReportResult) -> list[str]:
    return [line for line in result.summary_md.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]


def test_identical_rerun_passes_and_links_all_images(tmp_path: Path) -> None:
    accept(tmp_path, _write_run(tmp_path, FIRST, _case_records(FIRST)), None)
    run_dir = _write_run(tmp_path, SECOND, _case_records(SECOND))
    (run_dir / "graphs").mkdir()
    (run_dir / "graphs" / f"{CASE}.json").write_text(
        json.dumps({"graph": {"id": "g", "nodes": {"denoise": {"id": "denoise", "type": "z_image_denoise"}}}})
    )

    result = write_report(tmp_path, run_dir)

    assert (result.failures, result.outliers, result.invalid) == (0, 0, 0)
    status, min_psnr, time, time_vs_ref, time_vs_prev, *_ = _summary(result)[CASE]
    assert (status, min_psnr, time, time_vs_ref, time_vs_prev) == ("PASS", "∞", "10.00 s", "+0.0%", "+0.0%")
    report = result.report_html.read_text(encoding="utf-8")
    assert "z_image_denoise · denoise" in report
    sources = re.findall(r'<img src="([^"]+)"', report)
    assert len(sources) == 4  # reference, this run, previous run, difference
    assert all((run_dir / urllib.parse.unquote(src)).is_file() for src in sources)


def test_quality_regression_fails_on_the_worst_measured_image(tmp_path: Path) -> None:
    accept(tmp_path, _write_run(tmp_path, FIRST, _case_records(FIRST)), None)
    records = _case_records(SECOND)
    # A uniform offset of 40 levels: MSE 1600, PSNR = 10·log10(255² / 1600) ≈ 16.09 dB.
    run_dir = _write_run(tmp_path, SECOND, records, fills={records[2].record_id: 140})

    result = write_report(tmp_path, run_dir)

    assert result.failures == 1
    assert _summary(result)[CASE][:2] == ["FAIL", "16.09 dB"]
    assert f"- FAIL {CASE}: min PSNR 16.09 dB is below 30 dB" in _attention(result)
    worst = urllib.parse.quote(f"images/{records[2].record_id}.png")
    assert f'<img src="{worst}"' in result.report_html.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("fill", "status", "reason", "exit_code"),
    [
        # A uniform offset of 40 levels: PSNR ≈ 16.09 dB.
        (140, "CHANGED", "differs from reference as expected (min PSNR 16.09 dB)", 0),
        (100, "FAIL", "expected a change, but the output matches the reference (min PSNR ∞)", 1),
    ],
)
def test_a_case_expected_to_change_passes_only_when_it_changed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fill: int, status: str, reason: str, exit_code: int
) -> None:
    accept(tmp_path, _write_run(tmp_path, FIRST, _case_records(FIRST)), None)
    records = _case_records(SECOND)
    expect_change = Thresholds(psnr_min_db=30.0, time_tolerance=0.10, peak_tolerance_mb=256.0, expect="changed")
    fills = {r.record_id: fill for r in records}
    run_dir = _write_run(tmp_path, SECOND, records, fills=fills, thresholds=expect_change)

    assert main(["report", str(run_dir)]) == exit_code

    changed = int(status == "CHANGED")
    counts = f"{1 - changed} failures, 0 outliers, 0 invalid, {changed} changed as expected, 0 baseline"
    assert counts in capsys.readouterr().out
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert f"run completed: 1 {status}, 0 with outliers" in summary
    assert f"- {status} {CASE}: {reason}" in summary.splitlines()


def test_a_run_judged_against_the_reference_it_set_is_a_baseline_unless_it_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [*_case_records(FIRST), *_case_records(FIRST, OTHER_CASE)[:-1]]
    records.append(make_record(OTHER_CASE, rep=2, run_id=FIRST, status="error", image=False, error="out of memory"))
    run_dir = _write_run(tmp_path, FIRST, records)
    accept(tmp_path, run_dir, None)

    assert main(["report", str(run_dir)]) == 1
    assert "1 failures, 0 outliers, 0 invalid, 0 changed as expected, 1 baseline" in capsys.readouterr().out

    result = write_report(tmp_path, run_dir)
    # Against its own medians there is nothing to compare.
    assert _summary(result)[CASE] == ["BASELINE", "∞", "10.00 s", "—", "—", "8,000 MB", "—", "—"]
    assert _summary(result)[OTHER_CASE][0] == "ERROR"
    assert f"- BASELINE {CASE}: reference set from this run" in _attention(result)
    assert "accepted from this run" in result.report_html.read_text(encoding="utf-8")


def test_image_of_the_wrong_size_fails(tmp_path: Path) -> None:
    accept(tmp_path, _write_run(tmp_path, FIRST, _case_records(FIRST)), None)
    run_dir = _write_run(tmp_path, SECOND, _case_records(SECOND, image_size=(8, 4)))

    result = write_report(tmp_path, run_dir)

    assert result.failures == 1
    assert _summary(result)[CASE][0] == "FAIL"
    assert f"- FAIL {CASE}: {CASE}.default.measure.0.0: image is 8x4, requested 8x8" in _attention(result)


def test_failed_generation_is_an_error(tmp_path: Path) -> None:
    records = _case_records(FIRST)
    records[1] = make_record(CASE, rep=0, run_id=FIRST, status="error", image=False, error="CUDA out of memory")
    run_dir = _write_run(tmp_path, FIRST, records)

    result = write_report(tmp_path, run_dir)

    assert result.failures == 1
    assert _summary(result)[CASE][0] == "ERROR"
    assert f"- ERROR {CASE}: {CASE}.default.measure.0.0: error: CUDA out of memory" in _attention(result)


def test_a_step_that_failed_and_then_succeeded_on_resume_is_judged_by_its_final_attempt(tmp_path: Path) -> None:
    accept(tmp_path, _write_run(tmp_path, FIRST, _case_records(FIRST)), None)
    records = _case_records(SECOND)
    crashed = make_record(CASE, rep=0, attempt=0, run_id=SECOND, status="error", image=False, error="server died")
    resumed = records[1].model_copy(update={"attempt": 1, "record_id": f"{CASE}.default.measure.0.1"})
    resumed = resumed.model_copy(update={"image_path": f"images/{resumed.record_id}.png"})
    run_dir = _write_run(tmp_path, SECOND, [records[0], crashed, resumed, *records[2:]])

    result = write_report(tmp_path, run_dir)

    assert result.failures == 0
    assert _summary(result)[CASE][0] == "PASS"
    assert f"- FLAG {CASE}: earlier attempt {CASE}.default.measure.0.0 error: server died" in _attention(result)


def test_case_without_valid_measurements_is_invalid_with_reasons(tmp_path: Path) -> None:
    reason = "model residency 80 % < 100 %"
    run_dir = _write_run(tmp_path, FIRST, _case_records(FIRST, valid_psnr=False, invalid_reasons=[reason]))

    result = write_report(tmp_path, run_dir)

    assert (result.failures, result.invalid) == (0, 1)
    assert _summary(result)[CASE][0] == "INVALID"
    attention = _attention(result)
    assert f"- INVALID {CASE}: no measured image is valid for PSNR" in attention
    assert f"- INVALID METRICS {CASE}: PSNR invalid: {reason} (3 of 3 measured)" in attention


def test_partial_residency_marks_time_and_vram_invalid_but_keeps_the_psnr_verdict(tmp_path: Path) -> None:
    # The shape of a real forced-partial-loading run: residency spoils time and memory, the image stays rated.
    reasons = [
        "residency: te:text_encoder loaded with 33.2% of its weights in VRAM",
        "residency: tr:transformer loaded with 41.7% of its weights in VRAM",
    ]
    shape = {"valid_timing": False, "valid_memory": False, "valid_psnr": True, "invalid_reasons": reasons}
    records = [make_record(CASE, "warmup", run_id=FIRST, **shape)]
    records += [make_record(CASE, rep=rep, run_id=FIRST, **shape) for rep in range(2)]
    run_dir = _write_run(tmp_path, FIRST, records)

    result = write_report(tmp_path, run_dir)

    assert (result.failures, result.invalid) == (0, 1)
    status, _, time, _, _, peak, *_ = _summary(result)[CASE]
    assert (status, time, peak) == ("NO-REFERENCE", "invalid", "invalid")
    summary = result.summary_md.read_text(encoding="utf-8")
    assert "1 NO-REFERENCE, 0 with outliers, 1 with invalid metrics." in summary
    attention = _attention(result)
    for reason in reasons:
        assert f"- INVALID METRICS {CASE}: time and VRAM invalid: {reason} (2 of 2 measured)" in attention
    assert f'title="{reasons[0]}; {reasons[1]}">invalid</td>' in result.report_html.read_text(encoding="utf-8")


def test_a_retried_attempt_does_not_make_metrics_invalid(tmp_path: Path) -> None:
    spoiled = make_record(
        CASE,
        rep=0,
        attempt=1,
        run_id=FIRST,
        total_s=30.0,
        valid_timing=False,
        valid_memory=False,
        invalid_reasons=["foreign queue item 17 ran during the measurement"],
    )
    retried = [make_record(CASE, rep=0, attempt=2, run_id=FIRST)]
    retried += [make_record(CASE, rep=rep, attempt=1, run_id=FIRST) for rep in (1, 2)]
    run_dir = _write_run(tmp_path, FIRST, [spoiled, *retried])

    result = write_report(tmp_path, run_dir)

    assert result.invalid == 0
    assert _summary(result)[CASE][2] == "10.00 s"
    assert not any("INVALID METRICS" in line for line in _attention(result))
    assert "retried as attempt 2" in result.report_html.read_text(encoding="utf-8")


def test_report_without_reference_never_creates_one(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, FIRST, _case_records(FIRST))

    result = write_report(tmp_path, run_dir)

    assert _summary(result)[CASE][0] == "NO-REFERENCE"
    assert result.failures == 0
    assert not (tmp_path / "references").exists()


def test_graph_and_environment_changes_against_the_reference_are_flagged(tmp_path: Path) -> None:
    accept(tmp_path, _write_run(tmp_path, FIRST, _case_records(FIRST)), None)
    changed = make_environment(
        app_deps={"torch": "2.9.0+cu130", "CUDA": "13.0"},
        gpu_driver="32.0.15.7680",
        runtime_config={
            "config": {"pytorch_cuda_alloc_conf": "backend:cudaMallocAsync", "enable_partial_loading": True}
        },
    )
    records = _case_records(SECOND, graph_hash="graph-2")
    run_dir = _write_run(tmp_path, SECOND, records, environment=changed, graph_hash="graph-2")

    result = write_report(tmp_path, run_dir)

    assert _summary(result)[CASE][0] == "PASS"
    attention = _attention(result)
    assert f"- FLAG {CASE}: graph changed vs reference" in attention
    assert (
        f"- FLAG {CASE}: environment changed vs reference: torch 2.8.0+cu128 → 2.9.0+cu130; CUDA 12.8 → 13.0; "
        "GPU driver 32.0.15.6094 → 32.0.15.7680; enable_partial_loading False → True"
    ) in attention


def test_working_memory_attention_and_int8_settings_count_as_environment_changes(tmp_path: Path) -> None:
    def environment(**config: object) -> Environment:
        settings = {"device_working_mem_gb": 3, "attention_type": "auto", "attention_backend": "auto", **config}
        return make_environment(runtime_config={"set_fields": [], "config": settings})

    accept(tmp_path, _write_run(tmp_path, FIRST, _case_records(FIRST), environment=environment()), None)
    changed = environment(
        device_working_mem_gb=6, attention_backend="sage", rocm_aotriton_experimental="on", int8_compute=True
    )
    run_dir = _write_run(tmp_path, SECOND, _case_records(SECOND), environment=changed)

    result = write_report(tmp_path, run_dir)

    assert (
        f"- FLAG {CASE}: environment changed vs reference: device_working_mem_gb 3 → 6; "
        "attention_backend auto → sage; rocm_aotriton_experimental — → on; int8_compute — → True"
    ) in _attention(result)


@pytest.mark.parametrize(
    ("totals", "outliers", "shown"),
    [
        ((12.0, 12.1, 11.9), 1, "**+20.0%**"),  # beyond the 10 % tolerance and 3×MAD = 0.3 s
        ((10.5, 10.4, 10.6), 0, "+5.0%"),  # within tolerance
        ((10.0, 12.0, 14.0), 0, "+20.0%"),  # beyond tolerance, but within 3×MAD = 6 s of this run's noise
    ],
)
def test_time_outlier_against_the_previous_run(
    tmp_path: Path, totals: tuple[float, ...], outliers: int, shown: str
) -> None:
    _write_run(tmp_path, FIRST, _case_records(FIRST))
    run_dir = _write_run(tmp_path, SECOND, _case_records(SECOND, totals=totals))

    result = write_report(tmp_path, run_dir)

    assert result.outliers == outliers
    assert _summary(result)[CASE][4] == shown
    assert any(line.startswith(f"- OUTLIER {CASE}: Total time vs previous run") for line in _attention(result)) is bool(
        outliers
    )


def test_previous_run_is_the_latest_earlier_one_of_the_profile_with_the_case(tmp_path: Path) -> None:
    current = "20261003-000000-rtx4090"
    _write_run(tmp_path, "20261001-000000-rtx4090", _case_records("20261001-000000-rtx4090", totals=(20.0,) * 3))
    _write_run(tmp_path, "20261002-000000-rtx4090", _case_records("20261002-000000-rtx4090", OTHER_CASE))
    failed = [make_record(CASE, rep=0, run_id="20261002-060000-rtx4090", status="failed", image=False)]
    _write_run(tmp_path, "20261002-060000-rtx4090", failed)
    _write_run(tmp_path, "20261002-120000-amd", _case_records("20261002-120000-amd", totals=(5.0,) * 3), profile="amd")
    run_dir = _write_run(tmp_path, current, _case_records(current))
    _write_run(tmp_path, "20261004-000000-rtx4090", _case_records("20261004-000000-rtx4090", totals=(40.0,) * 3))

    result = write_report(tmp_path, run_dir)

    assert _summary(result)[CASE][4] == "**-50.0%**"
    report = result.report_html.read_text(encoding="utf-8")
    assert "20261001-000000-rtx4090" in report
    assert not any(run_id in report for run_id in ("20261002-", "20261004-"))


def test_runs_of_other_profiles_are_never_read_but_an_unreadable_run_of_this_one_is_noted(tmp_path: Path) -> None:
    current = "20261003-000000-rtx4090"
    _write_run(tmp_path, "20261001-000000-rtx4090", _case_records("20261001-000000-rtx4090", totals=(20.0,) * 3))
    # run.json files of another version of the tool, with a field this one does not know. The first profile only
    # starts like this run's.
    for run_id in ("20261002-000000-rtx4090-vae", "20261002-060000-rtx4090"):
        (tmp_path / "runs" / run_id).mkdir()
        (tmp_path / "runs" / run_id / "run.json").write_text(json.dumps({"calibrate": False}))
    run_dir = _write_run(tmp_path, current, _case_records(current))

    result = write_report(tmp_path, run_dir)

    assert _summary(result)[CASE][4] == "**-50.0%**"
    notes = [line for line in _attention(result) if line.startswith("- NOTE")]
    assert len(notes) == 1
    assert notes[0].startswith("- NOTE Earlier run 20261002-060000-rtx4090 skipped, unreadable: ")


def test_broken_inputs_of_one_case_do_not_hide_the_others(tmp_path: Path) -> None:
    first = _write_run(tmp_path, FIRST, _case_records(FIRST) + _case_records(FIRST, OTHER_CASE))
    accept(tmp_path, first, None)
    (tmp_path / "references" / PROFILE / CASE / "reference.json").write_text("{")
    records = _case_records(SECOND) + _case_records(SECOND, OTHER_CASE)
    run_dir = _write_run(tmp_path, SECOND, records)
    (run_dir / records[-1].image_path).unlink()  # type: ignore[operator]

    result = write_report(tmp_path, run_dir)

    summary = _summary(result)
    assert summary[CASE][0] == "NO-REFERENCE"
    assert any(line.startswith(f"- NO-REFERENCE {CASE}: reference.json unreadable") for line in _attention(result))
    assert summary[OTHER_CASE][0] == "FAIL"
    assert f"- FAIL {OTHER_CASE}: {records[-1].record_id}: image missing" in _attention(result)


def test_ab_run_renders_each_arm(tmp_path: Path) -> None:
    records = [r for arm in ("A", "B") for r in _case_records(FIRST, arm=arm)]
    run_dir = _write_run(tmp_path, FIRST, records, arms=("A", "B"))

    result = write_report(tmp_path, run_dir)

    assert {label: cells[0] for label, cells in _summary(result).items()} == {
        f"{CASE} [A]": "NO-REFERENCE",
        f"{CASE} [B]": "NO-REFERENCE",
    }


def test_each_case_names_its_models_with_a_short_hash(tmp_path: Path) -> None:
    main = ModelIdentity(
        key="k1",
        name="Z-Image Turbo",
        base="z-image",
        type="main",
        format="diffusers",
        variant=None,
        hash="blake3:" + "0123456789abcdef" * 4,
        source="Tongyi-MAI/Z-Image-Turbo",
    )
    encoder = main.model_copy(
        update={"key": "k2", "name": "Qwen3 4B", "type": "qwen3_encoder", "hash": "sha256:" + "f" * 64}
    )
    third = "zimage-t2i@16x16"
    records = [*_case_records(FIRST), *_case_records(FIRST, OTHER_CASE), *_case_records(FIRST, third)]
    models = {CASE: [main, encoder], OTHER_CASE: [main], third: [main, encoder]}
    run_dir = _write_run(tmp_path, FIRST, records, case_models=models)

    result = write_report(tmp_path, run_dir)

    # Sizes of a case share its models, so the summary lists each distinct set once.
    assert (
        "\n## Models\n\n"
        f"- {CASE}, {third}: Z-Image Turbo (main, blake3:0123456789ab); Qwen3 4B (qwen3_encoder, sha256:ffffffffffff)\n"
        f"- {OTHER_CASE}: Z-Image Turbo (main, blake3:0123456789ab)\n"
    ) in result.summary_md.read_text(encoding="utf-8")
    report = result.report_html.read_text(encoding="utf-8")
    assert f'Qwen3 4B <span class="muted">qwen3_encoder · <span title="{encoder.hash}">sha256:ffffffffffff' in report


def test_the_residency_column_names_the_least_resident_weighted_load() -> None:
    tokenizer = Residency(model="te", submodel="tokenizer_2", vram_mb=0.0, total_mb=0.03, percent=0.0)
    encoder = Residency(model="te", submodel="text_encoder_2", vram_mb=7600.0, total_mb=7600.0, percent=100.0)
    transformer = Residency(model="tr", submodel="transformer", vram_mb=9000.0, total_mb=12000.0, percent=75.0)

    assert _residency_text(make_record().model_copy(update={"residency": [tokenizer]})) == "unknown"
    assert _residency_text(make_record().model_copy(update={"residency": [tokenizer, encoder]})) == "100 %"
    partial = make_record().model_copy(update={"residency": [tokenizer, encoder, transformer]})
    assert _residency_text(partial) == "75 % (tr/transformer)"
    # A record written before sizes were kept cannot be weightless: its 0.0 % is a load without weights in VRAM.
    old = Residency(model="tr", submodel="transformer", vram_mb=0.0, percent=0.0)
    assert _residency_text(make_record().model_copy(update={"residency": [old]})) == "0 % (tr/transformer)"
