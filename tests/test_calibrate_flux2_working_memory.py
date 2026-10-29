import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


def _load_calibration_script():
    path = Path(__file__).parents[1] / "scripts" / "calibrate_flux2_working_memory.py"
    spec = spec_from_file_location("calibrate_flux2_working_memory", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch, *args: str):
    script = _load_calibration_script()
    # Off the device either way, so an accepted command line stops at the device check instead of measuring.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["calibrate_flux2_working_memory.py", *args])
    with pytest.raises(SystemExit) as exit_info:
        script.main()
    return exit_info.value.code


@pytest.mark.parametrize("sections", [["--only", "vae"], []], ids=["only-vae", "all-sections"])
def test_a_vae_dtype_with_no_shipped_line_is_refused_before_anything_runs(monkeypatch, capsys, sections) -> None:
    """The FLUX.2 VAE ships bf16 and fp32 lines only; an fp16 run would measure everything else first and then fail
    on a bare `KeyError` when it compared against the line."""
    assert _run(monkeypatch, *sections, "--dtype", "float16") == 2
    err = capsys.readouterr().err
    assert "--dtype float16" in err
    assert "bfloat16, float32" in err


def test_float16_is_accepted_for_the_sections_that_do_not_compare_against_a_vae_line(monkeypatch) -> None:
    assert _run(monkeypatch, "--only", "sdpa", "--only", "denoise", "--dtype", "float16") == (
        "No CUDA/HIP device available."
    )
