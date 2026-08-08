"""Every PiD node must forward `pid_memory_optimization` — to the decode *and* to the memory estimate.

Two things go wrong silently here, so both are pinned structurally rather than per-node:

1. A new PiD node that forgets `pid_memory_optimization=` on its `PiDDecodeConfig` decodes unoptimized
   while the user believes the setting applies. There is no error and no log; the only symptom is VRAM
   use nobody is measuring.
2. Passing the flag to the decode but not to `estimate_pid_decode_working_memory` is worse than not
   wiring it at all. The cache takes `max(working_mem_bytes, device_working_mem_gb)` and subtracts it
   from the weight budget, so an estimate calibrated for the unoptimized peak withholds VRAM that the
   optimization just freed — PidNet then partial-loads to CPU on exactly the low-VRAM machines the
   feature exists for.

An AST sweep is used instead of invoking each node because it needs no model fixtures and, unlike a
per-node test, it automatically covers the eighth PiD node the day someone adds it.
"""

import ast
from pathlib import Path

import pytest

_INVOCATIONS_DIR = Path(__file__).parents[3] / "invokeai" / "app" / "invocations"
_FLAG = "pid_memory_optimization"
_ESTIMATE = "estimate_pid_decode_working_memory"


def _modules_constructing_a_decode_config() -> list[Path]:
    modules = sorted(p for p in _INVOCATIONS_DIR.glob("*.py") if "PiDDecodeConfig(" in p.read_text(encoding="utf-8"))
    assert modules, "no PiD nodes found - has the invocations directory moved?"
    return modules


def _calls(tree: ast.AST, func_name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == func_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == func_name)
        )
    ]


@pytest.mark.parametrize("module_path", _modules_constructing_a_decode_config(), ids=lambda p: p.stem)
def test_pid_node_forwards_the_setting_to_the_decode(module_path: Path) -> None:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    configs = _calls(tree, "PiDDecodeConfig")
    assert configs, f"{module_path.name}: expected at least one PiDDecodeConfig(...)"

    for call in configs:
        keywords = {kw.arg for kw in call.keywords}
        assert _FLAG in keywords, (
            f"{module_path.name}:{call.lineno} builds a PiDDecodeConfig without {_FLAG}=; "
            "the app config setting would silently not apply to this node."
        )


@pytest.mark.parametrize("module_path", _modules_constructing_a_decode_config(), ids=lambda p: p.stem)
def test_pid_node_estimates_working_memory_for_the_same_mode_it_decodes_in(module_path: Path) -> None:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    estimates = _calls(tree, _ESTIMATE)
    assert estimates, f"{module_path.name}: expected a call to {_ESTIMATE}(...)"

    for call in estimates:
        passes_flag = len(call.args) >= 3 or any(kw.arg == _FLAG for kw in call.keywords)
        assert passes_flag, (
            f"{module_path.name}:{call.lineno} estimates working memory without {_FLAG}; "
            "the cache would reserve the unoptimized peak and withhold the VRAM the flag frees."
        )
        assert any(kw.arg == "model" for kw in call.keywords), (
            f"{module_path.name}:{call.lineno} estimates working memory without the loaded model; "
            "CUDA autocast weight-cache memory would be omitted from the reservation."
        )
        assert any(kw.arg == "device" for kw in call.keywords), (
            f"{module_path.name}:{call.lineno} estimates working memory without the compute device; "
            "the CUDA-only autocast weight-cache term could be applied on the wrong device."
        )


@pytest.mark.parametrize("module_path", _modules_constructing_a_decode_config(), ids=lambda p: p.stem)
def test_pid_node_reads_the_setting_exactly_once(module_path: Path) -> None:
    """The estimate and the decode must be fed from one read, so they cannot disagree."""
    source = module_path.read_text(encoding="utf-8")
    reads = source.count(f"context.config.get().{_FLAG}")
    assert reads == 1, (
        f"{module_path.name}: reads the setting {reads} times; a single read keeps the working-memory "
        "estimate and the decode in lockstep."
    )
