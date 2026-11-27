"""Model residency read from the server log: was every model a run used fully in VRAM?

The model cache logs one `Loaded model '<key>' (...) ... VRAM: <mb>MB (<p>%)` line (INFO) for every lock that moves
weights or leaves the model short of fully resident (`ModelCache._load_locked_model`). A lock that finds the model
already fully resident logs that line at DEBUG only, so warm runs usually show no line at all. The tracker therefore
keeps the latest line per model since the last cache flush: a run is residency-valid when every line in its log
window is resident (`resident`), no "VRAM budget ... short" warning appears in the window, and at least one of the
graph's models has been seen loaded with weights since the flush. A run without such a line is "residency unknown"
and invalid. A load under WEIGHTLESS_MB, such as a tokenizer submodel, has no weights to place: the cache logs it at
0.0 %, and it counts as resident but tells nothing about where the weights are.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from scripts.family_e2e.records import Residency

# `ModelCache._load_locked_model`. The device label is "cuda device #0", "cuda device", "mps device", ...
LOAD_LINE = re.compile(
    r"Loaded model '(?P<key>[^']+)' \([^)]*\) onto .+? in [\d.]+s\. "
    r"Total model size: (?P<total>[\d.]+)MB, VRAM: (?P<vram>[\d.]+)MB \((?P<percent>[\d.]+)%\)"
)
# Same method, first pass of a lock whose budget stays short after offloading everything offloadable.
SHORT_LINE = re.compile(r"VRAM budget for '(?P<key>[^']+)' is short by \d+MB")
# The log prints one decimal: "100.0%" covers everything from 99.95 %.
FULL_PERCENT = 99.95
# A tokenizer logs "Total model size: 0.03MB"; the smallest weighted submodels are hundreds of MB.
WEIGHTLESS_MB = 1.0


@dataclass(frozen=True)
class LogWindow:
    loads: list[Residency]
    short: list[str]
    """Cache keys named by "VRAM budget ... short" warnings."""


def parse_log(text: str) -> LogWindow:
    loads = []
    for match in LOAD_LINE.finditer(text):
        # Cache keys are `<model key>` or `<model key>:<submodel>`; model keys are UUIDs.
        model, _, submodel = match["key"].partition(":")
        loads.append(
            Residency(
                model=model,
                submodel=submodel or None,
                vram_mb=float(match["vram"]),
                total_mb=float(match["total"]),
                percent=float(match["percent"]),
            )
        )
    return LogWindow(loads=loads, short=[m["key"] for m in SHORT_LINE.finditer(text)])


def weightless(residency: Residency) -> bool:
    return residency.total_mb is not None and residency.total_mb < WEIGHTLESS_MB


def resident(residency: Residency) -> bool:
    """Whether the load left the model's weights fully in VRAM; a weightless load has none to place."""
    return residency.percent >= FULL_PERCENT or weightless(residency)


def _label(residency: Residency) -> str:
    return f"{residency.model}:{residency.submodel}" if residency.submodel else residency.model


class ResidencyTracker:
    """The latest residency of every model loaded since the last cache flush (empty_model_cache or a new server)."""

    def __init__(self) -> None:
        self._latest: dict[tuple[str, str | None], Residency] = {}

    def reset(self) -> None:
        self._latest.clear()

    def observe(self, window: LogWindow, graph_models: Iterable[str]) -> tuple[list[Residency], list[str]]:
        """Folds in one run's log window. Returns the residency of the graph's models and the invalid reasons."""
        for load in window.loads:
            self._latest[(load.model, load.submodel)] = load
        reasons = [
            f"residency: {_label(load)} loaded with {load.percent:.1f}% of its weights in VRAM"
            for load in window.loads
            if not resident(load)
        ]
        reasons.extend(f"residency: VRAM budget short for {key}" for key in window.short)
        models = set(graph_models)
        relevant = [r for (model, _), r in self._latest.items() if model in models]
        if all(weightless(r) for r in relevant):
            reasons.append("residency unknown: no load with weights for the graph's models since the last cache flush")
        reasons.extend(
            f"residency: {_label(r)} at {r.percent:.1f}% since an earlier run"
            for r in relevant
            if not resident(r) and r not in window.loads
        )
        return relevant, reasons
