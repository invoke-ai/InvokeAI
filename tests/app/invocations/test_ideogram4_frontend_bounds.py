"""webv2's Ideogram 4 controls are bounded by the same numbers the denoise node validates.

These three bounds are not served: they belong to controls only Ideogram 4 has, and the shared
guidance slider never reaches the field (see `NO_GUIDANCE_SLIDER` in
`tests/backend/architectures/test_guidance_range.py`). So the frontend keeps them as constants, and
this reads both sides -- the constants from `settings.ts`, the bounds from the node's own schema --
so a change to either fails here instead of at enqueue. All three were wrong once: `mu` offered 0..10
against a node that accepts -4..4.
"""

import re
from pathlib import Path

import pytest

from invokeai.app.invocations.ideogram4.ideogram4_denoise import Ideogram4DenoiseInvocation

REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_TS = REPO_ROOT / "invokeai/frontend/webv2/src/features/generation/core/settings.ts"


def _frontend_constant(name: str) -> float:
    match = re.search(rf"export const {name} = (-?\d+(?:\.\d+)?);", SETTINGS_TS.read_text(encoding="utf-8"))
    assert match is not None, f"{name} is not a numeric constant in {SETTINGS_TS.name}"
    return float(match.group(1))


def _node_bounds(field: str) -> tuple[float, float]:
    """`ge`/`le` of an optional numeric field, which sit on the numeric branch of its `anyOf`."""
    schema = Ideogram4DenoiseInvocation.model_json_schema()["properties"][field]
    branches = schema.get("anyOf", [schema])
    numeric = next(branch for branch in branches if "minimum" in branch or "maximum" in branch)
    return float(numeric["minimum"]), float(numeric["maximum"])


@pytest.mark.parametrize(
    ("field", "prefix"),
    [("steps", "IDEOGRAM4_STEPS"), ("guidance_scale", "IDEOGRAM4_GUIDANCE"), ("mu", "IDEOGRAM4_MU")],
)
def test_the_control_bounds_match_the_node(field: str, prefix: str) -> None:
    assert (_frontend_constant(f"{prefix}_MIN"), _frontend_constant(f"{prefix}_MAX")) == _node_bounds(field)
