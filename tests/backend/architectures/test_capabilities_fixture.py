"""The capabilities response webv2 is built against.

webv2 has no generated OpenAPI types -- it hand-writes its wire DTOs -- so nothing on the frontend
side would notice a renamed field, a dropped facet or a new architecture. This fixture is the
contract between the two, and it earns its keep twice: `scripts/mock-backend.mjs` serves it, so the
journey tests exercise the same payload shape the real route returns, and the frontend's unit tests
map it into their own types.

The mirror image of `tests/app/invocations/test_frontend_graph_node_types.py`, which reads a
frontend-written file and checks it against the backend registry. This one goes the other way.
"""

import json
import os
from pathlib import Path
from typing import Any

from invokeai.backend.architectures.capabilities import architecture_capabilities

FIXTURE_PATH = (
    Path(__file__).parents[3]
    / "invokeai"
    / "frontend"
    / "webv2"
    / "src"
    / "features"
    / "generation"
    / "core"
    / "__fixtures__"
    / "architectureCapabilities.json"
)

REGEN_HINT = (
    f"Regenerate it with:\n"
    f"    REGEN_CAPABILITIES_FIXTURE=1 pytest {Path(__file__).name}\n"
    f"and commit the result, so webv2's tests and mock backend see what this backend serves."
)

# The fixture is written by `json.dumps(indent=2)` below, which oxfmt would reflow -- so following
# the hint above would redden webv2's `format:check` on a file nobody hand-edited. It is listed in
# `.oxfmtrc.json`'s `ignorePatterns` for that reason, alongside `generateGraphNodeTypes.json`, the
# other generated cross-stack artifact. Machine-written files are not the formatter's to own; if
# that entry is removed, this writer has to emit oxfmt's exact output instead.


def _rendered() -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in architecture_capabilities()]


def test_the_fixture_matches_what_the_endpoint_serves() -> None:
    rendered = _rendered()

    if os.environ.get("REGEN_CAPABILITIES_FIXTURE"):
        FIXTURE_PATH.write_text(
            json.dumps(rendered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )

    assert FIXTURE_PATH.exists(), f"{FIXTURE_PATH} is missing. {REGEN_HINT}"
    committed = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert committed == rendered, (
        f"The committed capabilities fixture no longer matches this backend.\n{REGEN_HINT}\n"
        "If a field was renamed or removed, webv2's hand-written wire type in "
        "`features/generation/core/` needs the same edit -- that is what this test is for."
    )


def test_every_architecture_webv2_can_generate_with_has_a_base_row() -> None:
    """The half that matters at runtime: a missing row means the Generate panel has no policy.

    Read from the graph-builder contract webv2 already writes for the invocation check, so the two
    cross-stack artifacts stay in step rather than each carrying their own list of architectures.
    """
    contract_path = FIXTURE_PATH.parents[1] / "__snapshots__" / "generateGraphNodeTypes.json"
    if not contract_path.exists():
        # The frontend test that writes it may not have run in this checkout; the invocation-side
        # test owns that failure, so do not duplicate it here.
        return

    generatable = set(json.loads(contract_path.read_text(encoding="utf-8"))["byBase"])
    served = {row["base"] for row in _rendered() if row["variant"] is None}

    assert generatable <= served, sorted(generatable - served)
