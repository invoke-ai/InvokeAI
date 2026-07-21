"""Verify the bundled Wan/video workflows agree with the invocation registry.

The pre-existing default workflows (SD1.5/SDXL/FLUX...) carry stale node versions and
even removed node types — the editor tolerates this with "node needs update" badges, so
they are deliberately NOT checked here. The workflows this PR ships should not start
life stale: every embedded node must exist and carry the invocation's current version,
and every embedded input must be a real field on the invocation. This is exactly the
check that would have caught the wan_ref_image_encoder 1.0.0/1.1.0 embeds shipped while
the invocation was at 1.2.0.
"""

import json
from pathlib import Path

import pytest

from invokeai.app.invocations.baseinvocation import InvocationRegistry
from invokeai.app.services.shared.graph import *  # noqa: F401 F403 -- imports all invocations, populating the registry

WORKFLOW_DIR = Path("invokeai/app/services/workflow_records/default_workflows")
WAN_WORKFLOWS = sorted(
    {p for p in WORKFLOW_DIR.glob("*.json") if "Wan" in p.name or "Video" in p.name},
    key=lambda p: p.name,
)


def test_wan_workflow_glob_finds_the_bundled_workflows() -> None:
    # Guard against the glob silently matching nothing (e.g. after a rename).
    assert len(WAN_WORKFLOWS) == 12


@pytest.mark.parametrize("workflow_path", WAN_WORKFLOWS, ids=lambda p: p.stem)
def test_bundled_workflow_nodes_match_invocation_registry(workflow_path: Path) -> None:
    invocations = InvocationRegistry.get_invocations_map()
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    for node in workflow["nodes"]:
        data = node["data"]
        node_type = data["type"]
        cls = invocations.get(node_type)
        assert cls is not None, f"{workflow_path.name}: node type '{node_type}' is not a registered invocation"

        current_version = cls.UIConfig.version
        assert data["version"] == current_version, (
            f"{workflow_path.name}: node '{node_type}' embeds version {data['version']} "
            f"but the invocation is at {current_version} — update the bundled workflow"
        )

        field_names = set(cls.model_fields.keys())
        for input_name in data.get("inputs", {}):
            assert input_name in field_names, (
                f"{workflow_path.name}: node '{node_type}' embeds unknown input '{input_name}'"
            )
