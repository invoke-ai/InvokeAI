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
from invokeai.app.services.shared.graph import are_connection_types_compatible

WORKFLOW_DIR = Path("invokeai/app/services/workflow_records/default_workflows")
WAN_WORKFLOWS = sorted(
    {p for p in WORKFLOW_DIR.glob("*.json") if "Wan" in p.name or "Video" in p.name},
    key=lambda p: p.name,
)


def test_wan_workflow_glob_finds_the_bundled_workflows() -> None:
    # Guard against the glob silently matching nothing (e.g. after a rename).
    # 12 Wan workflows + 6 MiniMax H3 video workflows + 10 LTX-2 video workflows.
    assert len(WAN_WORKFLOWS) == 28


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


@pytest.mark.parametrize("workflow_path", WAN_WORKFLOWS, ids=lambda p: p.stem)
def test_bundled_workflow_edges_are_ones_the_queue_will_accept(workflow_path: Path) -> None:
    """Every edge must connect field types the queue accepts, as `Graph.validate_self()` checks.

    This covers edge TYPING only -- not the cycle, duplicate-input or required-input checks
    `validate_self` also performs.

    A workflow whose wiring the queue refuses is worse than a broken one: it loads, it draws, and
    it fails only when the user presses Invoke. The trap is asymmetric numeric coercion -- `int` is
    accepted where a `float` is wanted but never the reverse -- so an edge like
    `extract_video_range.fps -> video_concat.fps` (float into `Optional[int]`) is rejected after
    the model has loaded. This is the same check the compiled-graph contract test applies to the
    panel; bundled workflows are hand-authored and need it more.
    """
    invocations = InvocationRegistry.get_invocations_map()
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    node_types = {node["data"]["id"]: node["data"]["type"] for node in workflow["nodes"]}

    for edge in workflow["edges"]:
        # A collapsed edge stands in for every connection between two collapsed nodes and names no
        # handles; there is nothing to type-check. Several bundled workflows outside this glob carry
        # them, and one saved from the editor would land here.
        if "sourceHandle" not in edge or "targetHandle" not in edge:
            continue
        source_type = node_types.get(edge["source"])
        target_type = node_types.get(edge["target"])
        assert source_type is not None, f"{workflow_path.name}: edge from unknown node {edge['source']}"
        assert target_type is not None, f"{workflow_path.name}: edge to unknown node {edge['target']}"

        output_cls = invocations[source_type].get_output_annotation()
        source_field = output_cls.model_fields.get(edge["sourceHandle"])
        target_field = invocations[target_type].model_fields.get(edge["targetHandle"])
        assert source_field is not None, f"{workflow_path.name}: '{source_type}' has no output '{edge['sourceHandle']}'"
        assert target_field is not None, f"{workflow_path.name}: '{target_type}' has no input '{edge['targetHandle']}'"

        assert are_connection_types_compatible(source_field.annotation, target_field.annotation), (
            f"{workflow_path.name}: the queue refuses "
            f"{source_type}.{edge['sourceHandle']} -> {target_type}.{edge['targetHandle']} "
            f"({source_field.annotation} into {target_field.annotation})"
        )


@pytest.mark.parametrize("workflow_path", WAN_WORKFLOWS, ids=lambda p: p.stem)
def test_every_exposed_field_has_a_value_instance_on_its_node(workflow_path: Path) -> None:
    """An exposed field with no `data.inputs` entry is silently dropped, and its value never runs.

    Legacy web rebuilds the linear form from `exposedFields` when the stored form is empty and
    DELETES every rebuilt element whose node carries no matching input instance, so the control
    never appears; `buildNodesGraph` separately reduces over `data.inputs` alone, so the field is
    also absent from the enqueued graph. A workflow exposing a required field it stores no instance
    for therefore loads with warnings, shows no control for it, and cannot be invoked at all -- the
    failure this catches, which node versions and edge types both look past.
    """
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    instances = {node["data"]["id"]: set(node["data"].get("inputs", {})) for node in workflow["nodes"]}

    for exposed in workflow.get("exposedFields", []):
        node_id, field_name = exposed["nodeId"], exposed["fieldName"]
        assert node_id in instances, f"{workflow_path.name}: exposes a field on unknown node {node_id}"
        assert field_name in instances[node_id], (
            f"{workflow_path.name}: exposes '{field_name}' on node {node_id}, which stores no value "
            f"instance for it — the control is dropped on load and the field never reaches the graph"
        )
