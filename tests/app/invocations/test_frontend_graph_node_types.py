"""The graphs webv2 compiles must be buildable from the invocations this backend registers.

The frontend builds generation graphs from its own per-base tables and never imports a backend
schema, so a node type or field that moves, is renamed, or loses its module import fails at
*enqueue* time with a validation error against a graph the user cannot edit. Nothing in either test
suite sees that on its own: the frontend only knows the strings it emits, and the backend only knows
the invocations it has.

The contracts in between are `generateGraphNodeTypes.json` and `videoGraphNodeTypes.json`, written
by `graphCoverage.test.ts` and `videoGraphCoverage.test.ts` (regenerate with `vitest -u`). Each
records every node type, every edge field, and every *literal* input value — the ones webv2 writes
straight onto a node, with no edge to record them — that its compiler produces for every supported
architecture. This module checks both against the real registry.

The literal values are checked against the field's own annotation, not merely against its name. That
is where the frontend is blindest: it picks a scheduler from its own `FLOW_SCHEDULER_OPTIONS` while
`ernie_image_denoise.scheduler` is a three-value `Literal`, and nothing but this comparison relates
the two.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import TypeAdapter, ValidationError

from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationRegistry
from invokeai.app.services.shared.graph import *  # noqa: F401 F403 -- imports all invocations, populating the registry
from invokeai.app.services.shared.graph import are_connection_types_compatible

_WEBV2 = Path(__file__).parents[3] / "invokeai" / "frontend" / "webv2" / "src" / "features"

CONTRACT_PATHS = {
    "generate": _WEBV2 / "generation" / "core" / "__snapshots__" / "generateGraphNodeTypes.json",
    "video": _WEBV2 / "video" / "core" / "__snapshots__" / "videoGraphNodeTypes.json",
}

CONTRACTS: dict[str, dict[str, Any]] = {
    name: json.loads(path.read_text(encoding="utf-8")) for name, path in CONTRACT_PATHS.items()
}

# The two panels compile different graphs from different tables, so each contract is checked on its
# own; every assertion below is parameterized over both.
BASE_CASES = [(name, base) for name, contract in CONTRACTS.items() for base in sorted(contract["byBase"])]
NODE_CASES = [
    (name, node_type) for name, contract in CONTRACTS.items() for node_type in sorted(contract["fieldsByNodeType"])
]


def _fields(contract: str, node_type: str) -> dict[str, Any]:
    return CONTRACTS[contract]["fieldsByNodeType"][node_type]


def _invocation(node_type: str) -> type[BaseInvocation]:
    cls = InvocationRegistry.get_invocations_map().get(node_type)
    assert cls is not None, (
        f"webv2 compiles '{node_type}' into a generation graph but no invocation is registered under "
        f"that type. Either the node was renamed, or its module is no longer imported — see "
        f"tests/app/invocations/test_node_discovery.py."
    )
    return cls


CONNECTION_CASES = [
    (name, connection) for name, contract in CONTRACTS.items() for connection in sorted(contract.get("connections", []))
]


@pytest.mark.parametrize(
    ("contract", "connection"), CONNECTION_CASES, ids=lambda case: case if isinstance(case, str) else str(case)
)
def test_every_connection_the_panel_compiles_is_one_the_queue_will_accept(contract: str, connection: str) -> None:
    """Field NAMES matching is not enough: a float wired into an int names two real fields and is
    still refused, and `Graph.validate_self` runs on every enqueue -- so the whole generation dies
    at submit, before anything runs, with no partial output to diagnose from. Checking the names
    alone let exactly that ship in a video extension (`extract_video_range.fps` is a float,
    `video_concat.fps` an `Optional[int]`); both sides of the fixture recorded the field happily.
    """
    source_type, source_field, destination_type, destination_field = connection.split(" ")
    source_output = _invocation(source_type).get_output_annotation()
    destination = _invocation(destination_type)

    assert source_field in source_output.model_fields, (
        f"webv2 reads '{source_field}' off a '{source_type}' node, which its output does not have."
    )
    assert destination_field in destination.model_fields, (
        f"webv2 wires into '{destination_type}.{destination_field}', which that node does not have."
    )

    from_annotation = source_output.model_fields[source_field].annotation
    to_annotation = destination.model_fields[destination_field].annotation

    assert are_connection_types_compatible(from_annotation, to_annotation), (
        f"webv2 connects {source_type}.{source_field} ({from_annotation}) into "
        f"{destination_type}.{destination_field} ({to_annotation}), which the queue refuses as an "
        f"invalid edge at enqueue."
    )


# The number of architectures each panel supports. A contract that silently stopped being
# regenerated would make every assertion below vacuous, so the count is pinned: a new architecture
# updates this number and the fixture in the same commit.
SUPPORTED_BASE_COUNTS = {"generate": 15, "video": 3}

# Floors separating "the fixture records literals" from "it records a fraction of them". Set just
# under the current numbers (generate: 58 node types carrying 252 scalar values; video: 25 carrying
# 142) rather than far below them: a recording bug that dropped most of the values while keeping the
# names would otherwise leave every annotation check below passing over a short list. They are not
# per-architecture expectations, so removing an architecture means lowering them deliberately.
LITERAL_FLOORS = {"generate": (55, 240), "video": (23, 135)}


@pytest.mark.parametrize("contract", sorted(CONTRACTS), ids=lambda contract: contract)
def test_the_contract_covers_every_supported_base(contract: str) -> None:
    by_base = CONTRACTS[contract]["byBase"]

    assert len(by_base) == SUPPORTED_BASE_COUNTS[contract]
    assert all(entry["nodeTypes"] for entry in by_base.values())


@pytest.mark.parametrize("contract", sorted(CONTRACTS), ids=lambda contract: contract)
def test_the_contract_records_the_values_webv2_writes_onto_nodes(contract: str) -> None:
    """The literal section going vacuous is the failure mode the edge-only contract already had.

    A recording bug that dropped the values while keeping the names would leave every annotation
    check below passing over an empty list.
    """
    fields_by_node_type = CONTRACTS[contract]["fieldsByNodeType"]
    with_literals = [node_type for node_type, entry in fields_by_node_type.items() if entry["literalInputs"]]
    recorded_values = sum(
        len(values) for entry in fields_by_node_type.values() for values in entry["literalInputs"].values()
    )
    node_floor, value_floor = LITERAL_FLOORS[contract]

    assert len(with_literals) >= node_floor
    assert recorded_values >= value_floor


@pytest.mark.parametrize(("contract", "base"), BASE_CASES, ids=lambda value: value)
def test_every_node_type_a_base_compiles_is_registered(contract: str, base: str) -> None:
    missing = [
        node_type
        for node_type in CONTRACTS[contract]["byBase"][base]["nodeTypes"]
        if node_type not in InvocationRegistry.get_invocations_map()
    ]
    assert missing == [], f"webv2's '{base}' graph uses unregistered node types: {missing}"


@pytest.mark.parametrize(("contract", "node_type"), NODE_CASES, ids=lambda value: value)
def test_edge_destination_fields_are_real_invocation_fields(contract: str, node_type: str) -> None:
    """An edge into a field the invocation does not have is rejected when the graph is enqueued."""
    cls = _invocation(node_type)
    unknown = sorted(set(_fields(contract, node_type)["inputs"]) - set(cls.model_fields))
    assert unknown == [], f"webv2 wires edges into unknown inputs on '{node_type}': {unknown}"


@pytest.mark.parametrize(("contract", "node_type"), NODE_CASES, ids=lambda value: value)
def test_edge_source_fields_are_real_output_fields(contract: str, node_type: str) -> None:
    """The other half: an edge out of a field the invocation's output does not expose."""
    cls = _invocation(node_type)
    output_fields = set(cls.get_output_annotation().model_fields)
    unknown = sorted(set(_fields(contract, node_type)["outputs"]) - output_fields)
    assert unknown == [], f"webv2 wires edges out of unknown outputs on '{node_type}': {unknown}"


def _annotation(cls: type[BaseInvocation], field_name: str) -> Any:
    """The field's annotation together with the constraints declared beside it.

    `multiple_of=16` on a denoise node's `width` lives in `FieldInfo.metadata`, not in the
    annotation, and it is exactly the kind of bound the frontend restates in its own table
    (`BASE_GENERATION['ernie-image'].dimensions.grid`) with nothing relating the two.
    """
    field = cls.model_fields[field_name]
    return Annotated[(field.annotation, *field.metadata)] if field.metadata else field.annotation


SILENTLY_IGNORED: dict[str, set[str]] = {
    # `color_compensation` is a field of `i2l`, the *encode* node — legacy web sets it only there,
    # on the img2img/inpaint/outpaint paths. webv2 writes it onto the `l2i` decode node instead
    # (`graph.ts`, `l2iProps`), where no such field exists, so the Generate tab's SDXL colour
    # -compensation toggle changes nothing about the generated image. Invocations ignore extra
    # fields rather than rejecting them, which is why this never surfaced as an enqueue error.
    "l2i": {"color_compensation"},
}
"""Inputs webv2 writes that the invocation does not declare.

Not an exemption list: the assertion below is an equality, so a new mismatch fails here and so does
fixing one of these — the entry has to be deleted with the fix. Each is a silent no-op today,
because `BaseInvocation` does not forbid extra fields.
"""


@pytest.mark.parametrize(("contract", "node_type"), NODE_CASES, ids=lambda value: value)
def test_literal_inputs_are_real_invocation_fields(contract: str, node_type: str) -> None:
    """The half no edge records: a field webv2 sets directly, with no edge to point at it."""
    cls = _invocation(node_type)
    if cls.model_config.get("extra") == "allow":
        # `CoreMetadataInvocation` is an open bag by design — `invoke` dumps whatever was set on it
        # into the image's metadata record — so there is no closed field set to check against.
        pytest.skip(f"'{node_type}' accepts extra fields by design")

    unknown = sorted(set(_fields(contract, node_type)["literalInputs"]) - set(cls.model_fields))
    assert unknown == sorted(SILENTLY_IGNORED.get(node_type, set())), (
        f"webv2 sets inputs '{node_type}' does not declare: {unknown}. They are ignored rather than "
        f"rejected, so whatever they were meant to do silently does not happen."
    )


@pytest.mark.parametrize(("contract", "node_type"), NODE_CASES, ids=lambda value: value)
def test_literal_input_values_satisfy_the_field_they_are_written_to(contract: str, node_type: str) -> None:
    """A name that still exists but no longer accepts the value is the same enqueue failure.

    The scheduler fields are the motivating case: webv2 offers its own option list per base, the
    node declares a `Literal`, and until the two are compared here an option the node never accepted
    reaches the user as a validation error after they press Invoke.
    """
    cls = _invocation(node_type)
    rejected: list[str] = []

    for field_name, values in sorted(_fields(contract, node_type)["literalInputs"].items()):
        if field_name not in cls.model_fields:
            continue  # Reported by the test above; a missing field has no annotation to check.
        adapter = TypeAdapter(_annotation(cls, field_name))
        for value in values:
            try:
                adapter.validate_python(value)
            except ValidationError as error:
                rejected.append(f"{field_name}={value!r}: {error.errors()[0]['msg']}")

    assert rejected == [], f"webv2 writes values '{node_type}' rejects: {'; '.join(rejected)}"
