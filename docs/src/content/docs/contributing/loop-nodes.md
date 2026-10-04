---
title: Loop Nodes Architecture
---

This page records the implementation contract for the collection-based `For` and `ForReturn` nodes. The durable
contract is shared by the backend graph executor, saved workflow format, and workflow editor.

## Current execution-engine shape

The author-time graph contract is unchanged. `GraphExecutionState` remains the execution façade: it owns graph
validation, prepared execution-node identity, frame and iteration-path metadata, persistence, and the boundary between
invocation results and scheduling.

Readiness is split by admitted topology:

- `ExecutionPlan` and `ExecutionScheduler` provide opaque, deterministic node readiness and completion. They do not
  choose successors by node type or accept literal successor-node IDs.
- `_GenericGraphSchedulerAdapter` projects existing graph state into that scheduler for admitted fresh graphs and for
  the ordinary static-DAG/legacy compatibility path.
- `_GenericForPlanner`, the bounded `Iterate` planners, and the frame-scoped `If` dependency code supply the
  control-flow-specific preparation and activation rules for their exact admitted shapes.
- `_ExecutionMaterializer` remains the compatibility owner for unsupported topologies and legacy snapshots.
- `CallSavedWorkflowInvocation` declares one capability-bound `spawn_execution` plus matching `await` effect (or a
  capability-bound `fail`). `GraphExecutionState` persists the pending lifecycle effects and generic child dependency;
  `WorkflowCallCoordinator` and `WorkflowCallQueueLifecycle` remain the durable owner of saved-workflow child queue
  rows, statuses, events, authorization, capacity, cancellation, recovery, retry, and parent/child projection.

Control-flow nodes remain concrete `BaseInvocation` subclasses. They do not inherit from a new control-flow engine
hierarchy, and they do not return a literal next-node ID. Internal planners and records may share implementation
helpers, while successor readiness is selected from frame-scoped dependencies and effects.

Ownership is topology-dependent: admitted fresh `If` branches require matching activation tokens; bounded planners own
admitted loop expansion and continuation. Compatibility adapters own the remaining shapes. Execution ledgers are
internal persistence data, excluded from ordinary model serialization and public schemas. Version-2
`dump_execution_state()` rebuilds references and ordinary output tokens, retains activation tokens and effects needed
for resume, and omits terminal lifecycle records and completed child dependencies. They add no author-time ports or
frontend/backend external fields.

## Core contract

`For` is a bounded collection loop, not a general `While` node. Its source is one `collection: list[Any]` input. Each
iteration exposes `item`, `index`, `total`, and `state`; the final execution surface exposes `output_collection` and
`final_state`. `ForReturn` closes one iteration and may provide an output item, updated state, and a
`continue_condition`.

Loop state is explicit `LoopState` graph data. It is copied and serialized with normal invocation inputs and results;
it is not stored in transient process-local context. If a return omits state, the previous state carries forward. A
missing or `None` continue condition continues; `False` finalizes the current loop after recording its output and
state.

The loop is sequential. A body failure or cancellation stops later iterations and does not release partial final-scoped
outputs. An empty collection is successful: no body node runs, `output_collection` is empty, and `final_state` is the
provided initial state or an empty state.

## Durable loop linkage

Every `For` and `ForReturn` pair has one canonical runtime `loop_linkage` edge:

```text
For.loop_linkage - - - - - - - - - - - - - - - - - > ForReturn.loop_linkage
For.item -> body path -> ForReturn.output
```

This edge is an association, not executable data flow. It is excluded from ordinary input propagation, cycle detection,
and scheduling. The backend requires exactly one outgoing linkage for every `For`, exactly one incoming linkage for
every `ForReturn`, and the exact `For.loop_linkage` to `ForReturn.loop_linkage` endpoints. Default edges using the
reserved `loop_linkage` field are invalid.

Authoring workflow JSON may represent the association through a one-to-one connector alias:

```text
For.loop_linkage -> connector.in -> connector.out -> ForReturn.loop_linkage
```

Every connector on that path must have exactly one input and one output. The path cannot branch, be reused as ordinary
data flow, or terminate at a different node. Graph construction canonicalizes a complete valid alias to one direct
runtime `loop_linkage` edge. No loop identity or body metadata is inferred from arbitrary topology or migrated from
unrelated fields.

## Body and output scopes

Iteration-scoped outputs (`item`, `index`, `total`, and `state`) define the loop body. Final-scoped outputs
(`output_collection` and `final_state`) are available only after the matching loop context completes. Body nodes must
terminate at the linked `ForReturn`; they cannot escape directly to after-loop nodes. Final outputs cannot feed back
into the loop body.

Downstream consumers, including a selected `If` branch, may consume `output_collection` directly. Its parent-frame scope
is preserved through branch selection, so a `final_state` edge is not needed just to make those consumers run.

`ForReturn.output` and `ForReturn.state` are scheduler-facing result fields and are hidden as downstream editor
outputs. They are still retained in execution results for aggregation, persistence, and resume. Ordinary state helper
nodes (`state_empty`, `state_get`, `state_set`, and `state_merge`) carry explicit `LoopState` values through the body.

## Supported nested shapes

Nested `For` boundaries are supported recursively when each inner boundary has its own direct linkage and matching
`ForReturn`. The inner final collection may feed the parent return directly or through an ordinary parent-scoped
continuation. Independent inner loops must all feed one explicit fan-in continuation; collection concatenation,
zipping, or Cartesian semantics come from the connected collection operation, not from loop scheduling.

A bounded single internal `Iterate` is supported for the canonical outer-`For` shape: one ordinary preparation node
converts `For.item` to the inner collection, one ordinary body node consumes `Iterate.item`, and one `Collect`
collapses that item dimension before the parent `ForReturn`:

```text
For.item -> preparation -> Iterate.collection
Iterate.item -> body -> Collect.item
Collect.collection -> ForReturn.output
```

The fresh generic scheduler admits this exact shape when the outer collection is a non-empty literal or one of the
supported input-driven outer-collection variants, and there is one final consumer of `For.output_collection`. Each
outer frame is isolated, including when the preparation node produces an empty inner collection. A checkpoint after a
nested iterator boundary restores the generic class-drain state and continues the same frame/stream order. Valid
topologies outside these bounds retain compatibility ownership; escaped body paths fail graph validation.

The exact serial two-`Iterate` body is also supported:

```text
For.item -> preparation1 -> Iterate1.collection
Iterate1.item -> preparation2 -> Iterate2.collection
Iterate2.item -> body -> Collect.item
Collect.collection -> ForReturn.output
```

Generic admission requires a non-empty literal outer collection, those two ordinary preparation nodes, one ordinary
body node, and one ordinary `For.output_collection` consumer: nine nodes and nine edges including linkage. This does
not admit sibling iterators, fan-in, or additional control-flow nodes.

Unsupported shapes, including independent iterator-derived body inputs, mixed nested `For`/`Iterate` bodies, escaping
body paths, ambiguous returns, and arbitrary cyclic graphs, are rejected before execution.

## Generic `For` routing matrix

`_GenericForPlanner` owns these fresh shapes. `_ExecutionMaterializer` remains the compatibility facade for every other
shape and for explicitly forced compatibility runs:

- One flat `For`/`ForReturn`: literal empty or non-empty collection, supported non-empty input producer, or the exact
  four-node/four-edge empty input producer with no downstream consumer.
- Canonical two-level nested `For` with a non-empty literal outer collection or supported non-empty input producer;
  empty input-driven outer results remain compatibility-owned.
- Exact two-sibling nested `For` fan-in through `CollectionConcat`, with a non-empty literal outer collection.
- Exact serial three-level nested `For`, including its statically non-empty `CollectionConcat` producer variant.
- Exact serial four-level nested `For`.
- Exact bounded outer `For`/`Iterate`/`Collect`.
- Exact serial two-`Iterate` nested outer-`For` shape with one ordinary final consumer.

Valid shapes outside this admission matrix retain compatibility ownership, including legacy snapshots, unsupported
input producers, extra nodes or consumers, fan-out, nested `For` depth five or greater, unsupported sibling or deeper
shapes, and saved-workflow or `If`-containing `For` graphs. Malformed linkage, invalid output-scope edges, and
unsupported mixed bodies fail graph validation; a compatibility route does not make them executable. No invocation
emits or requires a literal successor node ID. This matrix changes no invocation, API, saved-workflow, frontend, or
generated-schema contract.

## Persistence and validation

Prepared execution nodes, source/prepared mappings, iteration paths, results, indegrees, and finalized loop contexts
are persisted through `GraphExecutionState`. Runtime-only queues and metadata are rebuilt when state is rehydrated.
Finalization is keyed by the loop source and its parent iteration path so nested or repeated contexts cannot mix output
collections or state.

The execution-engine seam is additive to this loop contract. The session runner applies each result through
`GraphExecutionState.apply()` using a stable execution reference and records frame-aware output tokens; accepted
`For`/`ForReturn` continuation state is mirrored by a typed, frame-scoped `ContinuationRecord`. `ForInvocation` and
`ForReturnInvocation` also declare one validated, frame-scoped `continuation` effect per prepared invocation. The `For`
effect starts the `for` continuation with the current iteration and state; the `ForReturn` effect completes it with
output, state, and the continue decision. These effects are persisted under the invocation reference. Session-built
effect references carry graph-state, durable-frame, iteration-path, and workflow-call-depth identity, and stale or
cross-scope effects are rejected before mutation. They never encode `loop_linkage` as a data token. A fresh graph with
exactly one static flat `For`/`ForReturn` pair, including an empty literal collection, a supported non-empty
input-driven collection producer, or the exact bounded empty input-driven producer shape, now uses the generic
scheduler adapter for readiness and continuation transitions. Graph state owns the generic continuation boundary.
`_GenericForPlanner` owns admitted For-specific preparation, materializes the next body iteration, preserves frame
paths and carried state, and finalizes nested completion. It shares only low-level execution-node construction
mechanics with the compatibility materializer. An empty literal collection completes through the existing synthetic
terminal `For` result without running the body or `ForReturn`. The generic scheduler remains opaque and never receives
a literal successor node ID. The compatibility continuation bridge is retained only for unsupported loop shapes and
explicitly legacy-loaded snapshots. Empty input-driven shapes outside the exact four-node/four-edge bounded topology,
unsupported input-driven outer collections, five-level-or-deeper or unsupported sibling nested loops, and unsupported
mixed control flow remain on the compatibility scheduler; the exact fresh two-sibling `For`/`ForReturn`
`CollectionConcat` fan-in shape, the narrow canonical two-level, and exact fresh three-level and four-level serial
nested-`For` shapes (one literal outer collection, or the exact statically non-empty `CollectionConcat` producer for
the three-level case, one direct child at each level, no continuation nodes) and the exact bounded
outer-`For`/`Iterate`/`Collect` shape are now generic-routed. This additive effect seam does not claim generic
scheduling for the remaining shapes. `Iterate` records ordered item tokens in a closed `StreamBuffer`; an empty
`Iterate` records an explicit empty close. A direct `Iterate.item` consumer waits for the canonical stream to close,
then `Collect` consumes its ordered values; a missing stream falls back to materialized results for legacy snapshots.
The exact fresh four-node `literal collection source -> Iterate -> one ordinary body -> Collect` shape and any number
of ordinary downstream consumers from `Collect.collection` use a private graph-state planner for prepared-copy
expansion, iteration paths, and the empty-stream barrier. The exact fresh
two-source/two-`Iterate`/shared-`Collect.item` and exact fresh three-source/three-`Iterate`/shared-`Collect.item`
fan-in shapes also use the planner for independent stream expansion, closure gating, and deterministic source-ordered
hydration. The exact fresh two-branch body-mediated shape, `source_a -> Iterate_a -> body_a -> Collect.item` plus
`source_b -> Iterate_b -> body_b -> Collect.item`, is also planner-owned: source IDs are ordered lexically, branch
order is preserved, and `Collect` waits for both streams to close, including empty streams. This is a private bounded
planner case with exactly seven nodes and six ordinary edges: two inputless ordinary sources, two `Iterate` nodes, two
ordinary bodies, and one `Collect`. It has no `Collect.collection` input, downstream consumer, or extra topology. Three
or more body-mediated branches, fan-in with four or more direct branches, unsupported nested or input-driven iterator
topologies, and mixed control flow outside the bounded per-item `Iterate`/`If`/`Collect` topology remain on the
compatibility materializer. The exact admitted serial nested-`Iterate` chains and producer-driven bounded
`For`/`Iterate`/`Collect` variants use the generic planner described above. Direct `Iterate`/`Collect`-only graphs and
the exact bounded nested shape use the generic scheduler adapter: its readiness predicate waits for the canonical
stream to close, and completion mirrors Iterate outputs into that ledger before releasing Collect. Rehydration restores
the active class-drain boundary so nested stream order is preserved across dump/load. For the exact fresh direct shape,
the planner owns expansion, downstream admission, and empty closure atomically: a failed expansion leaves no partial
prepared copies to be resumed. Versioned queue checkpoints and retries preserve the existing durable execution-state
boundary and receive fresh execution identities. Materialization still owns those responsibilities for fallback shapes.
Mixed control-flow outside the exact bounded admitted topology and queue lifecycle remain compatibility-owned.
`loop_linkage` remains association metadata and never becomes a data token.

The supported flat-loop behavior preserves generic/compatibility parity for successful execution, carried and replaced
state, early break, empty and input-driven fallback routing, `None` output items, body/return failure, partial
rehydration, frame identity, and SQLite cancellation/retry isolation. Continuation effects are normalized to JSON
values and validated before mutation: exact duplicate effects in one batch are retained once, while conflicting
terminal payloads (including type-distinct JSON values) are rejected. Before scheduler completion,
`ForInvocationOutput` and `ForReturnInvocationOutput` continuation fields must match their prepared nodes; the `For`
item must match its prepared collection item, and effects must independently match those values. Current versioned
snapshots must include an execution-effect ledger: every ledger key must name an executed prepared node, and pending
nodes, unknown references, missing executed markers, missing continuation effects, and malformed ownership are
rejected. Persisted `For` outputs are checked against prepared iteration data and, after finalization, the
authoritative returned collection and final state. Unversioned legacy snapshots retain their compatibility loader. A
`For` start effect must match the prepared index, collection total, and state; malformed effects leave the graph state
unchanged. Unsupported shapes retain the compatibility continuation/materialization owner.

`GraphExecutionState.complete()` remains the compatibility entry point for callers that already have a node result. Its
first JSON-safe completion uses the same validated, atomic ledger path as `apply()`, including output tokens and
synthetic `For`/`ForReturn` continuation effects. It still permits historical idempotent result replacement for an
already-applied node; arbitrary in-memory values that cannot be represented in the JSON ledger remain a scheduler-only
compatibility case and cannot be persisted. When an explicitly unversioned, partially completed legacy snapshot is
loaded, missing flat-loop continuation buckets are synthesized before a versioned re-save so the next load remains
valid. A loaded terminal legacy state may retain its empty in-memory ledger for compatibility, but
`dump_execution_state()` upgrades missing loop buckets in the versioned snapshot.

`IfInvocation` now declares the same seam for branch activation: it emits one frame-scoped activation token for the
selected branch, and the graph state validates and persists it after invocation. Fresh graphs with one ordinary-node
`If`, the exact one-level nested shape with two `If` nodes where the inner value feeds one outer branch and has no
other consumer, the exact bounded three-`If` chain with direct inner-to-middle-to-outer branch edges, the exact
four-`If` nested chain, or exactly two, exactly three, or exactly four independent ordinary-node sibling `If`s compile
opaque, frame-local activation dependencies in graph state. The three- and four-`If` admissions require `default`
edges, all branch inputs (`condition`, `true_input`, `false_input`) on each `If`. The three-`If` chain permits one
extra `middle If.value` edge to an ordinary leaf consumer with no outputs; that leaf inherits only the outer branch
dependency. Other inner/middle outputs must be direct nested branch edges, and the four-`If` chain permits no extra
output fan-out. Five-or-more nested `If`s, other fan-out, mixed shapes outside the bounded per-item
`Iterate`/`If`/`Collect` topology, loop-containing `For`/`ForReturn`, saved-workflow, legacy, and five-or-more sibling
shapes use the compatibility scheduler with the dedicated `_IfActivationController` fallback. Legacy snapshots retain
their generic compatibility projection. Legacy skipped-state projection remains only for old snapshots. Fresh
materialization prepares the condition boundary, resolves the activation token, and attaches only the selected branch
input; unselected branch nodes remain unprepared. The generic and compatibility adapters consume the same decisions and
append-only execution edges; no type-specific branch scheduler prunes edges or propagates skips. Legacy skipped-state
metadata remains only for snapshots that already contain the old projection. `apply()` replaces the activation token by
stable identity, and activation effects are excluded from stream handling. Graph state derives branch membership for
the supported fresh shape; the fallback controller retains it for unsupported and legacy shapes. Author-time activation
ports and literal successor IDs remain absent, while unsupported loop shapes and saved-workflow control flow remain on
their compatibility paths until differential coverage proves their generic replacements.

The engine keeps compatibility materialization and queue readiness for loop-containing graphs outside the routing
matrix and for existing snapshots. Admitted fresh `For` graphs use the generic opaque plan/scheduler plus
`_GenericForPlanner`; `_ExecutionNodeBuilder` supplies shared low-level graph mutation mechanics. Ordinary static DAGs,
supported fresh `If` graphs, direct `Iterate`/`Collect`-only graphs, and the exact bounded mixed
`Iterate`/`If`/`Collect` graph use the generic adapter; unsupported or legacy shapes retain their compatibility routes.
No activation, stream, or continuation ports are added to author-time graph JSON. Exact fresh serial nested-`Iterate`
chains through eight levels use the generic planner for ordered frame expansion, empty-stream closure, checkpoint
rehydration, source completion, and failure parity. Nine-level or deeper chains, five-level-or-deeper or unsupported
sibling nested loops, and unsupported mixed shapes remain compatibility-owned. The exact fresh two-sibling
`For`/`ForReturn` `CollectionConcat` fan-in shape, the exact fresh three-level serial nested-`For` shape with its
bounded static `CollectionConcat` producer, and the exact fresh four-level serial nested-`For` shape are
generic-routed. The execution engine does not modify any file under `invokeai/frontend/...`, including generated
schemas; the existing frontend/backend external interface remains stable. Branch-membership analysis remains internal,
while fresh execution no longer creates skipped-state projection and old snapshots retain it for compatibility.

The frontend and backend validate the same boundary rules. Saved workflows preserve node types, field handles, and the
direct linkage edge. The current invocation templates provide output-scope metadata when a workflow is loaded. The
editor's boundary overlay and contextual `ForReturn` picker are presentation aids; they do not replace whole-graph
validation.

Collection helpers are ordinary explicit nodes: `CollectionConcat` preserves left-to-right order and accepts unequal
lengths, `CollectionZip` requires equal lengths, and `CollectionCartesian` produces deterministic
left-major/right-minor pairs with a 100,000-pair limit. They do not add implicit loop dimensions.
