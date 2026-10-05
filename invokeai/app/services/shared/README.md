# InvokeAI Graph - Design Overview

High-level design for the graph module. Focuses on responsibilities, data flow, and how traversal works.

## 1) Purpose

Provide a typed, acyclic workflow model (**Graph**) plus a runtime scheduler (**GraphExecutionState**) that expands
iterator patterns, tracks readiness via indegree (the number of incoming edges to a node in the directed graph), and
executes nodes from class-grouped ready queues. In normal execution, runtime expansion happens in a separate execution
graph instead of mutating the source graph. Ordinary static DAGs and legacy-shaped `If` graphs use the opaque
`ExecutionPlan` and deterministic `ExecutionScheduler` in `execution_engine/scheduler.py`. Fresh graphs with one
ordinary-node `If`, the exact one-level nested shape with two `If` nodes where the inner value feeds one outer branch
and has no other consumer, the exact bounded three-`If` chain where inner `value` feeds a middle branch and middle
`value` feeds an outer branch, the exact four-`If` nested chain where each `If.value` feeds one branch port of the next
`If`, or exactly two, exactly three, or exactly four independent ordinary-node sibling `If`s compile opaque,
frame-local activation dependencies in `GraphExecutionState`; unsupported fresh shapes use the compatibility scheduler
with the dedicated `_IfActivationController` fallback, while legacy snapshots retain their generic compatibility
projection. Current fresh sibling support includes exactly four independent `If`s with no nesting, fan-out, or other
control-flow nodes. The bounded three-`If` chain requires direct `default` edges, all three `If`s to have `condition`,
`true_input`, and `false_input` inputs. The inner `If` has only its direct nested branch output; the middle `If.value`
may also feed one ordinary leaf consumer with no outputs. That leaf inherits the outer branch dependency, not
middle-`If` polarity. The exact four-`If` chain has the same input and direct-edge requirements, with no extra output
fan-out. Five or more nested `If`s, other fan-out, mixed graphs outside the bounded per-item `Iterate`/`If`/`Collect`
topology, loop-containing, saved-workflow, legacy, and five-or-more sibling shapes remain on the compatibility
fallback. Legacy snapshots retain their generic compatibility projection and legacy skipped-state metadata. Legacy
skipped-state projection remains only for old snapshots. Fresh `If` materialization prepares the condition boundary
first, resolves the activation token, and attaches only the selected branch input. Unselected branch nodes are never
prepared, skipped, or added to fresh execution history. Both scheduler adapters consume these dependency decisions
without pruning execution edges or calling a type-specific branch scheduler. Legacy skipped-state metadata remains only
for snapshots that already contain the old projection. Branch-membership analysis remains an internal author-graph
admission decision. The focused fresh-shape ownership matrix also verifies that supported flat, nested, bounded
fan-out, and four-sibling graphs do not instantiate the compatibility controller or project fresh skipped nodes in
either scheduler route. This proves the current ownership boundary; it does not remove the controller, which remains
required for unsupported fresh shapes and legacy snapshots. The bounded mixed-control-flow gate admits one exact
per-item topology: exactly six nodes and seven edges consisting of an inputless ordinary collection source feeding
`Iterate.collection`, `Iterate.item` feeding `If.condition` and two distinct ordinary one-input branch adapters, those
adapters feeding `If.true_input` and `If.false_input`, and `If.value` feeding `Collect.item`. Only this topology uses
fresh generic scheduling and frame-local activation dependencies; alternating true/false items and empty collections
preserve ordered completion. `For`, `ForReturn`, saved-workflow calls, nested/sibling/fan-in/extra-node variants,
legacy snapshots, and all other mixed graphs remain compatibility-owned. This is backend-only and changes no
invocation, frontend, generated-schema, image, or external-interface contract. The token-authoritative topology is used
for every currently supported fresh shape listed above. The controller remains required for unsupported fresh shapes
and legacy snapshots. The execution ledgers (`execution_refs`, `execution_tokens`, `execution_effects`, and
`execution_child_dependencies`) are internal persistence data. They are not part of public API schemas or client
responses. Version-2 `dump_execution_state()` writes a compact projection: execution references and ordinary output
tokens are rebuilt from the prepared graph, mappings, and results; frame-scoped activation tokens remain durable;
effects remain durable when needed for stream, continuation, or active workflow-call recovery; completed workflow-call
lifecycle effects and completed child-dependency records are omitted. Attached child state is retained only while its
parent call is active. Direct `Iterate`/`Collect` graphs that do not contain `If`, `For`, `ForReturn`, or saved-workflow control flow
also use the generic adapter. Its adapter-level readiness predicate waits for canonical Iterate streams to close, and
generic completion mirrors each Iterate result into that ledger before releasing `Collect`. The exact fresh four-node
shape `literal collection source -> Iterate -> one ordinary body -> Collect`, plus any number of ordinary downstream
consumers from `Collect.collection`, is planned by a private graph-state planner, which owns its prepared-copy
expansion, iteration paths, downstream admission, and explicit empty-stream barrier without calling the legacy
materializer. The exact fresh two-source/two-`Iterate`/shared-`Collect.item` and exact fresh
three-source/three-`Iterate`/ shared-`Collect.item` fan-in shapes also use the planner for independent stream
expansions, closure gating, and deterministic source-ordered hydration. The exact fresh two-branch body-mediated shape
`source_a -> Iterate_a -> body_a -> Collect.item` plus `source_b -> Iterate_b -> body_b -> Collect.item` uses the same
planner: source IDs are ordered lexically, each branch preserves its item order, and `Collect` runs only after both
streams close, including explicit empty closes. This is a private bounded planner case: exactly seven nodes and six
ordinary edges—two inputless ordinary sources, two `Iterate` nodes, two ordinary bodies, and one `Collect`; it has no
`Collect.collection` input or downstream/extra topology. It is not a change to the public workflow contract. The
planner also admits one bounded input-driven extension: one inputless ordinary collection producer feeds one ordinary
collection-preparation node, which feeds the same `Iterate` -> ordinary body -> `Collect` path, with ordinary
downstream consumers allowed. It owns the preparation input edge, ordered stream effects, empty closure, checkpoint
rehydration, failure, rollback, and fresh retry behavior for that exact shape. It does not admit `If`, `For`,
`ForReturn`, saved-workflow, nested-`Iterate` chain, fan-in, or mixed control-flow nodes outside the bounded per-item
topology above; those remain compatibility-owned. The exact fresh seven-node nested shape
`outer For -> ordinary collection preparation -> Iterate -> ordinary body -> Collect -> linked outer ForReturn`, with
one ordinary `outer For.output_collection` consumer, is also generic-routed. The nested planner owns per-outer-frame
stream expansion, parent-frame stream identity, empty closure, `Collect` readiness/hydration, checkpoint rehydration,
and failure parity for this shape without calling the materializer's nested-copy helper. Five-level-or-deeper or
sibling nested `For` outside the exact two-sibling `CollectionConcat` fan-in shape, deeper or multiple nested
iterators, other input-driven outer collections, other mixed control flow, and legacy snapshots retain compatibility
materializer ownership. The exact eight-node producer-driven extension—one inputless ordinary collection producer
feeding `outer For.collection`, with the same nested body and one ordinary outer-output consumer—is also
generic-routed. It owns outer input hydration, per-outer-frame stream expansion, checkpoint rehydration, and failure
parity; input-driven inner preparation, deeper or multiple nesting, other mixed control flow, and legacy snapshots
remain compatibility-owned. The exact nine-node producer-driven extension adds the same inputless, statically non-empty
`CollectionConcatInvocation` outer producer to the three-level serial nested-`For` chain. It is generic-routed with the
same trace, output, checkpoint, and failure guarantees; other producer classes and deeper, multiple, sibling, mixed,
and legacy shapes remain compatibility-owned. The exact serial two-level
extension—`For.item -> preparation1 -> Iterate1 -> preparation2 -> Iterate2 -> ordinary body -> Collect -> linked ForReturn`,
plus one ordinary outer-output consumer—is also generic-routed. Its planner and compatibility adapter preserve ordered
nested frame paths, close empty streams, resume from checkpoints, and stop downstream completion on failure/retry.
Fan-in, sibling iterators, input-driven inner preparation, other mixed control flow, deeper nesting, and legacy
snapshots remain compatibility-owned. The exact five-node/four-edge nested-`Iterate` chain—an inputless ordinary source
feeding `outer Iterate`, whose item feeds one ordinary preparation node, which feeds `inner Iterate`, whose item feeds
one ordinary body—is also generic-routed. Its private planner owns ordered outer/inner frame expansion, empty outer and
inner closure, source completion, checkpoint rehydration, and failure parity without calling the compatibility
materializer. `For`, `Collect`, fan-in, sibling iterators, other mixed control flow, and legacy snapshots remain
compatibility-owned; the bounded deeper serial extensions are described next. The exact seven-node/six-edge serial
nested-`Iterate` chain extends this bounded planner by adding one ordinary preparation/`Iterate` pair between the outer
iterator and body. It owns three-component frame paths, ordered expansion, empty outer and intermediate streams, source
completion, checkpoint rehydration, and failure parity. The exact nine-node/eight-edge serial chain extends the same
planner to four `Iterate` nodes and four-component frame paths, with the same empty-stream, checkpoint,
source-completion, and failure parity guarantees. The exact eleven-node/ten-edge serial chain extends the same planner
to five `Iterate` nodes and five-component frame paths, with the same guarantees. The exact thirteen-node/twelve-edge
serial chain extends the same planner to six `Iterate` nodes and six-component frame paths, with the same empty-stream,
checkpoint, source-completion, and failure-parity guarantees. The exact fifteen-node/fourteen-edge serial chain extends
the same planner to seven `Iterate` nodes and seven-component frame paths, with the same empty-stream, checkpoint,
source-completion, and failure-parity guarantees. Three or more body-mediated branches, fan-in with four or more direct
branches, broader nested iterators, and mixed control flow outside the bounded per-item topology retain
`_ExecutionMaterializer` ownership for copy expansion, grouping, and empty-source handling.
`_GenericGraphSchedulerAdapter` may still own readiness for these unsupported fresh `Iterate`/`Collect` graphs; the
compatibility materializer owns expansion and materialization. Scheduler/materializer ownership is topology-dependent:
pure legacy `Iterate`/`Collect` uses generic scheduler readiness with compatibility materializer expansion; forced
compatibility and legacy control-flow paths retain `_ExecutionScheduler` where applicable.

The exact seventeen-node/sixteen-edge serial chain extends the same planner to eight `Iterate` nodes and
eight-component frame paths, with the same empty-stream, checkpoint, source-completion, and failure-parity guarantees.
Nine-level or deeper chains and all other expanded, fan-in, sibling shapes outside the exact two-sibling
`For`/`ForReturn` `CollectionConcat` contract, mixed, five-level-or-deeper `For`, `Collect`, and legacy shapes remain
compatibility-owned. A fresh graph with exactly one flat `For`/`ForReturn` pair and ordinary body nodes also uses the
generic adapter for a literal collection, a supported non-empty input-driven collection, or the exact
four-node/four-edge empty input-driven shape with one inputless producer and no downstream node. It projects readiness
and invokes the graph-state continuation boundary, which selects the next iteration or finalizes the aggregate without
exposing a successor node ID to the generic scheduler. The admitted For planner owns For-specific preparation,
continuation, input hydration, iteration paths, and body expansion; it shares only low-level execution-node
construction mechanics with the compatibility materializer. Other empty/input-driven shapes, five-level-or-deeper or
sibling loops outside the exact two-sibling nested-`For` `CollectionConcat` fan-in shape, mixed control flow, and
saved-workflow calls remain on the compatibility scheduler. One narrow fresh two-level nested-`For` shape (one
non-empty literal outer collection, one inner `For` sourced from `outer.item`, and no continuation nodes) now uses the
generic adapter; deeper-than-four-level, multiple-child shapes outside the exact two-sibling `CollectionConcat` fan-in
contract, other empty/input-driven, mixed, and legacy-loaded shapes remain compatibility-owned. The exact fresh
three-level serial nested-`For` shape (one literal outer collection or one statically non-empty
`CollectionConcatInvocation` producer, one direct child at each level, no continuation nodes) also uses the generic
adapter. The exact fresh four-level serial nested-`For` shape (one literal outer collection, one direct child at each
level, no continuation nodes) is also generic-routed. Its deepest-empty behavior matches compatibility at the trace,
output, and completion boundaries; the generic projection may retain synthetic empty-frame records.
Five-level-or-deeper and sibling nested-`For` shapes outside the exact two-sibling `CollectionConcat` fan-in contract
remain compatibility-owned. The runtime exposes an execution-engine seam: frame-scoped gates, ordered streams,
continuations, and authorized child-dependency records are stored in `invokeai.app.services.shared.execution_engine`;
legacy graph and queue behavior is retained behind adapters while those records provide the generic runtime projection
for admitted shapes.

The graph runtime is backend-only. It does not add author-time fields or frontend/backend external fields; the existing
frontend/backend interface remains stable.

## 2) Major Data Types

### EdgeConnection

- Fields: `node_id: str`, `field: str`.
- Hashable; printed as `node.field` for readable diagnostics.

### Edge

- Fields: `source: EdgeConnection`, `destination: EdgeConnection`.
- One directed connection from a specific output port to a specific input port.

### AnyInvocation / AnyInvocationOutput

- Pydantic wrappers that carry concrete invocation models and outputs.
- No registry logic in this file; they are permissive containers for heterogeneous nodes.

### IterateInvocation / CollectInvocation

- Control nodes used by validation and execution:

  - **IterateInvocation**: input `collection`, outputs include `item` (and index/total).
  - **CollectInvocation**: many `item` inputs aggregated to one `collection` output.

### Internal execution-engine records

- `ExecutionFrame`, `ActivationGate`, `StreamBuffer`, and `ContinuationRecord` carry typed runtime identity and
  lifecycle state without changing author-time graph models. Their private registries are rebuilt from persisted graph
  results, tokens, and workflow-call state after rehydration.
- `ChildExecutionCapability`, `ChildExecutionRecord`, and `ChildDependencyRecord` validate authorized parent/child
  relationships, resource limits, ordered all-of aggregation, failure, cancellation, and idempotent completion.
- `ExecutionPlan` stores opaque execution-node IDs, class names, prerequisite IDs, frame values, activation-dependency
  records, and stable insertion order. `ExecutionScheduler` owns generic readiness, opaque readiness predicates,
  intentional skips, deterministic ordering, completion, and durable plan rehydration. It does not import invocation
  classes and cannot alter author-time graph or frontend contracts.
- `ExecutionEngineRuntime` owns these records for one graph state. It is private runtime machinery; it is not a new
  frontend node, input handle, or public workflow contract.

## 3) Graph (author-time model)

A container for declared nodes and edges. Does **not** perform iteration expansion.

### 3.1 Data

- `nodes: dict[str, AnyInvocation]` - key must equal `node.id`.
- `edges: list[Edge]` - zero or more.
- Utility: `_get_input_edges(node_id, field?)`, `_get_output_edges(node_id, field?)` These use cached per-node
  adjacency indexes rebuilt when the edge list changes.

### 3.2 Validation (`validate_self`)

Runs a sequence of checks:

1. **Node ID uniqueness** No duplicate IDs; map key equals `node.id`.

1. **Endpoint existence** Source and destination node IDs must exist.

1. **Port existence** Input ports must exist on the node class; output ports on the node's output model.

1. **DAG constraint** Build a *flat* `DiGraph` (no runtime expansion) and assert acyclicity.

1. **Type compatibility** `get_output_field_type` vs `get_input_field_type` and `are_connection_types_compatible`.

   If output types:

   - For `IfInvocation.value`, resolve both branch inputs to their source ports, following nested `If` outputs and
     deduplicating shared sources. Check every resolved source against the destination using the existing compatibility
     hooks; the current condition value does not narrow the possible types.
   - Collector item/collection checks and iterator collection checks also use the resolved branch sources. This allows
     matching string branches to feed a string collector without an intermediate string node.
   - Iterators over collectors validate their item consumers against the same inferred collector root type. Mixed
     `int` and `float` items therefore require consumers compatible with `float`.
   - If a branch is unresolved, retain the declared output compatibility behavior (`Any` for collector type
     inference). The graph's separate DAG check rejects cycles before projection.
   - This widens compatibility: a direct collector with both `int` and `float` items, previously rejected, now resolves
     to `float`. Its downstream consumers must accept floats; unrelated mixed item types remain invalid.
   - Ordinary source ports use the direct compatibility path with already resolved nodes, without If traversal or
     projection allocations.

   Saved-workflow special case:

   - `call_saved_workflow` currently accepts dynamic destination handles of the form
     `saved_workflow_input::{childNodeId}::{childFieldName}` as part of its dynamic call-boundary contract.
   - Those handles are allowed through graph validation even though they are not static Python model fields on the
     invocation class.
   - Runtime validates them against the selected child workflow's exposed callable interface before applying values to
     the child graph.
   - The editor preserves dynamic caller values only while the exposed field type remains compatible; type drift at the
     same child node/field path resets to the selected workflow's current initial value.
   - Saved-workflow picker search is server-backed so large workflow libraries do not require scrolling every page
     before selecting a workflow by name.

1. **Iterator / collector structure** Enforce special rules:

   - Iterator's input must be `collection`; its outgoing edges use `item`.
   - Collector accepts many `item` inputs; outputs a single `collection`.
   - Edge fan-in to a non-collector input is rejected.

### 3.3 Edge admission (`_validate_edge`)

Checks a single prospective edge before insertion:

- Endpoints/ports exist.
- Destination port is not already occupied unless it's a collector `item`.
- Adding the edge to the flat DAG must keep it acyclic.
- Iterator/collector constraints re-checked when the edge creates relevant patterns.

After inserting an edge into `If.true_input` or `If.false_input`, `add_edge` also revalidates affected downstream
connections. It follows resolved If outputs and collector chains, checking each affected If, collector, or iterator
once. Ordinary nodes end this dependency walk; unrelated unfinished nodes are not revalidated. If a check fails, the
new edge is removed and adjacency indexes are restored before the validation error is raised.

### 3.4 Topology utilities

- `nx_graph()` - DiGraph of declared nodes and edges.
- `nx_graph_flat()` - "flattened" DAG (still author-time; no runtime copies). Used in validation and in `_prepare()`
  during execution planning.

### 3.5 Mutation helpers

- `add_node`, `update_node` (preserve edges, rewrite endpoints if id changes), `delete_node`.
- `add_edge`, `delete_edge` (with validation).

## 4) GraphExecutionState (runtime)

Holds the state for a single run. Keeps the source graph intact and materializes a separate execution graph.
`GraphExecutionState` is still the public runtime entry point, but most execution behavior is now delegated to a small
set of internal helper classes. For ordinary static DAGs and legacy-shaped `If` graphs, readiness and completion are
projected through the generic `ExecutionPlan`/`ExecutionScheduler` adapter. Direct `Iterate`/`Collect` graphs without
other control-flow nodes also use that adapter: its readiness predicate waits for canonical streams to close and its
completion mirrors Iterate outputs into the stream ledger. A fresh graph with one static flat `For`/`ForReturn` pair,
including a supported non-empty input-driven collection producer or the exact bounded empty input-driven producer
shape, also uses the adapter for readiness and continuation transitions; graph state owns the invocation-specific
continuation boundary while the generic scheduler remains opaque. Other empty or unsupported input-driven shapes,
five-level-or-deeper or sibling loops outside the exact two-sibling nested-`For` `CollectionConcat` fan-in contract,
saved-workflow calls, and unsupported mixed control-flow shapes outside the bounded per-item topology continue to use
the legacy compatibility scheduler until their differential coverage is complete. The narrow canonical two-level
nested-`For` shape, the canonical outer-`For`/bounded-`Iterate`/`Collect` shape, the exact five-node nested-`Iterate`
chain, and the bounded three-, four-, five-, and six-, seven-, and eight-level serial nested-`Iterate` chains are
generic-routed. The nested `For` shape has one ordinary preparation node from `For.item` into `Iterate.collection`, one
ordinary body node into `Collect.item`, the `Collect.collection` output into the linked `ForReturn`, and one final
outer-output consumer. Each outer iteration, including an empty inner collection, remains isolated by its frame path.
`Iterate` also records non-empty item streams through the generic effect ledger; for this exact seven-node shape, the
private nested planner owns per-outer-frame copy expansion, stream identity, empty closure, `Collect`
readiness/hydration, checkpoint rehydration, and failure parity. The exact producer-driven outer extension described
above is also generic-routed. The exact fresh three-level serial nested-`For` shape, including its statically non-empty
producer-driven outer variant, and the exact fresh four-level serial nested-`For` shape are also generic-routed;
five-level-or-deeper or sibling nested `For` outside the exact two-sibling `CollectionConcat` fan-in contract,
nine-level or deeper nested iterators, other input-driven outer collections, mixed control flow, and legacy snapshots
retain compatibility materializer ownership. The exact nested-`Iterate` chains above do not admit `For`, `Collect`,
fan-in, sibling, or deeper-than-eight-level iterator topology. The materializer remains authoritative for fallback
expansion, iteration paths, collector grouping, and empty-source compatibility handling. The exact fresh four-node body
shape, exact two- or three-stream fan-in shapes, and exact two-branch body-mediated fan-in use the private planner for
those responsibilities; three or more body-mediated branches and four or more direct streams remain
compatibility-owned. Direct `Collect.item` consumers now use the closed stream ledger when available; the full
Iterate/Collect compatibility matrix covers empty, nested, fan-in, partial rehydration, failure, cancellation, and
retry behavior. The materializer and queue adapters remain the compatibility owners for other shapes. The
SQLite/session boundary also preserves this exact nested shape across cancellation, simulated interrupted-process
startup, and retry: startup cancels stale in-progress rows without changing the persisted partial snapshot, while retry
creates fresh execution/frame/stream identities and completes the same outer collection. Broader nested shapes remain
compatibility-owned.

The source graph is treated as stable during normal execution, but the runtime object still exposes guarded graph
mutation helpers. Those helpers reject changes once the affected nodes have already been prepared or executed.

### 4.1 Data

- `graph: Graph` - source graph for the run; treated as stable during normal execution.
- `execution_graph: Graph` - materialized runtime nodes/edges. This is mutable runtime state, not an immutable audit
  log. Fresh `If` admission adds only the condition and selected-branch edges; legacy loaded execution graphs may still
  contain append-only discarded nodes without deleted input edges. Retry paths rebuild from `graph`, not from a
  previously persisted `execution_graph`.
- `executed: set[str]`, `executed_history: list[str]`.
- `results: dict[str, AnyInvocationOutput]`, `errors: dict[str, str]`.
- `prepared_source_mapping: dict[str, str]` - exec id -> source id.
- `source_prepared_mapping: dict[str, set[str]]` - source id -> exec ids.
- `indegree: dict[str, int]` - unmet inputs per exec node.
- Workflow-call runtime state:
  - `workflow_call_stack` - active parent call frames.
  - `workflow_call_history` - completed or failed workflow-call relationships observed by this execution state.
  - `workflow_call_parent` - parent workflow-call relationship metadata when this execution state is a child session.
  - `waiting_workflow_call` - the call frame currently suspending this execution state, if any.
  - `waiting_workflow_call_execution` - the active parent/child workflow-call relationship record for the waiting call.
  - `waiting_workflow_call_child_session` - attached child execution state for the waiting workflow call, if any.
  - `max_workflow_call_depth` - runtime guardrail for nested or recursive workflow calls.
- Prepared exec metadata caches:
  - source node id
  - iteration path
  - runtime state such as pending, ready, executed, or skipped
- `execution_refs: dict[str, ExecutionReference]` - in-memory stable references for prepared execution nodes, including
  their source node and execution frame; version-2 snapshots rebuild these references instead of storing them.
- `execution_tokens: dict[str, ExecutionToken]` - in-memory output tokens produced by applied execution results;
  version-2 snapshots retain only frame-scoped activation tokens.
- `execution_effects: dict[str, list[Any]]` - JSON-safe effects accepted for each execution reference.
- **Ready queues grouped by class** (private projection): `_ready_queues: dict[class_name, deque[str]]` and
  `_active_class: Optional[str]`. Ordinary static DAGs and legacy-shaped `If` graphs derive readiness from the generic
  scheduler; the `If` adapter stores frame-local activation dependencies whose private gate state plus persisted token
  checks control generic branch readiness. Supported static flat `For` graphs, the narrow canonical two-level nested
  `For` shape, the exact fresh three-level and four-level serial nested-`For` shapes (including the exact three-level
  producer-driven outer variant), the exact producer-driven nested extension, the exact bounded empty input-driven flat
  `For`, the bounded two- through eight-level serial nested-`Iterate` chains, and the bounded per-item
  `Iterate`/`If`/`Collect` topology use the generic adapter for readiness and continuation projection; other
  unsupported input-driven, nine-level or deeper, mixed graphs outside that topology retain the legacy scheduler, while
  saved-workflow calls use the generic lifecycle effect/dependency projection with the dedicated durable queue adapter.
  Optional `ready_order: list[str]` prioritizes classes. Queues are rebuilt from persisted execution state when a
  session is deserialized.

### 4.2 Core methods

- `next()` Returns the next ready exec node. If none are ready, it asks the admitted planner or compatibility
  materializer to expand more source nodes and then retries. If the execution state is paused on a workflow call
  boundary, it returns `None` without scheduling more work. Before returning a node, the runtime helper deep-copies
  inbound values into the node fields.
- `complete(node_id, output)` is the compatibility completion boundary. For a first JSON-safe completion it delegates
  validation and the atomic scheduler/ledger transition to `apply()`, including output tokens and synthetic
  continuation effects for direct `For`/`ForReturn` callers. An already-applied node retains the historical idempotent
  result-replacement behavior; in-memory values that cannot be represented in the JSON effect ledger use the
  scheduler-only compatibility path and are not persistable.
- `apply(execution_ref, output, effects)` validates a result against its prepared node and frame, then applies the
  result through the existing scheduler while recording references, output/effect tokens, and JSON-safe effects. The
  transition and ledger update are atomic. Direct `complete()` callers therefore receive the same durable state for
  supported values without changing the invocation output contract.

#### Current execution-effects seam

`InvocationContext` exposes a restricted `execution` facade and the underlying `execution_effects` recorder. The
existing invocation output contract remains unchanged. The session runner calls `invoke_internal_with_effects()` and
passes its `InvocationRunResult` to `GraphExecutionState.apply()`.

The recorder dispatches `emit` and `close_stream` by default. A runner may opt an invocation into lifecycle recording
with an engine-issued `ChildExecutionCapability`; this records validated `spawn_execution`, `await`, and `fail` effects
and returns a capability-bound child handle. Calls without that capability remain rejected. Mutation effects and queue
row creation remain owned by graph/queue adapters, so an invocation cannot mutate either directly. A cache hit never
suppresses effects: effect-enabled invocations bypass the ordinary output cache for that dispatch.

`IfInvocation` is the first control-flow invocation to declare an activation-effect contract. It emits one frame-scoped
activation token for the selected `true_input` or `false_input` port; `GraphExecutionState` validates that port against
the producing invocation's declared activation fields and persists it without creating a data stream. On rehydration,
every activation token is bound to a currently prepared owner and its derived execution reference; its declared port,
value, canonical token id, mapping key, and known frame fields must match. Unknown extra frame metadata remains
forward-compatible. For a fresh generic `If` graph with one ordinary-node `If`, the exact one-level nested shape
described above, the exact bounded three-`If` inner/middle/outer chain (including its single middle-value leaf
consumer), the exact four-`If` nested chain, or exactly two, exactly three, or exactly four independent sibling `If`s,
`GraphExecutionState` compiles opaque, frame-local activation-dependency records privately on each branch-local plan
node. Five-or-more nesting, five-or-more sibling `If`s, other fan-out, mixed graphs outside the bounded per-item
`Iterate`/`If`/`Collect` topology, loop-containing, and saved-workflow shapes use the compatibility scheduler and
`_IfActivationController` for the same records; legacy snapshots retain the generic compatibility projection and legacy
skipped-state metadata. The controller remains the fallback runtime dependency owner, and the schedulers no longer call
a legacy compiler. Fresh materialization prepares the condition boundary, resolves the activation token, and attaches
only the selected branch input; rejected branch sources remain unprepared. `_GenericGraphSchedulerAdapter` consumes
those records through the opaque plan: its readiness callback accepts a node only when the required private
`ActivationGate` runtime state is resolved and a matching persisted activation token is present for its frame, while
rejected dependencies cause scheduler discard. Both scheduler adapters consume these dependency records; neither prunes
execution edges or calls a type-specific branch scheduler. The compatibility path still projects discarded prepared
nodes into legacy skipped metadata for rehydration and completion accounting when loading old execution graphs; fresh
runs do not create those nodes. `apply()` validates and persists the invocation-emitted effect afterward, replacing the
compatibility token by stable identity. Activation effects are excluded from data-stream handling. `IterateInvocation`
is the first stream-producing control-flow invocation on this seam: each non-empty prepared copy emits one ordered
`item` effect with its iteration index, and the final copy emits one `close_stream` effect. Direct Iterate/Collect-only
graphs now run through the generic scheduler adapter; its completion maps these results/effects to the existing
frame-scoped iteration-stream identity, so legacy output mirroring is idempotent. The exact fresh four-node
source/Iterate/body/Collect shape, its ordinary downstream consumers, the exact two- or three-stream fan-in shapes, and
the exact two-branch body-mediated fan-in use the private planner described above; broader body-mediated fan-in, four
or more streams, and other fallback shapes still let the materializer create prepared copies, derive iteration paths,
group collector inputs, and record the explicit close for an empty source. For a direct `Iterate.item` edge, the
scheduler defers `CollectInvocation` while its canonical stream is open, and runtime hydration consumes the closed
stream in sequence order. If no stream exists, hydration retains the legacy materialized-result fallback needed by
older snapshots. This is not yet token-authoritative downstream topology beyond the supported direct-planner shape or
full `Collect` support: fallback shapes still let the materializer own copy expansion, iteration paths, collector
grouping, collection-input hydration, and empty-source closure; the exact fresh four-node source/Iterate/body/Collect
shape and any number of ordinary downstream-consumer extensions use the private planner instead. No author-time
activation ports or literal successor IDs are introduced.

For a fresh graph with one static flat `For`/`ForReturn` pair, including the exact empty literal collection case, the
same adapter now projects ordinary readiness and calls the graph-state generic continuation boundary after each
`ForReturn` when iterations exist. That boundary carries returned state, honors `continue_condition`, creates the next
prepared iteration when needed, and finalizes `output_collection` and `final_state`. For an empty literal collection,
the generic path completes through the existing synthetic terminal `For` result without running the body or
`ForReturn`. The generic scheduler receives only opaque node IDs and dependencies; it never receives a literal next
node ID. The compatibility continuation bridge remains available only for unsupported loop shapes and explicitly
legacy-loaded snapshots. The exact four-node/four-edge empty input-driven shape with one inputless static-list producer
is also generic-routed. Other input-driven outer collections, five-level-or-deeper or sibling nested loops outside the
exact two-sibling `CollectionConcat` fan-in contract, or unsupported mixed loop shapes remain compatibility-owned; the
exact fresh two-sibling `For`/`ForReturn` `CollectionConcat` fan-in shape, the narrow canonical two-level, and exact
fresh three-level and four-level serial nested-`For` shapes (including the exact three-level producer-driven outer
variant), the canonical bounded internal `Iterate`/`Collect` shape, the exact producer-driven outer nested extension,
and the bounded two- through six-, seven-, and eight-level serial nested-`Iterate` chains are generic-routed. On
rehydration, the generic adapter restores the active class-drain boundary so an in-flight nested stream resumes in the
same frame/sequence order.

For scheduling is linear in the collection size. The continuation transfers ownership of the remaining collection to
the next prepared `For` node instead of deep-copying it at every iteration. Source completion uses a derived, lazy
per-source count of pending prepared executions, and continuation validation uses the existing prepared-`For` index
rather than rescanning every prior iteration. The count is runtime-only and is rebuilt after snapshot rehydration; it
does not change the persisted execution-state or invocation contracts. The focused For performance,
collection-ownership, count-invariant, and generic/compatibility differential tests cover this boundary. The direct
Iterate planner has separate readiness projection costs and is not part of this For-specific optimization.

`ExecutionFrame` identifies the owning state, loop iteration path, and workflow-call depth. `ExecutionReference`
identifies one prepared execution node and its frame. `ExecutionToken` records an output port, value, frame, token
kind, and optional sequence. `loop_linkage` remains association metadata and never becomes a data token. This ledger is
currently additive. Ordinary static-DAG and legacy-shaped `If` readiness comes from the generic scheduler through a
compatibility projection; the activation token is authoritative for generic `If` branch readiness. Supported static
flat `For` readiness and continuation transitions also use the generic adapter, while materialization and type-specific
control paths remain authoritative for unsupported loop shapes and workflow-call graphs.

The generic scheduler is an in-memory graph-state component only. It does not create, update, retry, cancel, delete, or
recover `SessionQueueItem` rows and does not own queue statuses or events. A graph containing a saved-workflow call
therefore stays on the legacy scheduler and the dedicated workflow-call queue adapters.

Control-flow nodes remain concrete invocation subclasses that declare frame-scoped data, activation, stream-closure,
child, and terminal effects. The generic scheduler selects nodes from required effects for the current frame; it never
receives a literal successor-node ID. Current implementation is narrower: `IfInvocation` and non-empty
`IterateInvocation` use the effect recorder for activation and stream effects; the supported static flat
`For`/`ForReturn` shape, including the non-empty input-driven producer case, uses it for frame-scoped continuation
effects and generic readiness/continuation projection. Direct `Iterate`/`Collect`-only graphs use the generic scheduler
adapter for readiness and completion. The exact fresh four-node source/Iterate/body/Collect shape, its ordinary
downstream-consumer extensions, the exact two- or three-stream fan-in shapes, and the exact two-branch body-mediated
fan-in use the private planner for expansion and empty-stream closure; broader body-mediated fan-in, four or more
streams, workflow-call invocations, unsupported loop shapes, and mixed control-flow graphs remain on compatibility
paths. `ForInvocation` and `ForReturnInvocation` now declare one validated, frame-scoped `continuation` effect per
prepared invocation: `For` starts the `for` continuation with its iteration/state payload, and `ForReturn` completes it
with output/state/ continue-decision data. Session-built recorders bind the effect reference to the graph-state ID,
durable frame ID, iteration path, and workflow-call depth; graph-state validation rejects a stale or cross-frame
continuation before mutation. Graph state persists these effects under the invocation reference and excludes
`loop_linkage` from token data. Continuation payloads are normalized to JSON values before runtime or durable
comparison. Exact duplicate continuation effects in one batch are idempotently retained once; conflicting terminal
payloads, including type-distinct JSON values, are rejected transactionally. Before `complete()` mutates scheduler
state, `ForInvocationOutput` and `ForReturnInvocationOutput` continuation fields must match their prepared nodes; the
`For` item must match its prepared collection item, and continuation effects must independently match those prepared
values. Current versioned snapshots require an execution-effect ledger: each bucket must belong to an executed prepared
node, while unknown references, pending-node buckets, missing executed markers, missing required continuation effects,
and malformed ownership are rejected. Persisted `For` outputs are checked against prepared iteration data and, after
finalization, their authoritative returned collection and final state. Unversioned legacy snapshots retain the
compatibility loader. `For` start payloads must match the prepared index, collection total, and state before mutation.
This is the invocation/effect ownership seam: the generic adapter calls the graph-state generic continuation boundary,
which creates the next prepared iteration, aggregates outputs, and finalizes the supported flat loop. The compatibility
bridge remains for other unsupported input-driven, five-level-or-deeper/sibling nested outside the exact two-sibling
`CollectionConcat` fan-in contract, mixed-loop, and legacy-snapshot execution; those paths retain their existing
materialization and linkage ownership. The supported fresh static flat `For`/`ForReturn` shape, the exact bounded empty
input-driven flat shape, the narrow canonical two-level and exact fresh three-level and four-level serial nested-`For`
shapes (including the exact three-level producer-driven outer variant), and the canonical bounded internal
`Iterate`/`Collect` shape use the generic adapter. For `If`, generic readiness consumes opaque frame-local plan
dependencies and requires both matching private `ActivationGate` runtime state and a persisted activation token. Fresh
resolution attaches only the selected branch edge; the forced compatibility `_ExecutionScheduler` path preserves this
behavior for fresh states, while legacy skipped metadata remains only when loading old discarded projections. This is
not token-built successor topology. Loop and workflow-call adapters still own materialization for unsupported shapes
and durable queue lifecycle. The frontend boundary remains frozen: the graph runtime does not modify
`invokeai/frontend/...` or existing web/webv2 interactions.

The test-only differential harness at `tests/app/services/shared/test_execution_engine_differential.py` compares
generic and forced-compatibility scheduling for static DAGs, a constructed mixed/nested `If` graph, the supported
static flat `For`/`ForReturn` shape, the narrow canonical nested-`For` shape, and the canonical nested
`For`/`Iterate`/`Collect` shape. The latter covers empty inner collections and dump/load resume ordering. Its fixture
corpus covers fresh completion, true/false branch selection, nested branch isolation, partial checkpoints, versioned
rehydration, in-flight claim replay, activation-token persistence, injected failure, loop continuation, carried state,
early break, output `None`, cancellation/retry isolation, and continuation/effect integrity. The `If` comparison
includes strict source-level results, executed history, errors, terminal state, and normalized indegrees; compatibility
skip propagation must not leave stale downstream indegrees. Both scheduler adapters expose the same skip transition;
the legacy path releases downstream indegrees without trying to hydrate inputs from the skipped node. It does not claim
durable persistence of the generic scheduler's private claim set, Other unsupported input-driven, deeper/multiple,
mixed, or workflow-call loop shapes remain on their compatibility paths. Real queue/processor coverage in
`tests/app/services/session_processor/test_if_processor_sqlite.py` also exercises true and false `If` selection,
cancellation before and after resolution, retry from a canceled SQLite item, fresh execution identities, and
selected-only completion. Stale identity and legacy loop-frame rehydration are covered separately by the differential
and graph-state tests.

Workflow-call note:

- `GraphExecutionState` can represent a paused parent execution plus an attached child execution state, but it does not
  itself orchestrate child execution.
- In the current implementation, `DefaultSessionRunner.run_node()` invokes `CallSavedWorkflowInvocation` through the
  effect-aware path. The invocation records capability-bound `spawn_execution` plus `await` (or `fail`) intent;
  `GraphExecutionState.apply()` persists the pending batch, `WorkflowCallCoordinator` handles durable child setup, and
  `WorkflowCallQueueLifecycle` later resumes or fails the parent based on the generic child dependency and child queue
  outcome.
- Child `SessionQueueItem` rows created by the coordinator now carry explicit relationship metadata such as
  `workflow_call_id`, `parent_item_id`, `parent_session_id`, `root_item_id`, and `workflow_call_depth`, even though the
  higher-level scheduler semantics are still evolving.
- The `session_queue` schema now has matching columns for those relationship fields, and parent queue items can enter a
  `waiting` status while suspended on a child workflow execution.
- Queue lifecycle semantics for workflow-call chains are:
  - child success resumes the waiting parent
  - multiple child queue rows may complete under one waiting parent when the called workflow contains direct batch
    nodes; the parent resumes only after all expected child rows complete
  - child failure fails the waiting parent and can cascade upward through ancestors
  - failing child rows cancel their remaining workflow-call siblings before the parent is failed
  - cancelation is chain-aware across parents and children, including nested descendants of batched siblings
  - "all except current" queue actions preserve the active current item plus its workflow-call chain, while still
    canceling or deleting unrelated waiting chains
  - startup recovery cancels interrupted `in_progress` or `waiting` workflow-call chains, including pending descendants
  - deleting a workflow-call queue row currently deletes the whole parent/child chain rather than leaving orphaned rows
    behind
  - retry is root-oriented and should not be exposed directly on child queue rows in the UI
  - child queue-row creation is cleaned up on boundary-setup failure and child fan-out is bounded by remaining queue
    capacity
  - child workflows that mix supported batch nodes with unrelated generator nodes are rejected
- The generic `ChildDependencyRecord` now owns the internal child identity, ordered all-of aggregation, capability
  identity, resource limits, idempotent terminal transition, and parent terminal decision alongside this workflow-call
  lifecycle. Existing queue fields, statuses, cancellation, retry, and event behavior remain authoritative; queue-row
  creation and recovery are not silently delegated to an in-memory record.

### 4.3 Runtime helper classes

`GraphExecutionState` now delegates most runtime behavior to internal helpers:

- `_PreparedExecRegistry` Owns the relationship between source graph nodes and prepared execution graph nodes, plus
  cached metadata such as iteration path and runtime state.
- `_ExecutionMaterializer` is the compatibility planner facade. It expands source graph nodes into concrete execution
  graph nodes when an unsupported or legacy shape runs out of ready work. On those compatibility paths it owns iterator
  expansion, collector grouping, prepared-parent selection, and creation of execution-graph edges. When matching
  prepared parents for a downstream exec node, skipped prepared exec nodes are ignored and cannot be selected as live
  inputs. The class lives in `graph_materializer.py` and is re-exported by `graph.py`.
- `_GenericForPlanner` owns admitted fresh `For`/`ForReturn` preparation, continuation transitions, carried state,
  frame paths, and nested `For` completion. It uses `_ExecutionNodeBuilder` only for low-level copy, edge, input, and
  scheduler-registration mechanics; it does not construct `_ExecutionMaterializer`. Unsupported and legacy shapes
  retain the compatibility facade.
- Private `graph_iterate_planner.py` expands the supported direct and body-mediated Iterate/Collect shapes. Graph-state
  method wrappers preserve the admission, copy creation, edge attachment, and atomic preparation entry points. The
  planner continues to use the graph state's journal, mappings, caches, and scheduler.
- `_IfActivationController` Owns fallback runtime `If` admission and compiles opaque, frame-local activation dependency
  records for unsupported fresh shapes and legacy prepared nodes. Fresh ordinary-node single-`If` graphs, the bounded
  nested pair, the exact bounded three-`If` inner/middle/outer chain, the exact four-`If` nested chain, and exactly
  two, exactly three, or exactly four independent sibling `If`s compile those dependencies in graph state. The bounded
  three-`If` chain also admits one middle-`If` value fan-out to one ordinary leaf consumer; that leaf inherits only the
  outer branch dependency, not middle-`If` polarity. Other fan-out remains fallback. Fresh admission leaves rejected
  branch sources unprepared; legacy skipped-state projections remain loadable.
- Private `graph_if_dependencies.py` compiles fresh activation dependencies, and `graph_if_runtime.py` records
  dependencies and evaluates admission against gates and tokens. `GraphExecutionState` retains the method entry points,
  durable token/reference/effect storage, derived caches, transaction journal, and fallback controller selection.
- Private `graph_scheduler.py` contains `_ExecutionScheduler` and `_GenericGraphSchedulerAdapter`. `graph.py`
  deliberately re-exports both classes, preserving their existing import paths and monkeypatch seams while keeping
  scheduler implementation details out of the graph-state façade.
- `_GenericGraphSchedulerAdapter` Projects the generic `ExecutionPlan`/`ExecutionScheduler` into the existing state
  fields for ordinary static DAGs and legacy-shaped `If` graphs; the generic scheduler owns opaque readiness,
  intentional discards, indegree transitions, deterministic ordering, claimed work, and completion. The adapter
  registers activation dependencies from graph-state compilation or the fallback controller and checks private gate
  state plus persisted activation tokens. Fresh `If` branch nodes are admitted before materialization; legacy prepared
  nodes may still be discarded for compatibility. `If` scheduling does not call a type-specific branch scheduler,
  prune, or delete execution edges.
- `_ExecutionScheduler` Owns materialized-graph indegree transitions, class-grouped ready queues, downstream release,
  control-flow continuation scheduling, and the shared opaque activation-dependency projection for compatibility
  graphs.
- `_ExecutionRuntime` Owns iteration-path lookup, collect input ordering, and input hydration for prepared exec nodes.
  Its implementation lives in the private `graph_execution_runtime.py` module; `graph.py` re-exports the runtime class
  and its fan-in record types so existing imports and test seams remain stable.
- `ExecutionEngineRuntime` Owns the typed gate, stream, and continuation records used by compatibility adapters. The
  canonical stream for a prepared `IterateInvocation` is keyed by the source iterator and its parent iteration path;
  its item/close effects and legacy output mirroring update the same idempotent buffer. Direct `CollectInvocation.item`
  edges consume a closed canonical stream when available and otherwise use the legacy snapshot fallback.

`GraphExecutionState.model_post_init()` rehydrates private runtime helpers and caches after normal construction or a
JSON/model round trip. Rehydration reconstructs prepared exec metadata, cached iteration paths, private resolved `If`
gate state from condition results or persisted activation tokens, non-empty iteration streams from durable effects and
legacy Iterate results, explicit empty-source closes, For continuation identity, and ready queues from
`execution_graph`, `indegree`, `executed`, and `results`. Activation tokens persist; private `ActivationGate` runtime
state does not and is reconstructed from condition results or persisted activation tokens. Pending selected `If` inputs
remain excluded from ready-queue projection during rehydration, preserving ready-node order across checkpoints before
the pending `If` is admitted. Before token validation, missing legacy iteration-path metadata is rebuilt from the
prepared execution graph. Persisted activation identity is then validated fail-closed against prepared owners, derived
references, declared activation fields, canonical ids, and known frame fields; extra frame metadata is retained.
Version-2 `dump_execution_state()` omits derived execution references and ordinary output tokens, keeps activation
tokens, retains stream/continuation effects needed to resume, and drops terminal saved-workflow lifecycle effects and
completed child-dependency records. Active workflow-call child state remains attached for recovery; completed calls use
their workflow-call history instead. Ordinary model serialization and public schemas omit these four internal ledgers;
private helper objects are not serialized. Queue snapshots carry an additive execution-state version marker and use a
version-aware loader; legacy unmarked snapshots are treated as version 0, while unreadable snapshots are quarantined by
queue service.

### 4.4 Compatibility preparation (`_prepare()`)

The steps below describe compatibility materialization for unsupported shapes and legacy snapshots. Supported fresh
`Iterate`/`Collect` shapes use the bounded planners described above and do not perform this ancestry or collector
materialization.

- Build a flat DAG from the **source** graph.

- Choose the **next source node** in topological order that:

  1. has not been prepared,
  1. if it is an iterator, *its inputs are already executed*,
  1. it has *no unexecuted iterator ancestors*.

- If the node is a **CollectInvocation**: group prepared parent exec nodes by iteration path and create one collector
  exec node per group. A collector collapses the immediate iterator that feeds its `item` input, but preserves
  enclosing iterator paths. This lets a shape such as
  `outer_iter -> inner_collection -> inner_iter -> collect -> consumer` produce one collected result per outer
  iteration instead of mixing all inner items into one global collection. Incoming `collection` inputs are treated as
  ancestor groups and are copied into each matching descendant item group.

- Otherwise: compute all combinations of prepared iterator ancestors. For each combination, choose the prepared parent
  for each upstream by matching iterator ancestry, then create **one** exec node. If a node no longer has visible
  iterator ancestors because the source path crosses a collector, prepared parent iteration paths are still used to
  materialize one downstream exec node for each preserved collector path.

- For a fresh **IfInvocation**: prepare only its condition inputs first. After the condition resolves, graph-state
  activation dependencies admit selected branch contexts and attach only the selected branch input to the pending `If`
  execution. Unsupported fresh shapes and legacy snapshots use the compatibility controller. Rejected branch sources
  are not materialized. Pending `If` state is reconstructed from its durable execution edges, prepared mappings, and
  activation token after rehydration.

- For each new exec node:

  - Deep-copy the source node; assign a fresh ID (and `index` for iterators).
  - Cache the preserved iteration path when the materializer has one, such as for grouped collectors.
  - Wire edges from chosen prepared parents.
  - Set `indegree = number of unmet inputs` (i.e., parents not yet executed). The generic scheduler mirrors this into
    its opaque plan for ordinary static DAGs, legacy-shaped `If` graphs, and direct `Iterate`/`Collect`-only graphs.
  - Try to resolve any `If`-specific scheduling state.
  - If the node is ready and not deferred by an unresolved `If`, enqueue it into its class queue.

### 4.5 Readiness and class ordering

- `_enqueue_if_ready(nid)` applies generic readiness: `indegree == 0`, not executed or claimed, and, for an `If`
  branch-local node, both matching private `ActivationGate` runtime state and a persisted activation token are present
  for its frame. Fresh admission prevents rejected branch nodes from reaching this queue. For direct `Collect` nodes,
  the adapter also requires available Iterate streams to be closed. Compatibility uses the same opaque activation
  dependencies; legacy skipped metadata remains for old snapshots.
- `_get_next_node()` uses the generic scheduler for ordinary static DAGs, legacy-shaped `If` graphs, and the admitted
  fresh control-flow shapes described above, projecting deterministic class/frame order into the compatibility queues.
  Fallback control-flow graphs, including saved-workflow calls, use `_active_class` and the legacy class queues. No
  batch-size or fairness cap is currently implemented.

#### 4.5.1 Indegree (what it is and how it's used)

**Indegree** is the number of incoming edges to a node in the execution graph that are still unmet. In this engine:

- For every materialized exec node, `indegree[node]` equals the count of its prerequisite parents that have **not**
  finished yet.
- A node is eligible for enqueue when `indegree[node] == 0`, it has not executed, and it is not deferred by an
  unresolved `If`.
- When a node completes, the active scheduler decrements `indegree[child]` for each outgoing edge. Any child that
  reaches 0 is enqueued. The generic plan preserves repeated edges as repeated prerequisites, matching execution-graph
  indegree semantics.

Example: edges `A->C`, `B->C`, `C->D`. Start: `A:0, B:0, C:2, D:1`. Run `A` -> `C:1`. Run `B` -> `C:0` -> enqueue `C`.
Run `C` -> `D:0` -> enqueue `D`. Run `D` -> done.

### 4.6 Input hydration (`_prepare_inputs()`)

- For **CollectInvocation**: merge incoming `collection` values first, then gather `item` inputs. A direct
  `Iterate.item` edge uses its closed canonical stream ledger, preserving stream sequence order; an open stream is not
  hydrated, and a missing stream falls back to the materialized source result for legacy snapshots. Materialized inputs
  are still grouped by iteration path, so hydration only sees inputs belonging to that collector exec node.
- For **IfInvocation**: hydrate only `condition` and the selected branch input. As a defensive guard against
  inconsistent runtime or deserialized session state, the runtime raises if the selected input edge points at an exec
  node with no stored runtime output. In normal scheduling this path should be unreachable.
- For all others: deep-copy each incoming edge's value into the destination field. This prevents cross-node mutation
  through shared references.

### 4.7 Lazy `If` semantics

`IfInvocation` now acts as a lazy branch boundary rather than a simple value multiplexer.

- The `condition` input must resolve first.
- Nodes that are exclusive to the true or false branch remain unmaterialized until their branch is admitted, even when
  their indegree would otherwise be zero.
- Once the condition resolves, graph-state activation logic records the selected activation token and attaches only the
  selected branch input to the pending `If` execution. Branch-exclusive ancestors of the unselected branch are never
  prepared, executed, or added to fresh execution history. Unsupported fresh shapes and legacy snapshots use the
  compatibility controller. Both scheduler adapters use append-only execution edges; no type-specific branch scheduler
  performs pruning or skip propagation.
- Legacy snapshots that already contain skipped prepared nodes retain their compatibility metadata and scheduler
  projection. Fresh execution does not create that projection.
- The SQLite queue/processor path has evidence for cancellation before and after `If` resolution and retry from each
  boundary for both condition polarities. A canceled attempt keeps its activation ledger and cannot resume; retry
  starts a fresh state and must emit a fresh matching activation token before selected-only completion.
- Legacy retired or skipped branch-local exec nodes may still be treated as executed for compatibility scheduling, but
  they do not create entries in `results`.
- Shared ancestors still execute if they are required by the selected branch or by any other live path in the graph.

This behavior is implemented in the runtime scheduler, not in the invocation body itself.

## 5) Traversal Summary

1. Author builds a valid **Graph**.

1. Create **GraphExecutionState** with that graph.

1. Loop:

   - `node = state.next()` -> may trigger `_prepare()` expansion.
   - Execute node externally -> `run_result`.
   - `state.apply(execution_ref, run_result)` -> updates indegrees, `If` state, ready queues, and the execution ledger.

1. Finish when `next()` returns `None` and the execution state is not paused waiting on a workflow call boundary.

When an invocation reuses cached media, `BaseInvocation.invoke_internal()` registers a temporary hold for the consuming
queue session before returning the cached output. The intermediates service checks that every referenced image/video
record still exists and registers the hold under the same database lock as cleanup. If cleanup already deleted a
record, the invocation recomputes its output. Holds remain effective while the session is pending, running or
waiting. A cache hit reuses an old row, so its media is then treated as recent for the usual grace window, counted from
when cleanup first sees the session ended.
A completed child remains protected while its root workflow is active: completion precedes the transaction that records
its returned media in the parent. Cleanup protects both its cached references and its produced media through that
handoff, including nested calls. Queue history pruning retains these children until their root ends. Other
terminal/deleted sessions' holds are released on the next cache hold or cleanup query and dropped after that grace.
Holds share the cache's process lifetime; after restart, persisted queue sessions supply their own media references and
the invocation cache is empty.

In normal execution, all runtime expansion occurs in `execution_graph` with traceability back to source nodes.

## 6) Invariants

- Source **Graph** remains a DAG and type-consistent.
- `execution_graph` remains a DAG.
- Nodes are enqueued only when `indegree == 0` and they are not deferred by an unresolved `If`; generic `If`
  branch-local readiness additionally requires matching private `ActivationGate` runtime state and a persisted
  activation token.
- `results` and `errors` are keyed by **exec node id**.
- Applied execution references are unique to one prepared node and frame; their output/effect records are JSON-safe.
- Output and `emit` effects produce frame-aware tokens. `close_stream` produces a `stream_end` token. Association
  fields such as `loop_linkage` are never stored as data tokens.
- A non-empty `IterateInvocation` emits one `item` effect per prepared copy, with a contiguous sequence beginning at
  zero, and closes its canonical source/parent-path stream on the final copy. Exact output mirroring is idempotent. The
  private planner closes empty streams for the exact fresh source/Iterate/body/Collect, exact two- or three-stream
  fan-in, and exact two-branch body-mediated fan-in shapes; broader body-mediated fan-in, four or more streams, and
  other fallback shapes retain materializer compatibility ownership.
- Collectors wait for available direct Iterate streams to close, then aggregate their ledger values in stream order and
  may also merge incoming `collection` inputs during runtime hydration. A missing ledger remains a legacy snapshot
  fallback; stale legacy Iterate result mirrors do not override durable effects. Collectors nested under iterators
  preserve enclosing iteration paths, so downstream consumers materialize per enclosing iteration instead of receiving
  a mixed collection from unrelated outer iterations.
- Branch-exclusive nodes behind an unselected fresh `If` branch remain unmaterialized. Legacy snapshots may retain
  skipped nodes for compatibility; they are not failed.

## 7) Extensibility

- **New node types**: implement as Pydantic models with typed fields and outputs. Register per your invocation system;
  this file accepts them as `AnyInvocation`.
- **Scheduling policy**: adjust `ready_order` to prioritize class queues. A batch-size or fairness cap is not currently
  implemented.
- **Dynamic behaviors**: effect-enabled invocations may record frame-scoped stream, activation, and authorized child
  lifecycle intent. `GraphExecutionState.apply()` remains the transactional graph boundary: it persists a pending
  saved-workflow lifecycle batch and generic child dependency, while the matching queue adapter owns durable row
  creation and queue transitions. Mutation effects remain rejected.
- **Workflow call boundaries**: `GraphExecutionState` can suspend a parent execution state on a workflow call, attach a
  child execution state, and later resume the parent without mutating the source graph.

Current limitation:

- Child workflow executions are represented as first-class queue items. Parent resume/failure remains handled by the
  dedicated workflow-call queue lifecycle component; `ChildDependencyRecord` is the generic identity and aggregation
  seam, not a replacement for durable queue operations.
- Called workflows currently require exactly one valid `workflow_return` node to be callable at all.
- A single `workflow_return_value.value` may connect directly to `workflow_return.values`; multiple named return
  members should be collected and then connected to `workflow_return.values`.
- Direct batch-special child workflows are now supported by expanding them into multiple child queue rows.
- Batch outputs may feed a named `workflow_return_value.value` directly. Parent resume aggregates named return maps as
  `values: dict[str, list[Any]]`, and all rows in one batch call must return the same key set.
- Generator-backed batch child workflows are now supported when the batch node is fed directly by a supported integer,
  float, string, or image generator.
- Connected batch child inputs produced by ordinary non-generator upstream nodes are still rejected before any child
  queue row is created.
- Workflow library API responses now include compatibility metadata so the frontend can disable unsupported callees
  before execution rather than failing only at runtime.
- Workflow library list compatibility uses structural generator-backed batch validation so list and picker rendering do
  not enumerate every image in board-backed generators; workflow detail and runtime execution still resolve real
  generator values.
- Batch-specific compatibility failures, including multiple connected inputs to one batch field, are reported as
  `unsupported_batch_input` rather than generic unsupported-node failures.
- The workflow library list also surfaces that metadata as an informational unsupported state; workflows remain
  viewable/editable even when they are not currently callable by `call_saved_workflow`.
- Single-user workflow CRUD socket events emit only to the admin room because every single-user socket already joins
  that room, avoiding duplicate delivery through both `user:system` and `admin`.

## 8) Error Model (selected)

- `DuplicateNodeIdError`, `NodeAlreadyInGraphError`
- `NodeNotFoundError`, `NodeFieldNotFoundError`
- `InvalidEdgeError`, `CyclicalGraphError`
- `NodeInputError` (raised when preparing inputs for execution)

Messages favor short, precise diagnostics (node id, field, and failing condition).

## 9) Rationale

- **Two-graph approach** isolates authoring from execution expansion and keeps validation simple.
- **Indegree + queues** gives O(1) readiness decisions with clear class-ordering semantics.
- **Iterator/collector separation** keeps fan-out/fan-in explicit and testable.
- **Deep-copy hydration** avoids incidental aliasing bugs between nodes.
