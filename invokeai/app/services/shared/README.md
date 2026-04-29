# InvokeAI Graph - Design Overview

High-level design for the graph module. Focuses on responsibilities, data flow, and how traversal works.

## 1) Purpose

Provide a typed, acyclic workflow model (**Graph**) plus a runtime scheduler (**GraphExecutionState**) that expands
iterator patterns, tracks readiness via indegree (the number of incoming edges to a node in the directed graph), and
executes nodes in class-grouped batches. In normal execution, runtime expansion happens in a separate execution graph
instead of mutating the source graph.

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

## 3) Graph (author-time model)

A container for declared nodes and edges. Does **not** perform iteration expansion.

### 3.1 Data

- `nodes: dict[str, AnyInvocation]` - key must equal `node.id`.
- `edges: list[Edge]` - zero or more.
- Utility: `_get_input_edges(node_id, field?)`, `_get_output_edges(node_id, field?)` These scan `self.edges` (no
  adjacency indices in the current code).

### 3.2 Validation (`validate_self`)

Runs a sequence of checks:

1. **Node ID uniqueness** No duplicate IDs; map key equals `node.id`.

1. **Endpoint existence** Source and destination node IDs must exist.

1. **Port existence** Input ports must exist on the node class; output ports on the node's output model.

1. **DAG constraint** Build a *flat* `DiGraph` (no runtime expansion) and assert acyclicity.

1. **Type compatibility** `get_output_field_type` vs `get_input_field_type` and `are_connection_types_compatible`.

   Special case:

   - `call_saved_workflow` currently accepts dynamic destination handles of the form
     `saved_workflow_input::{childNodeId}::{childFieldName}` as part of its temporary call-boundary contract.
   - Those handles are allowed through graph validation even though they are not static Python model fields on the
     invocation class.
  - Runtime later validates them against the selected child workflow's exposed callable interface before applying
    values to the child graph.
  - The editor preserves dynamic caller values only while the exposed field type remains compatible; type drift at the
    same child node/field path resets to the selected workflow's current initial value.
  - Saved-workflow picker search is server-backed so large workflow libraries do not require scrolling every page before
    selecting a workflow by name.

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
set of internal helper classes.

The source graph is treated as stable during normal execution, but the runtime object still exposes guarded graph
mutation helpers. Those helpers reject changes once the affected nodes have already been prepared or executed.

### 4.1 Data

- `graph: Graph` - source graph for the run; treated as stable during normal execution.
- `execution_graph: Graph` - materialized runtime nodes/edges. This is mutable runtime state, not an immutable audit
  log. Lazy `If` pruning may remove unselected input edges during execution, so persisted failed/completed session
  snapshots can contain a structurally pruned execution graph. Retry paths rebuild from `graph`, not from a previously
  persisted `execution_graph`.
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
- **Ready queues grouped by class** (private attrs): `_ready_queues: dict[class_name, deque[str]]`,
  `_active_class: Optional[str]`. Optional `ready_order: list[str]` to prioritize classes.

### 4.2 Core methods

- `next()` Returns the next ready exec node. If none are ready, it asks the materializer to expand more source nodes and
  then retries. If the execution state is paused on a workflow call boundary, it returns `None` without scheduling more
  work. Before returning a node, the runtime helper deep-copies inbound values into the node fields.
- `complete(node_id, output)` Records the result, marks the exec node executed, marks the source node executed once all
  of its prepared exec copies are done, then decrements downstream indegrees and enqueues newly ready nodes.

Workflow-call note:

- `GraphExecutionState` can represent a paused parent execution plus an attached child execution state, but it does not
  itself orchestrate child execution.
- In the current implementation, `DefaultSessionRunner.run_node()` establishes the workflow call boundary and attaches
  the child execution state, while `WorkflowCallCoordinator` handles call-specific setup and
  `WorkflowCallQueueLifecycle` later resumes or fails the parent based on that child queue row's outcome.
- Child `SessionQueueItem` rows created by the coordinator now carry explicit relationship metadata such as
  `workflow_call_id`, `parent_item_id`, `parent_session_id`, `root_item_id`, and `workflow_call_depth`, even though the
  higher-level scheduler semantics are still evolving.
- The `session_queue` schema now has matching columns for those relationship fields, and parent queue items can enter a
  `waiting` status while suspended on a child workflow execution.
- Queue lifecycle semantics are now partially defined for workflow-call chains:
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
  - child workflows that mix supported batch nodes with unrelated generator nodes are rejected for now
- This is still an intermediate architecture step and should eventually be replaced by a more general parent/child
  execution mechanism rather than workflow-call-specific queue lifecycle handling.

### 4.3 Runtime helper classes

`GraphExecutionState` now delegates most runtime behavior to internal helpers:

- `_PreparedExecRegistry` Owns the relationship between source graph nodes and prepared execution graph nodes, plus
  cached metadata such as iteration path and runtime state.
- `_ExecutionMaterializer` Expands source graph nodes into concrete execution graph nodes when the scheduler runs out of
  ready work. When matching prepared parents for a downstream exec node, skipped prepared exec nodes are ignored and
  cannot be selected as live inputs.
- `_ExecutionScheduler` Owns indegree transitions, ready queues, class batching, and downstream release on completion.
- `_ExecutionRuntime` Owns iteration-path lookup and input hydration for prepared exec nodes.
- `_IfBranchScheduler` Applies lazy `If` semantics by deferring branch-local work until the condition is known, then
  releasing the selected branch and skipping the unselected branch.

### 4.4 Preparation (`_prepare()`)

- Build a flat DAG from the **source** graph.

- Choose the **next source node** in topological order that:

  1. has not been prepared,
  1. if it is an iterator, *its inputs are already executed*,
  1. it has *no unexecuted iterator ancestors*.

- If the node is a **CollectInvocation**: collapse all prepared parents into one mapping and create **one** exec node.

- Otherwise: compute all combinations of prepared iterator ancestors. For each combination, choose the prepared parent
  for each upstream by matching iterator ancestry, then create **one** exec node.

- For each new exec node:

  - Deep-copy the source node; assign a fresh ID (and `index` for iterators).
  - Wire edges from chosen prepared parents.
  - Set `indegree = number of unmet inputs` (i.e., parents not yet executed).
  - Try to resolve any `If`-specific scheduling state.
  - If the node is ready and not deferred by an unresolved `If`, enqueue it into its class queue.

### 4.5 Readiness and batching

- `_enqueue_if_ready(nid)` enqueues by class name only when `indegree == 0`, the node has not already executed, and the
  node is not deferred by an unresolved `If`.
- `_get_next_node()` drains the `_active_class` queue FIFO; when empty, selects the next nonempty class queue (by
  `ready_order` if set, else alphabetical), and continues. Optional fairness knobs can limit batch size per class;
  default is drain fully.

#### 4.5.1 Indegree (what it is and how it's used)

**Indegree** is the number of incoming edges to a node in the execution graph that are still unmet. In this engine:

- For every materialized exec node, `indegree[node]` equals the count of its prerequisite parents that have **not**
  finished yet.
- A node is "ready" exactly when `indegree[node] == 0`; only then is it enqueued.
- When a node completes, the scheduler decrements `indegree[child]` for each outgoing edge. Any child that reaches 0 is
  enqueued.

Example: edges `A->C`, `B->C`, `C->D`. Start: `A:0, B:0, C:2, D:1`. Run `A` -> `C:1`. Run `B` -> `C:0` -> enqueue `C`.
Run `C` -> `D:0` -> enqueue `D`. Run `D` -> done.

### 4.6 Input hydration (`_prepare_inputs()`)

- For **CollectInvocation**: gather all incoming `item` values into `collection`, sorting inputs by iteration path so
  collected results are stable across expanded iterations. Incoming `collection` values are merged first, then incoming
  `item` values are appended.
- For **IfInvocation**: hydrate only `condition` and the selected branch input. As a defensive guard against
  inconsistent runtime or deserialized session state, the runtime raises if the selected input edge points at an exec
  node with no stored runtime output. In normal scheduling this path should be unreachable.
- For all others: deep-copy each incoming edge's value into the destination field. This prevents cross-node mutation
  through shared references.

### 4.7 Lazy `If` semantics

`IfInvocation` now acts as a lazy branch boundary rather than a simple value multiplexer.

- The `condition` input must resolve first.
- Nodes that are exclusive to the true or false branch can remain deferred even when their indegree is zero.
- Once the prepared `If` node resolves its condition:
  - the selected branch is released
  - the unselected branch is marked skipped
  - unselected input edges on the prepared `If` exec node are pruned from the execution graph so they no longer
    participate in downstream indegree accounting
  - branch-exclusive ancestors of the unselected branch are never executed
- Skipped branch-local exec nodes may still be treated as executed for scheduling purposes, but they do not create
  entries in `results`.
- Shared ancestors still execute if they are required by the selected branch or by any other live path in the graph.

This behavior is implemented in the runtime scheduler, not in the invocation body itself.

## 5) Traversal Summary

1. Author builds a valid **Graph**.

1. Create **GraphExecutionState** with that graph.

1. Loop:

   - `node = state.next()` -> may trigger `_prepare()` expansion.
   - Execute node externally -> `output`.
   - `state.complete(node.id, output)` -> updates indegrees, `If` state, and ready queues.

1. Finish when `next()` returns `None` and the execution state is not paused waiting on a workflow call boundary.

In normal execution, all runtime expansion occurs in `execution_graph` with traceability back to source nodes.

## 6) Invariants

- Source **Graph** remains a DAG and type-consistent.
- `execution_graph` remains a DAG.
- Nodes are enqueued only when `indegree == 0` and they are not deferred by an unresolved `If`.
- `results` and `errors` are keyed by **exec node id**.
- Collectors aggregate `item` inputs and may also merge incoming `collection` inputs during runtime hydration.
- Branch-exclusive nodes behind an unselected `If` branch are skipped, not failed.

## 7) Extensibility

- **New node types**: implement as Pydantic models with typed fields and outputs. Register per your invocation system;
  this file accepts them as `AnyInvocation`.
- **Scheduling policy**: adjust `ready_order` to batch by class; add a batch cap for fairness without changing
  complexity.
- **Dynamic behaviors** (future): can be added in `GraphExecutionState` by creating exec nodes and edges at `complete()`
  time, as long as the DAG invariant holds.
- **Workflow call boundaries**: `GraphExecutionState` can suspend a parent execution state on a workflow call, attach a
  child execution state, and later resume the parent without mutating the source graph.

Current limitation:

- Child workflow executions are now represented as first-class queue items. Parent resume/failure is intentionally
  handled by a dedicated workflow-call queue lifecycle component for this PR because no other feature currently needs a
  generalized dependent-queue scheduler.
- Called workflows currently require exactly one valid `workflow_return` node to be callable at all.
- Direct batch-special child workflows are now supported by expanding them into multiple child queue rows.
- Batch outputs may feed `workflow_return.collection` directly; each expanded child receives a singleton collection and
  parent resume aggregates those child return collections.
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
- **Indegree + queues** gives O(1) scheduling decisions with clear batching semantics.
- **Iterator/collector separation** keeps fan-out/fan-in explicit and testable.
- **Deep-copy hydration** avoids incidental aliasing bugs between nodes.
