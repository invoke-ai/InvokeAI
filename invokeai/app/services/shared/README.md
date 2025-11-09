# InvokeAI Graph - Design Overview

High-level design for the graph module. Focuses on responsibilities, data flow, and how traversal works.

## 1) Purpose

Provide a typed, acyclic workflow model (**Graph**) plus a runtime scheduler (**GraphExecutionState**) that expands
iterator patterns, tracks readiness via indegree (the number of incoming edges to a node in the directed graph), and
executes nodes in class-grouped batches. Source graphs remain immutable during a run; runtime expansion happens in a
separate execution graph.

## 2) Major Data Types

### EdgeConnection

* Fields: `node_id: str`, `field: str`.
* Hashable; printed as `node.field` for readable diagnostics.

### Edge

* Fields: `source: EdgeConnection`, `destination: EdgeConnection`.
* One directed connection from a specific output port to a specific input port.

### AnyInvocation / AnyInvocationOutput

* Pydantic wrappers that carry concrete invocation models and outputs.
* No registry logic in this file; they are permissive containers for heterogeneous nodes.

### IterateInvocation / CollectInvocation

* Control nodes used by validation and execution:

  * **IterateInvocation**: input `collection`, outputs include `item` (and index/total).
  * **CollectInvocation**: many `item` inputs aggregated to one `collection` output.

## 3) Graph (author-time model)

A container for declared nodes and edges. Does **not** perform iteration expansion.

### 3.1 Data

* `nodes: dict[str, AnyInvocation]` - key must equal `node.id`.
* `edges: list[Edge]` - zero or more.
* Utility: `_get_input_edges(node_id, field?)`, `_get_output_edges(node_id, field?)`
  These scan `self.edges` (no adjacency indices in the current code).

### 3.2 Validation (`validate_self`)

Runs a sequence of checks:

1. **Node ID uniqueness**
   No duplicate IDs; map key equals `node.id`.
2. **Endpoint existence**
   Source and destination node IDs must exist.
3. **Port existence**
   Input ports must exist on the node class; output ports on the node's output model.
4. **Type compatibility**
   `get_output_field_type` vs `get_input_field_type` and `are_connection_types_compatible`.
5. **DAG constraint**
   Build a *flat* `DiGraph` (no runtime expansion) and assert acyclicity.
6. **Iterator / collector structure**
   Enforce special rules:

   * Iterator's input must be `collection`; its outgoing edges use `item`.
   * Collector accepts many `item` inputs; outputs a single `collection`.
   * Edge fan-in to a non-collector input is rejected.

### 3.3 Edge admission (`_validate_edge`)

Checks a single prospective edge before insertion:

* Endpoints/ports exist.
* Destination port is not already occupied unless it's a collector `item`.
* Adding the edge to the flat DAG must keep it acyclic.
* Iterator/collector constraints re-checked when the edge creates relevant patterns.

### 3.4 Topology utilities

* `nx_graph()` - DiGraph of declared nodes and edges.
* `nx_graph_with_data()` - includes node/edge attributes.
* `nx_graph_flat()` - "flattened" DAG (still author-time; no runtime copies).
  Used in validation and in `_prepare()` during execution planning.

### 3.5 Mutation helpers

* `add_node`, `update_node` (preserve edges, rewrite endpoints if id changes), `delete_node`.
* `add_edge`, `delete_edge` (with validation).

## 4) GraphExecutionState (runtime)

Holds the state for a single run. Keeps the source graph intact; materializes a separate execution graph.

### 4.1 Data

* `graph: Graph` - immutable source during a run.
* `execution_graph: Graph` - materialized runtime nodes/edges.
* `executed: set[str]`, `executed_history: list[str]`.
* `results: dict[str, AnyInvocationOutput]`, `errors: dict[str, str]`.
* `prepared_source_mapping: dict[str, str]` - exec id → source id.
* `source_prepared_mapping: dict[str, set[str]]` - source id → exec ids.
* `indegree: dict[str, int]` - unmet inputs per exec node.
* **Ready queues grouped by class** (private attrs):
  `_ready_queues: dict[class_name, deque[str]]`, `_active_class: Optional[str]`. Optional `ready_order: list[str]` to
  prioritize classes.

### 4.2 Core methods

* `next()`
  Returns the next ready exec node. If none, calls `_prepare()` to materialize more, then retries. Before returning a
  node, `_prepare_inputs()` deep-copies inbound values into the node fields.
* `complete(node_id, output)`
  Record result; mark exec node executed; if all exec copies of the same **source** are done, mark the source executed.
  For each outgoing exec edge, decrement child indegree and enqueue when it reaches zero.

### 4.3 Preparation (`_prepare()`)

* Build a flat DAG from the **source** graph.
* Choose the **next source node** in topological order that:

  1. has not been prepared,
  2. if it is an iterator, *its inputs are already executed*,
  3. it has *no unexecuted iterator ancestors*.
* If the node is a **CollectInvocation**: collapse all prepared parents into one mapping and create **one** exec node.
* Otherwise: compute all combinations of prepared iterator ancestors. For each combination, pick the matching prepared parent per upstream and create **one** exec node.
* For each new exec node:

  * Deep-copy the source node; assign a fresh ID (and `index` for iterators).
  * Wire edges from chosen prepared parents.
  * Set `indegree = number of unmet inputs` (i.e., parents not yet executed).
  * If `indegree == 0`, enqueue into its class queue.

### 4.4 Readiness and batching

* `_enqueue_if_ready(nid)` enqueues by class name only when `indegree == 0` and not executed.
* `_get_next_node()` drains the `_active_class` queue FIFO; when empty, selects the next nonempty class queue (by `ready_order` if set, else alphabetical), and continues. Optional fairness knobs can limit batch size per class; default is drain fully.

#### 4.4.1 Indegree (what it is and how it's used)

**Indegree** is the number of incoming edges to a node in the execution graph that are still unmet. In this engine:
* For every materialized exec node, `indegree[node]` equals the count of its prerequisite parents that have **not** finished yet.
* A node is "ready" exactly when `indegree[node] == 0`; only then is it enqueued.
* When a node completes, the scheduler decrements `indegree[child]` for each outgoing edge. Any child that reaches 0 is enqueued.

Example: edges `A→C`, `B→C`, `C→D`. Start: `A:0, B:0, C:2, D:1`. Run `A` → `C:1`. Run `B` → `C:0` → enqueue `C`. Run `C`
→ `D:0` → enqueue `D`. Run `D` → done.

### 4.5 Input hydration (`_prepare_inputs()`)

* For **CollectInvocation**: gather all incoming `item` values into `collection`.
* For all others: deep-copy each incoming edge's value into the destination field.
  This prevents cross-node mutation through shared references.

## 5) Traversal Summary

1. Author builds a valid **Graph**.
2. Create **GraphExecutionState** with that graph.
3. Loop:

   * `node = state.next()` → may trigger `_prepare()` expansion.
   * Execute node externally → `output`.
   * `state.complete(node.id, output)` → updates indegrees and queues.
4. Finish when `next()` returns `None`.

The source graph is never mutated; all expansion occurs in `execution_graph` with traceability back to source nodes.

## 6) Invariants

* Source **Graph** remains a DAG and type-consistent.
* `execution_graph` remains a DAG.
* Nodes are enqueued only when `indegree == 0`.
* `results` and `errors` are keyed by **exec node id**.
* Collectors only aggregate `item` inputs; other inputs behave one-to-one.

## 7) Extensibility

* **New node types**: implement as Pydantic models with typed fields and outputs. Register per your invocation system; this file accepts them as `AnyInvocation`.
* **Scheduling policy**: adjust `ready_order` to batch by class; add a batch cap for fairness without changing complexity.
* **Dynamic behaviors** (future): can be added in `GraphExecutionState` by creating exec nodes and edges at `complete()` time, as long as the DAG invariant holds.

## 8) Error Model (selected)

* `DuplicateNodeIdError`, `NodeAlreadyInGraphError`
* `NodeNotFoundError`, `NodeFieldNotFoundError`
* `InvalidEdgeError`, `CyclicalGraphError`
* `NodeInputError` (raised when preparing inputs for execution)

Messages favor short, precise diagnostics (node id, field, and failing condition).

## 9) Rationale

* **Two-graph approach** isolates authoring from execution expansion and keeps validation simple.
* **Indegree + queues** gives O(1) scheduling decisions with clear batching semantics.
* **Iterator/collector separation** keeps fan-out/fan-in explicit and testable.
* **Deep-copy hydration** avoids incidental aliasing bugs between nodes.
