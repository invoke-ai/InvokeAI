# Call Saved Workflow Architecture

## Goal

`CallSavedWorkflowInvocation` should become an engine-native workflow call boundary, not a frontend-only dynamic node
and not a compile-time graph inliner.

The long-term feature goal is:

- A parent workflow can call a saved workflow selected by ID.
- The call node redraws in the editor based on the selected workflow's exposed form fields.
- Parent values and inbound connections bind to those exposed fields as call arguments.
- Execution suspends at the call node, runs the selected workflow as a dependent workflow execution, captures explicit
  return values, and then resumes the parent workflow.
- The architecture must work for Invoke frontend graphs and for externally submitted graphs that use the same node type.

This document records the current state, the target architecture, and the execution contract needed to continue
development later.

## Implementation Priority

Favor the architecturally correct design over the fastest implementation path.

The work may still proceed incrementally, but each increment should satisfy all of the following:

- testable in isolation
- compatible with the long-term architecture described here
- non-breaking to existing code and existing workflow execution behavior

Speed is not the primary goal for this phase. The primary goal is to move toward the durable design without introducing
throwaway execution semantics that would need to be unwound later.

## Current State

Implemented already in the branch:

- A real invocation exists: `call_saved_workflow`.
- A real return node exists: `workflow_return`.
- `workflow_return` accepts a `list[Any]` collection input and returns that collection through a dedicated output.
- Only one `workflow_return` node is allowed per workflow, enforced in both frontend validation and Python validation.
- The frontend provides a saved-workflow picker using a reusable `SavedWorkflowField` UI type.
- The node redraws dynamically based on the selected saved workflow's exposed form fields.
- Dynamic field values persist with the parent workflow.
- Compatible inbound edges are preserved when switching between workflows with matching exposed field identities and
  compatible types.
- Incompatible or no-longer-exposed inbound edges are removed in the editor.
- Backend validation exists for `workflow_id` existence and access rights.

Implemented runtime scaffolding:

- `GraphExecutionState` now persists workflow-call runtime state:
  - `workflow_call_stack`
  - `workflow_call_history`
  - `workflow_call_parent`
  - `waiting_workflow_call`
  - `waiting_workflow_call_execution`
  - `waiting_workflow_call_child_session`
  - `max_workflow_call_depth`
- Nested and recursive calls are represented by the stack, with a runtime depth cap of 4.
- Parent/child workflow-call identity is now explicit in runtime state:
  - the parent tracks an active `WorkflowCallExecution` record while waiting
  - completed and failed calls are preserved in `workflow_call_history`
  - child sessions carry a `workflow_call_parent` reference back to the parent call relationship
- `GraphExecutionState.next()` returns no runnable node while the parent session is waiting on a child workflow call.
- `GraphExecutionState.is_complete()` stays false while waiting.
- `DefaultSessionRunner.run_node()` now treats `call_saved_workflow` as a call boundary instead of a normal executable
  node.
- On boundary entry, the runner:
  - validates the selected workflow
  - builds a workflow call frame
  - converts the saved workflow JSON into a backend `Graph`
  - validates and applies parent call arguments to the child graph
  - creates a child `GraphExecutionState`
  - attaches that child session to the waiting parent session
- Workflow-call runtime responsibilities are now split:
  - `WorkflowCallCoordinator` handles call-specific setup:
    - build the child graph
    - apply parent call arguments
    - create the child `GraphExecutionState`
    - suspend the parent and enqueue the child queue item
  - `WorkflowCallQueueLifecycle` handles queue-visible parent/child lifecycle:
    - run child queue items
    - resume waiting parents after child success
    - complete the parent call node with the child `workflow_return` collection
    - fail suspended parents after child failure and cascade that failure upward through parent call chains
- Child `SessionQueueItem` rows now carry explicit relationship metadata:
  - `workflow_call_id`
  - `parent_item_id`
  - `parent_session_id`
  - `root_item_id`
  - `workflow_call_depth`
  - this metadata is now used directly by queue-visible child execution and parent resume/failure handling
- The `session_queue` table now has matching durable columns for that relationship metadata:
  - `workflow_call_id`
  - `parent_item_id`
  - `parent_session_id`
  - `root_item_id`
  - `workflow_call_depth`
  - child workflow executions are now inserted as their own pending queue rows using those columns
- Parent queue items now enter a real `waiting` status while suspended on a child workflow execution.
- `_on_after_run_session()` no longer completes queue items whose sessions are incomplete but waiting.
- Dynamic call arguments now execute end-to-end in the current runner path:
  - literal dynamic values are serialized into a hidden `workflow_inputs` payload on the parent node
  - connected dynamic values are accepted as special call-boundary edges and are resolved from parent results at runtime
  - both are validated against the child workflow's exposed form interface before being applied to the child graph
- Queue lifecycle semantics now exist for workflow-call chains:
  - parent queue items are suspended in `waiting` while a child queue row runs
  - child success resumes the suspended parent and completes the parent call node with the child `workflow_return`
  - child failure fails the suspended parent and cascades upward through any waiting parent chain
  - canceling a parent cancels its descendant child chain
  - canceling a child cancels the waiting parent chain upward
  - deleting any queue row in a workflow-call chain deletes the full chain to avoid leaving orphaned parent or child
    rows behind
  - retry is root-oriented rather than child-oriented; child queue rows should not be directly retried from the UI
  - the current UI policy is:
    - child queue rows keep `Cancel`
    - child queue rows hide `Retry`
  - child queue-row creation is now fail-clean:
    - if call-boundary setup fails after some child rows have already been inserted, those child rows are deleted before
      the parent invocation is failed
  - child queue-row fan-out is bounded by remaining queue capacity, not just the global queue-size setting:
    - a workflow call that would exceed the remaining pending capacity now fails instead of silently truncating or
      over-enqueuing child rows

Implemented conversion helper:

- `workflow_graph_builder.py` converts saved workflow JSON into an executable backend `Graph`.
- It currently supports the invocation-node subset needed for this feature.
- It flattens connector nodes and omits explicit destination field values when a connection exists, matching frontend
  graph-build semantics.
- It now serves as the first explicit callable-workflow compatibility gate:
  - the selected workflow must contain exactly one `workflow_return` node
  - connected batch child inputs produced by ordinary non-generator upstream nodes still fail early with a clear
    unsupported-feature error
  - malformed batch input wiring, including multiple connected inputs to one batch field, is reported as
    `unsupported_batch_input` compatibility rather than a generic unsupported-node failure
  - child workflows that mix supported batch nodes with unrelated generator nodes are currently rejected with a clear
    unsupported-feature error
  - unsupported callees are rejected before any child queue row is created
- Compatibility metadata is now exposed through workflow library API responses:
  - workflow list items and workflow detail responses include `call_saved_workflow_compatibility`
  - the saved-workflow picker uses that metadata to disable unsupported workflows before execution
  - the picker still allows an already-selected unsupported workflow to render, with an explicit unsupported state and
    backend-provided reason message
  - workflow library list items now surface an explicit unsupported badge and backend-provided reason message without
    blocking normal workflow viewing or editing

What is still not implemented:

- connected batch child inputs whose batch values are produced by ordinary non-generator upstream nodes are still not
  supported and must fail with a clear domain error
- child workflows that mix supported batch nodes with unrelated generator nodes are still not supported and must fail
  with a clear domain error
- broader child-workflow compatibility coverage still needs to be expanded from real unsupported shapes rather than
  trying to interpret every frontend-only workflow representation through the current graph-builder path
- the current workflow-call queue lifecycle is still implemented through dedicated workflow-call runtime classes rather
  than a fully generalized parent/child scheduler model

Conclusion:

- the editor contract is largely in place
- the parent-side runtime call boundary is in place
- child execution, argument forwarding, explicit child return capture, suspended parent status, queue-visible child
  rows, and upward failure cascade now work
- the remaining major runtime work is to harden and generalize the parent/child scheduler model rather than prove the
  basic call boundary

## Architectural Direction

Use the architecture that is more likely to be kept long-term:

- `call_saved_workflow` is a call boundary.
- The parent graph does not inline the full child workflow into itself at queue time.
- Runtime execution pauses at the call node and creates a dependent child workflow execution.
- The child workflow receives arguments from the parent.
- The child workflow returns explicit outputs to the parent.
- The parent resumes once the child returns successfully.

This is preferred over full graph expansion because it:

- avoids execution-graph blowup
- preserves workflow boundaries
- matches the conceptual model of workflow reuse
- supports explicit return values
- keeps externally submitted graphs viable as long as they use the same node type and contract

## Non-Goals For The Next Phase

These should not be the first implementation target:

- full inline graph expansion of called workflows
- unlimited nested workflow call support
- automatic exposure of arbitrary internal child workflow state
- implicit output inference from arbitrary child nodes

## Execution Contract

### 1. Callable Interface

The callable interface of a saved workflow is defined by its saved workflow JSON.

Primary source:

- `workflow.form`

Fallback source for older workflows:

- `workflow.exposedFields`

Only fields exposed by the child workflow form are callable inputs. Internal child inputs that exist in the workflow
graph but are not exposed by the form are not part of the public call interface.

### 2. Input Arguments

`CallSavedWorkflowInvocation` exposes dynamic inputs in the editor based on the selected workflow's callable interface.

Each dynamic input must have:

- a stable external handle name
- a type
- a default value if defined by the child workflow
- a user-facing label and description when available

Current fast-path identity is based on child `nodeId + fieldName`. That is acceptable short-term in the editor, but a
longer-term stable interface ID would be better if child workflows are frequently duplicated or refactored.

### 3. Input Binding At Runtime

At runtime, when the parent reaches `call_saved_workflow`:

- the engine resolves `workflow_id`
- the engine loads the selected child workflow record
- the engine reconstructs the callable interface from the saved workflow JSON
- the engine collects argument values from the parent node's dynamic inputs
- the engine starts a dependent child workflow execution using those arguments

Argument values may come from:

- parent literal field values
- resolved inbound connections into the call node's dynamic inputs

For batch-aware child workflows, the parent call boundary should still pass normal exposed form inputs. Batching should
emerge from the child workflow's own internal batch nodes or generators, not from a separate caller-side batch protocol.

### 4. Child Workflow Execution

The child workflow runs as its own dependent execution context, not as an inlined copy of the parent graph.

Desired semantics:

- parent execution pauses at the call node
- child execution runs with inherited context where appropriate
- child workflow finishes or fails
- parent resumes only if child execution succeeds

This implies the queue/session/runtime layer needs an explicit parent-child execution relationship.

Current limitation:

- the temporary `workflow_graph_builder.py` path still reconstructs only the ordinary invocation subset of child
  workflows
- direct batch-special child workflows now bypass that path and use queue batch expansion instead
- generator-backed batch child workflows now bypass that path too when the batch is fed directly by a supported
  generator node
- connected batch child inputs produced by ordinary non-generator upstream nodes are still not supported and should fail
  early with a clear unsupported-feature error
- the current queue-visible child execution path still relies on `WorkflowCallCoordinator` to resume or fail parents
  directly rather than a more general queue scheduler abstraction
- the current implementation is still an intermediate architecture step, but it is now materially closer to the intended
  durable parent/child model than the earlier inline-runner path

### 4a. Queue Lifecycle Contract

The current queue-visible implementation uses the following lifecycle contract:

- root or parent queue items may enter `waiting` while suspended on a child workflow call
- child workflow executions are represented as real queue rows with explicit parent/child relationship metadata
- child completion resumes the suspended parent and returns control to normal queue execution
- child failure fails the suspended parent call node and cascades upward through any ancestor chain
- cancel operations are chain-aware:
  - canceling a waiting parent cancels descendants
  - canceling a child cancels waiting ancestors
- retry operations are root-aware:
  - retrying a root queue item creates a new root execution
  - retrying a child queue item should be normalized to the root by backend code
  - child queue rows should not expose direct retry affordances in the UI
  - retry websocket delivery is owner-scoped; when an admin retries roots owned by multiple users, each non-admin user
    must receive only the retry item ids for their own roots, while admins can still observe the full retried set

This is now part of the intended user-facing contract, even though the orchestration still lives in
`WorkflowCallCoordinator`.

### 4b. Batch Child Workflows

The current implementation now supports direct batch-special child workflows for:

- `image_batch`
- `string_batch`
- `integer_batch`
- `float_batch`

It also supports generator-backed batch child workflows when those batch nodes are fed directly by:

- `integer_generator`
- `float_generator`
- `string_generator`
- `image_generator` using `image_generator_images_from_board`

Current semantics:

- batch-special nodes are removed from the executable child graph before ordinary graph validation
- supported generator nodes that feed those batch-special nodes are removed from the executable child graph as well
- their outgoing edges are converted into queue batch substitutions
- ungrouped batch nodes expand as a cartesian product
- grouped batch nodes zip by `batch_group_id`
- the workflow call creates one child queue row per expanded batch session
- supported generator value shapes are resolved into concrete batch items before queue batch expansion
- batch outputs may feed `workflow_return.collection` directly; each expanded child receives a singleton collection, and
  the parent still aggregates all returned child collections
- parent resume waits for all child rows tied to that workflow call
- parent return aggregation appends each child `workflow_return.collection` into one parent collection
- if any child row fails, remaining sibling child rows are canceled and the parent call fails
- generator-backed image batches must respect board access:
  - the caller may expand images from a board they own
  - admins may expand any board
  - shared/public boards may be expanded by other users
  - inaccessible private boards must fail before image expansion rather than leaking board contents across users

Current generator coverage:

- integer generators:
  - arithmetic sequence
  - linear distribution
  - parse string
  - seeded uniform random distribution
- float generators:
  - arithmetic sequence
  - linear distribution
  - parse string
  - seeded uniform random distribution
- string generators:
  - parse string
  - dynamic prompts combinatorial
  - dynamic prompts random
- image generators:
  - images from board

Still unsupported:

- connected batch inputs whose batch values are produced by non-generator upstream nodes

Plain-English summary:

1. The parent workflow reaches `call_saved_workflow`.
1. The parent pauses and enters `waiting`.
1. The child workflow is inspected before execution.
1. If the child contains supported batch inputs, that one call expands into multiple child executions instead of one.
1. Each expanded child execution becomes its own queue row.
1. Those child queue rows run independently.
1. The parent does not resume until all child queue rows for that call have finished.
1. Each child execution produces its own `workflow_return.collection`.
1. The parent aggregates those returned collections into one combined collection.
1. The `call_saved_workflow` node completes with that combined collection, and the parent workflow continues.

Expansion rules:

- ungrouped batch inputs expand as a cartesian product
- batch inputs that share the same `batch_group_id` zip together by position

Example:

- ungrouped inputs `[1, 2]` and `[10, 20]` produce 4 child executions:
  - `(1, 10)`
  - `(1, 20)`
  - `(2, 10)`
  - `(2, 20)`
- grouped inputs `[1, 2, 3]` and `[10, 20, 30]` with the same `batch_group_id` produce 3 child executions:
  - `(1, 10)`
  - `(2, 20)`
  - `(3, 30)`

### 4c. Tricky Areas

The following parts of the runtime contract are easy to misread and should stay explicit in both code and tests.

Waiting and resume:

- a parent queue row in `waiting` is suspended, not completed
- a parent resumes only after every child queue row tied to that workflow call has reached a terminal state

Return aggregation:

- each child queue row returns its own `workflow_return.collection`
- the parent call node output is the aggregate of those child collections, in child-completion order
- this is different from returning exactly one child collection unchanged

Sibling failure behavior:

- if one child queue row in a batched workflow call fails, remaining sibling child rows for that same workflow call are
  canceled
- after sibling cancelation, the parent call fails
- if that parent is itself a child of another workflow call, failure continues upward through the ancestor chain

Cancel behavior:

- canceling a waiting parent cancels descendant child rows
- canceling a child row cancels waiting ancestors
- cancelation should stay cancelation; it should not be rewritten into ordinary failure semantics

Retry behavior:

- retry is root-oriented
- child queue rows should not be directly retried from the UI
- backend retry of a child id should normalize to the root workflow call chain rather than create an isolated child-only
  rerun

### 5. Return Values

Return values should be explicit.

Recommended model:

- introduce a workflow return node analogous in concept to Canvas Output
- the child workflow declares what values it returns through that explicit node
- the return node accepts a `list[Any]` collection input
- when the workflow is run independently, the return node has no caller-visible effect
- when the workflow is run via `call_saved_workflow`, that collection becomes the return value of the call
- `call_saved_workflow` should expose that collection as its return value in the first runtime version

Only one workflow return node may exist per workflow. That rule should be enforced in both the frontend editor and in
Python validation/runtime code.

Do not infer child outputs from arbitrary terminal nodes. That is too ambiguous and too brittle.

### 6. Error Propagation

If child execution fails:

- the call node fails
- the parent workflow fails unless a later design adds explicit error-handling semantics

For the first implementation, failure propagation should be simple and strict.

### 7. Access Control

Runtime must enforce the same access rules used elsewhere for saved workflows.

The caller may execute a child workflow only if it is allowed to access that saved workflow at runtime.

This matters even if the parent workflow was authored in a context where the child was once visible.

### 8. Recursion And Nesting

Nested and recursive `call_saved_workflow` execution should be allowed, but bounded.

Initial implementation should enforce:

- nested workflow calls are allowed
- recursive workflow calls are allowed
- maximum workflow call depth is capped at 4 call frames
- the depth cap is enforced at runtime, based on the active call stack, not by static validation alone

This allows legitimate recursive or conditionally terminating workflow structures while still preventing unbounded call
growth.

## Where The Runtime Work Belongs

The goal is to support externally submitted graphs, not only frontend-authored graphs. Therefore the authoritative
execution logic must live in Python.

Recommended high-level design:

- a backend `GraphExpander` or broader graph-preparation service may still exist as an abstraction point
- but for this feature, the preferred long-term runtime model is not full graph expansion
- instead, the runtime needs a call-execution mechanism in the Python execution stack

Relevant existing path:

- frontend builds and submits a graph and workflow payload
- backend receives the batch via session queue APIs
- session queue stores session state
- runtime executes through `GraphExecutionState`

Current insertion points already used:

- `DefaultSessionRunner.run_node()` detects `call_saved_workflow` and enters boundary state
- `GraphExecutionState` stores the waiting/call-stack state and attached child session
- `WorkflowCallCoordinator` currently establishes the call boundary and enqueues child workflow executions as real queue
  rows
- `WorkflowCallQueueLifecycle` currently resumes or fails parents when those child rows complete
- child queue items already carry stable parent/child identifiers in both runtime objects and durable queue columns

Next runtime work still needed:

- keep `WorkflowCallQueueLifecycle` as the bounded workflow-call lifecycle component for this PR
  - the current workflow-call feature is the only caller of parent/child queue semantics
  - replacing it with a generalized queue dependency scheduler now would add regression risk without unlocking a
    concrete user workflow
  - revisit only if another feature needs dependent queue items, richer retry/cancel policies, or resumable waits
- if support expands beyond the currently supported direct and generator-backed batch shapes, route those new child
  workflow execution shapes through machinery that can honor ordinary Invoke batch semantics

## Suggested Runtime Components

### CallSavedWorkflowRuntime

A dedicated runtime helper for this node type should be introduced. Responsibilities:

- load and validate the selected child workflow record
- validate runtime access rights
- extract callable inputs from the child workflow definition
- build child execution arguments from the parent node state
- launch dependent execution
- collect declared returns
- map returned values back to the parent node outputs

### Workflow Return Node

A dedicated child-workflow return node should be introduced. Responsibilities:

- define the return interface of the called workflow
- accept a `list[Any]` collection input representing the workflow result
- provide that collection back to the parent call site when invoked through `call_saved_workflow`
- remain inert from a caller perspective when the workflow is run independently
- guarantee that only one such node exists per workflow
- behave as a normal node in the editor, with singularity enforced by both frontend and Python validation/runtime code

This should likely become the canonical reusable return mechanism for any future subworkflow call behavior.

### Execution Relationship Tracking

Session/runtime state will likely need to record:

- parent execution waiting on child execution
- child execution belonging to a parent node call site
- result propagation back to the parent
- strict failure propagation rules

### Workflow Return Value Flow

The workflow return value should not be persisted back into the saved workflow record and should not be derived from
frontend state.

The intended runtime flow is:

1. The child workflow computes the `workflow_return` node's collection input like any other node input.
1. When the child reaches `workflow_return`, runtime captures the resolved collection value as the child workflow
   result.
1. The child workflow result is stored in child execution state.
1. That result is handed back to the suspended parent call frame.
1. The parent `call_saved_workflow` node is completed with that returned collection.
1. The parent graph resumes.

## Frontend Responsibilities In The Long-Term Design

The frontend remains responsible for editor-time behavior:

- choosing the saved workflow
- redrawing dynamic inputs based on the child workflow callable interface
- persisting those dynamic fields and their values
- preserving compatible inbound edges when workflow selection changes
- clearing incompatible edges and invalid selections in a predictable way
- using backend compatibility metadata so unsupported saved workflows are not presented as callable choices
  - compatibility analysis now tolerates required exposed caller inputs by synthesizing placeholder values for those
    inputs during backend compatibility evaluation, so workflows that are valid once the caller supplies exposed values
    are not disabled prematurely

Potential future optimization:

- add a backend endpoint that returns a normalized callable workflow interface
- this would let the frontend avoid re-parsing full saved workflow payloads to redraw the node
- it would also give the frontend a backend-authoritative interface hash for drift detection

## Tests Needed Going Forward

Already covered:

- workflow-call stack and waiting state on `GraphExecutionState`
- depth-limit enforcement
- waiting blocks scheduling
- parent sessions are not completed while waiting
- runner boundary entry for `call_saved_workflow`
- validation failures and depth-limit failures still follow normal node-error behavior
- child workflow JSON conversion to backend `Graph`
- child graph build failure does not leave the parent in a partial waiting state
- child `GraphExecutionState` is attached to the waiting parent session
- coordinator-owned child execution completes the parent queue item instead of leaving it stuck in `in_progress`
- literal and connected dynamic call arguments are applied to the child graph at runtime
- non-exposed dynamic call arguments are rejected at runtime
- child `workflow_return` output is captured and becomes the parent `call_saved_workflow` output
- child workflows without a `workflow_return` node fail cleanly when called
- child execution events now include stable workflow-call relationship metadata on the child `SessionQueueItem`
- parent-child resume and failure propagation through queue-visible child rows
- nested runtime execution with bounded stack depth
- direct and generator-backed batch-special child workflows through queue child-row expansion
- compatibility metadata for required exposed inputs, missing/multiple returns, supported batch-to-return collection
  shapes, and unsupported batch input wiring

Still needed in later increments:

- focused coverage for any newly supported batch or generator shape when its contract changes
- possible migration from dedicated workflow-call queue lifecycle handling to a more general scheduler or
  queue-lifecycle model only if another feature needs reusable dependent queue items

## Recommended Immediate Next Step

The next incremental step should be:

- stop adding feature slices unless they close a concrete correctness gap or unlock a realistic user workflow
- stabilize the current branch with review, targeted test runs, and cleanup of stale design-doc language
- treat migration from `WorkflowCallQueueLifecycle` to a generalized parent/child queue lifecycle as a larger
  architecture slice, not as small follow-on busywork

The current branch is at the point where:

- parent call-boundary state exists
- child execution state can be created from the selected saved workflow
- child execution, argument forwarding, explicit return propagation, suspended parent status, queue-visible child rows,
  and upward failure cascade work through the current coordinator + queue path
- but long-term generalized parent/child scheduling semantics are still missing
