# Call Saved Workflow Architecture

## Goal

`CallSavedWorkflowInvocation` should become an engine-native workflow call boundary, not a frontend-only dynamic node and not a compile-time graph inliner.

The long-term feature goal is:

- A parent workflow can call a saved workflow selected by ID.
- The call node redraws in the editor based on the selected workflow's exposed form fields.
- Parent values and inbound connections bind to those exposed fields as call arguments.
- Execution suspends at the call node, runs the selected workflow as a dependent workflow execution, captures explicit return values, and then resumes the parent workflow.
- The architecture must work for Invoke frontend graphs and for externally submitted graphs that use the same node type.

This document records the current state, the target architecture, and the execution contract needed to continue development later.

## Implementation Priority

Favor the architecturally correct design over the fastest implementation path.

The work may still proceed incrementally, but each increment should satisfy all of the following:

- testable in isolation
- compatible with the long-term architecture described here
- non-breaking to existing code and existing workflow execution behavior

Speed is not the primary goal for this phase. The primary goal is to move toward the durable design without introducing throwaway execution semantics that would need to be unwound later.

## Current State

Implemented already:

- A real invocation exists: `call_saved_workflow`.
- The frontend provides a saved-workflow picker using a reusable `SavedWorkflowField` UI type.
- The node redraws dynamically based on the selected saved workflow's exposed form fields.
- Dynamic field values persist with the parent workflow.
- Compatible inbound edges are preserved when switching between workflows with matching exposed field identities and compatible types.
- Incompatible or no-longer-exposed inbound edges are removed in the editor.
- Backend validation exists for `workflow_id` existence and access rights.

Important limitation:

- The backend invocation class only has a static `workflow_id` input.
- Dynamic exposed fields currently exist only in frontend/editor state.
- Fresh connections to dynamic handles fail at invoke time because backend graph validation checks destination fields against real Python model fields.

Conclusion:

- More frontend work alone will not make the node executable.
- The next phase must be Python-side runtime architecture.

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

Only fields exposed by the child workflow form are callable inputs.
Internal child inputs that exist in the workflow graph but are not exposed by the form are not part of the public call interface.

### 2. Input Arguments

`CallSavedWorkflowInvocation` exposes dynamic inputs in the editor based on the selected workflow's callable interface.

Each dynamic input must have:

- a stable external handle name
- a type
- a default value if defined by the child workflow
- a user-facing label and description when available

Current fast-path identity is based on child `nodeId + fieldName`. That is acceptable short-term in the editor, but a longer-term stable interface ID would be better if child workflows are frequently duplicated or refactored.

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

### 4. Child Workflow Execution

The child workflow runs as its own dependent execution context, not as an inlined copy of the parent graph.

Desired semantics:

- parent execution pauses at the call node
- child execution runs with inherited context where appropriate
- child workflow finishes or fails
- parent resumes only if child execution succeeds

This implies the queue/session/runtime layer needs an explicit parent-child execution relationship.

### 5. Return Values

Return values should be explicit.

Recommended model:

- introduce a workflow return node analogous in concept to Canvas Output
- the child workflow declares what values it returns through that explicit node
- the return node accepts a `list[Any]` collection input
- when the workflow is run independently, the return node has no caller-visible effect
- when the workflow is run via `call_saved_workflow`, that collection becomes the return value of the call
- `call_saved_workflow` should expose that collection as its return value in the first runtime version

Only one workflow return node may exist per workflow.
That rule should be enforced in both the frontend editor and in Python validation/runtime code.

Do not infer child outputs from arbitrary terminal nodes.
That is too ambiguous and too brittle.

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

Initial implementation should forbid:

- direct self-call
- obvious recursion cycles
- nested `call_saved_workflow` inside a called child workflow

This keeps the first runtime implementation bounded.
Nested calls can be revisited later.

## Where The Runtime Work Belongs

The goal is to support externally submitted graphs, not only frontend-authored graphs.
Therefore the authoritative execution logic must live in Python.

Recommended high-level design:

- a backend `GraphExpander` or broader graph-preparation service may still exist as an abstraction point
- but for this feature, the preferred long-term runtime model is not full graph expansion
- instead, the runtime needs a call-execution mechanism in the Python execution stack

Relevant existing path:

- frontend builds and submits a graph and workflow payload
- backend receives the batch via session queue APIs
- session queue stores session state
- runtime executes through `GraphExecutionState`

The next phase should identify the best Python insertion point for:

- detecting when the next executable node is `call_saved_workflow`
- suspending parent execution
- launching a dependent child execution
- collecting child return values
- resuming the parent graph

At a minimum, expect changes in Python runtime/session code rather than only in queue submission code.

## Suggested Runtime Components

### CallSavedWorkflowRuntime

A dedicated runtime helper for this node type should be introduced.
Responsibilities:

- load and validate the selected child workflow record
- validate runtime access rights
- extract callable inputs from the child workflow definition
- build child execution arguments from the parent node state
- launch dependent execution
- collect declared returns
- map returned values back to the parent node outputs

### Workflow Return Node

A dedicated child-workflow return node should be introduced.
Responsibilities:

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

The workflow return value should not be persisted back into the saved workflow record and should not be derived from frontend state.

The intended runtime flow is:

1. The child workflow computes the `workflow_return` node's collection input like any other node input.
2. When the child reaches `workflow_return`, runtime captures the resolved collection value as the child workflow result.
3. That result is stored in child execution state, or equivalent parent-child call-frame state, until the child finishes.
4. When the child finishes successfully, the captured collection is passed back to the suspended parent call site.
5. `call_saved_workflow` completes using that collection as its output value.
6. The parent workflow resumes execution.

Consequences of this model:

- `workflow_return` is a normal invocation node in the child workflow
- only one workflow return result may exist, because only one return node is allowed per workflow
- the child result should live in runtime/session state, not in workflow persistence
- return propagation should be explicit and deterministic

## Frontend Responsibilities In The Long-Term Design

The frontend remains responsible for editor-time behavior:

- choosing the saved workflow
- redrawing dynamic inputs based on the child workflow callable interface
- persisting those dynamic fields and their values
- preserving compatible inbound edges when workflow selection changes
- removing no-longer-valid inbound edges when the callable interface changes
- eventually redrawing outputs if and when explicit workflow returns are added

The frontend should not be the authoritative implementation of execution semantics.

## Questions To Resolve Before Coding The Runtime

1. Where exactly does parent execution pause and child execution resume in the current runtime stack?
2. What is the narrowest first implementation of parent-child session state?
3. The first runtime version should use the explicit workflow return node with a single collection-valued return, rather than inputs-only or ad hoc fixed outputs.
4. Should child execution inherit all parent execution context, or only selected parts?
5. What cancellation semantics apply if the parent session is cancelled while a child workflow is running?
6. What metadata should be stored on queue items or sessions to represent call relationships and the captured child return value?
7. Do dynamic input identities need a more stable external interface ID before runtime work begins?

## Recommended Next Steps

1. Design the explicit workflow return mechanism.
2. Trace the Python runtime path needed to suspend and resume execution around a call node.
3. Define a minimal parent-child session relationship model.
4. Prototype runtime input passing for `call_saved_workflow` without nested calls.
5. Add the workflow return node with frontend and Python enforcement that only one return node may exist per workflow.
6. Add strict recursion guards.
7. Add end-to-end tests for successful child call execution, missing child workflow, unauthorized child workflow, duplicate return nodes, and child failure propagation.

## Minimum Test Matrix For The Next Phase

Positive tests:

- parent workflow calls child workflow successfully with literal arguments
- parent workflow calls child workflow successfully with connected arguments
- child workflow returns its explicit `list[Any]` collection value to the parent
- runtime enforces child defaults when parent does not override them

Negative tests:

- missing selected workflow fails cleanly
- unauthorized selected workflow fails cleanly
- child workflow missing return definition fails cleanly if returns are required
- duplicate workflow return nodes are rejected
- direct self-call is rejected
- nested workflow calls are rejected in the first implementation
- child workflow failure propagates to the parent
- cancellation while child is running produces a deterministic failure state

## Summary

The project is past the frontend proof-of-concept stage.

What remains is a real engine-level workflow call mechanism.
The architecture most likely to be kept is a runtime call boundary with dependent child execution and explicit returns, not compile-time graph inlining.

That should be the basis for the next phase of work.
