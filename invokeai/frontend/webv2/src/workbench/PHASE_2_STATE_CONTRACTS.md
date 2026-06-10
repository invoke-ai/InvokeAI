# Phase 2 State Contracts

Phase 2 locks down ownership boundaries before feature migration.

## Ownership

- `WorkbenchState.account` owns global user/workbench preferences such as the active layout preset registry selection.
- `WorkbenchState.autosave` owns persistence status only; it is excluded from the autosave comparison key to avoid save loops.
- `Project.layout` owns the active project layout. Layout presets are copied into the project instead of referenced as live global state.
- `Project.invocation` owns source, destination, and lock state for the global Invoke control.
- `Project.projectGraph` owns the primary project graph.
- `Project.widgetStates` owns widget-local state values.
- `Project.widgetGraphs` owns graph-bearing widget graphs that can be selected as invocation sources.
- `Project.canvas` owns canvas layers and staging-area state.
- `Project.graphHistory` owns immutable graph snapshots.
- `Project.undoRedo` owns undo/redo snapshots for project and widget state updates.
- `Project.queue` owns immutable queue submission records for the active project.
- `Project.events` owns internal timeline primitives. Conversation and Run Journal UI remain deferred.

## Persistence

`WorkbenchPersistenceService` defines the persistence boundary. The initial implementation uses `localStorage` for autosave, but feature code should depend on the service interface rather than direct storage calls.

## Undo/Redo Policy

Undoable project changes push a `ProjectUndoSnapshot` before mutation and clear redo history. Queue submissions are not undoable because they represent immutable external work requests.

## Queue Snapshot Policy

Clicking Invoke creates a `QueueItem` with a frozen-by-copy `QueueSubmissionSnapshot`:

- resolved source and destination
- selected graph copy
- widget state copy
- canvas state copy
- submission timestamp

Backend cancellation is represented per item through `QueueItem.cancellable`, allowing later API integration to expose cancellation only when supported.
