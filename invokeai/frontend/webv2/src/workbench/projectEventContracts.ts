/**
 * Project event vocabulary, split out of `projectContracts.ts` as a leaf.
 *
 * `CanvasProjectMutation` (owned by `canvas-engine/mutationContracts.ts`) carries
 * a `ProjectEvent` on its staging commits. Keeping these two types in
 * `projectContracts.ts` would force the engine to import a module that imports
 * `canvas-engine/api` right back — the edge that closed the 16-module import
 * cycle. This module imports nothing, so both sides can depend on it freely.
 */

export type ProjectEventType =
  | 'project-created'
  | 'layout-updated'
  | 'invocation-updated'
  | 'queue-submitted'
  | 'canvas-layer-accepted'
  | 'graph-replaced'
  | 'graph-snapshot-saved';

export interface ProjectEvent {
  id: string;
  type: ProjectEventType;
  createdAt: string;
  summary: string;
  runId?: string;
}
