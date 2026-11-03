/**
 * Keep ProjectEvent dependency-free: CanvasProjectMutation consumes it, while projectContracts imports
 * canvas-engine/api.
 */

export type ProjectEventType =
  | 'project-created'
  | 'layout-updated'
  | 'invocation-updated'
  | 'queue-submitted'
  | 'canvas-layer-accepted'
  | 'graph-replaced';

export interface ProjectEvent {
  id: string;
  type: ProjectEventType;
  createdAt: string;
  summary: string;
  runId?: string;
}
