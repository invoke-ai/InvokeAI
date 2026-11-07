import type { CanvasBezierPathContract } from './contracts';
import type { Rect, ToolId } from './types';

export interface VectorPointRef {
  pathId: string;
  pointIndex: number;
}

export type VectorEditDestination = { kind: 'tool'; toolId: ToolId } | { kind: 'layer'; layerId: string | null };

/** Engine-local draft: the persisted layer is unchanged until Apply. */
export interface VectorEditSession {
  layerId: string;
  paths: CanvasBezierPathContract[];
  activePathId: string | null;
  selectedPoints: VectorPointRef[];
  selectionRect: Rect | null;
  canUndo: boolean;
  canRedo: boolean;
  pendingExit: VectorEditDestination | null;
}
