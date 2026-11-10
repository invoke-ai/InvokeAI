import type { CanvasBezierPathContract } from './contracts';
import type { Rect } from './types';
import type { LayerTransform } from './transform/transformMath';

export interface VectorPointRef {
  pathId: string;
  pointIndex: number;
}

/** Point-tool selection and in-flight gesture preview; completed edits are persisted immediately. */
export interface VectorEditSession {
  layerId: string;
  paths: CanvasBezierPathContract[];
  activePathId: string | null;
  selectedPoints: VectorPointRef[];
  canSplit: boolean;
  canJoin: boolean;
  pathTransform: { pathId: string; rect: Rect; transform: LayerTransform } | null;
  selectionRect: Rect | null;
}
