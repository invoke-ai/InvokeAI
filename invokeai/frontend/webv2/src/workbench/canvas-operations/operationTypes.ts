import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type {
  RasterFilterCommitTarget,
  RasterFilterSettings,
} from '@workbench/canvas-engine/controllers/filterResultController';
import type { SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { Rect, Vec2 } from '@workbench/canvas-engine/types';
export type { SamVisualInput } from '@workbench/canvas-engine/samInteraction';

export type SamModel = 'segment-anything-2-large' | 'segment-anything-huge';
export type SamPointLabel = 'include' | 'exclude';
export type SamSessionErrorCode =
  | 'invalid'
  | 'not-ready'
  | 'empty'
  | 'upload'
  | 'queue'
  | 'no-output'
  | 'reconcile'
  | 'output-dimension'
  | 'decode'
  | 'locked'
  | 'unknown';
export interface SamSessionError {
  code: SamSessionErrorCode;
  detail?: string;
}
export type SamInput =
  | { type: 'prompt'; prompt: string }
  | { type: 'visual'; includePoints: readonly Vec2[]; excludePoints: readonly Vec2[]; bbox?: Rect | null };
export type SamSessionInput =
  | SamVisualInput
  | { type: 'prompt'; prompt: string; includePoints?: never; excludePoints?: never; bbox?: never };
export interface SamSessionSnapshot {
  sourceRect: Rect;
  layerName: string;
  layerType: 'raster' | 'control';
  input: SamSessionInput;
  pointLabel: SamPointLabel;
  model: SamModel;
  invert: boolean;
  applyPolygonRefinement: boolean;
  autoProcess: boolean;
  isolatedPreview: boolean;
  status:
    | 'ready'
    | 'scheduled'
    | 'preparing-source'
    | 'uploading'
    | 'processing-sam'
    | 'rendering-preview'
    | 'committing'
    | 'error';
  error: SamSessionError | null;
  hasPreview: boolean;
}
export type LayerFilterSettings = RasterFilterSettings;
export interface FilterOperationPreview {
  guard: LayerExportGuard;
  imageName: string;
  rect: Rect;
  width: number;
  height: number;
  origin: { x: number; y: number };
}
export interface FilterOperationSessionState {
  autoProcess: boolean;
  draft: LayerFilterSettings;
  error: string | null;
  initialFilter: LayerFilterSettings | null;
  layerId: string;
  layerName: string;
  layerType: 'raster' | 'control';
  preview: FilterOperationPreview | null;
  status: 'ready' | 'processing' | 'committing' | 'error';
}
export type FilterCommitTarget = RasterFilterCommitTarget;
