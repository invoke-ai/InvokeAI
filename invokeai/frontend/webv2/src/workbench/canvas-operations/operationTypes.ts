import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type { GuardedMutationRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasImageRef, ControlType } from '@workbench/canvas-engine/contracts';
import type {
  RasterFilterCommitTarget,
  RasterFilterSettings,
} from '@workbench/canvas-engine/controllers/filterResultController';
import type { SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect, Vec2 } from '@workbench/canvas-engine/types';
export type { SamVisualInput } from '@workbench/canvas-engine/samInteraction';

export type SamModel = 'segment-anything-2-large' | 'segment-anything-huge';
export type SamPointLabel = 'include' | 'exclude';
export type SamSessionErrorCode =
  | 'invalid'
  | 'not-ready'
  /** The layer's Clarity, Texture or Dehaze could not run on the source, so it was not used without them. */
  | 'adjustment-too-large'
  | 'adjustment-over-budget'
  | 'adjustment-error'
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
/**
 * A refused apply, a source the filter cannot read (hidden, empty, over the raster budget), or an unexpected failure
 * to process or apply whose message is kept as `detail`.
 */
export type FilterSessionErrorCode =
  | GuardedMutationRefusal
  | 'disabled'
  | 'empty'
  | 'source-over-budget'
  /** The layer's Clarity, Texture or Dehaze could not run on the source, so it was not filtered without them. */
  | 'source-adjustment-too-large'
  | 'source-adjustment-over-budget'
  | 'source-adjustment-error'
  | 'process-failed'
  | 'apply-failed';
export interface FilterSessionError {
  code: FilterSessionErrorCode;
  detail?: string;
}
export interface FilterOperationSessionState {
  autoProcess: boolean;
  draft: LayerFilterSettings;
  error: FilterSessionError | null;
  initialFilter: LayerFilterSettings | null;
  layerId: string;
  layerName: string;
  layerType: 'raster' | 'control';
  preview: FilterOperationPreview | null;
  status: 'ready' | 'processing' | 'committing' | 'error';
}
export type FilterCommitTarget = RasterFilterCommitTarget;

/**
 * A control layer edit from the Properties panel. `set-type` may rename in the same history entry; `set-detector`
 * starts from the detector's defaults with the type's fixed settings; `draw` empties the image for painting.
 */
export type ControlLayerIntent =
  | { kind: 'set-type'; controlType: ControlType | null; name?: string }
  | { kind: 'set-detector'; detector: string | null }
  | { kind: 'set-detector-settings'; settings: Record<string, unknown> }
  | { kind: 'replace-image'; image: CanvasImageRef; transform?: LayerTransform }
  | { kind: 'draw' };
/**
 * How an intent ended. `unchanged`: the layer already matched. `no-detection-input`: the layer's pixels are a map
 * without an original, so a detector cannot run until the image is replaced. `missing`: no such control layer.
 * `locked`: the layer or a group above it is locked. `blocked`: an operation, gesture or session holds the
 * document; try again once it ends. `over-budget` and `failed`: the engine could not apply the edit.
 */
export type ControlLayerIntentResult =
  | 'updated'
  | 'unchanged'
  | 'no-detection-input'
  | 'missing'
  | 'locked'
  | 'blocked'
  | 'over-budget'
  | 'failed';
/** `not-needed`: the map is current or there is nothing to detect from. */
export type RetryControlDetectionResult = 'scheduled' | 'not-needed' | 'missing';
/** Typed detection failures; the view translates the code and shows `detail` as technical detail. */
export type ControlDetectionErrorCode =
  | 'process-failed'
  | 'apply-failed'
  | 'unknown-detector'
  | 'output-dimensions'
  | 'source-not-ready'
  | 'over-budget'
  | 'commit-busy'
  | 'unrepresentable';
export interface ControlDetectionError {
  code: ControlDetectionErrorCode;
  detail?: string;
}
export type ControlDetectionStatus =
  | { status: 'idle' }
  | { status: 'queued' }
  | { status: 'processing'; detector: string }
  | { status: 'error'; error: ControlDetectionError };
