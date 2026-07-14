import type { CommitGeneratedImageResult, ReplaceSelectionFromImageResult } from '@workbench/canvas-engine/api';
import type { CommitRasterFilterResult } from '@workbench/canvas-engine/controllers/filterResultController';
import type {
  CommitMaskImageResult,
  MaskImageResultTarget,
} from '@workbench/canvas-engine/controllers/maskResultController';
import type {
  CanvasApplicationOperationStores,
  CanvasOperationController,
  SelectObjectSessionProcessResult,
} from '@workbench/canvas-operations/contracts';
import type {
  FilterCommitTarget,
  LayerFilterSettings,
  SamInput,
  SamModel,
  SamPointLabel,
} from '@workbench/canvas-operations/operationTypes';

export type StartSelectObjectSessionResult =
  | 'started'
  | 'missing'
  | 'disabled'
  | 'locked'
  | 'unsupported'
  | 'not-ready';
export type StartFilterOperationResult = 'started' | 'missing' | 'disabled' | 'locked' | 'unsupported' | 'not-ready';
export type CanvasOperationActionResult = 'completed' | 'blocked' | 'stale';
export type FilterCommitOperationResult = 'committed' | 'blocked' | 'stale';
export type CanvasOperationMutationResult = 'updated' | 'blocked' | 'stale';
export type SelectObjectSaveTarget = 'selection' | 'raster' | 'control' | MaskImageResultTarget;
export type SaveSelectObjectSessionResult =
  | CommitGeneratedImageResult
  | CommitMaskImageResult
  | ReplaceSelectionFromImageResult;

export interface SelectObjectSessionUpdate {
  input?: SamInput;
  pointLabel?: SamPointLabel;
  model?: SamModel;
  invert?: boolean;
  applyPolygonRefinement?: boolean;
  autoProcess?: boolean;
  isolatedPreview?: boolean;
}

export interface CanvasOperationCapability {
  readonly controller: CanvasOperationController;
  readonly stores: CanvasApplicationOperationStores;
  startSelectObject(layerId: string): StartSelectObjectSessionResult;
  startFilterOperation(layerId: string, recommendedFilterType?: string | null): StartFilterOperationResult;
  updateFilterOperation(draft: LayerFilterSettings): CanvasOperationMutationResult;
  setFilterOperationAutoProcess(value: boolean): CanvasOperationMutationResult;
  processFilterOperation(): Promise<CanvasOperationActionResult>;
  resetFilterOperation(settings: Record<string, unknown>): CanvasOperationMutationResult;
  commitFilterOperation(
    target: FilterCommitTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<FilterCommitOperationResult>;
  cancelFilterOperation(): void;
  updateSelectObjectSession(changes: SelectObjectSessionUpdate): CanvasOperationMutationResult;
  processSelectObjectSession(): Promise<SelectObjectSessionProcessResult>;
  applySelectObjectSession(makeImageDurable: (imageName: string) => Promise<void>): Promise<CommitGeneratedImageResult>;
  saveSelectObjectSession(
    target: SelectObjectSaveTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<SaveSelectObjectSessionResult>;
  resetSelectObjectSession(): CanvasOperationMutationResult;
  cancelSelectObjectSession(): void;
}

export type { CommitRasterFilterResult };
