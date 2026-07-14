import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type { History } from '@workbench/canvas-engine/history/history';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasImageRef } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import {
  createInpaintMaskFromImage,
  createRegionalGuidanceFromImage,
  DEFAULT_INPAINT_MASK_FILL,
  nextInpaintMaskName,
  nextRegionalGuidanceFillColor,
  nextRegionalGuidanceName,
} from '@workbench/canvas-engine/document/layerFactories';

export type MaskImageResultTarget = 'inpaint_mask' | 'regional_guidance';
export interface CommitMaskImageResultOptions {
  guard: LayerExportGuard;
  image: CanvasImageRef;
  rect: Rect;
  target: MaskImageResultTarget;
  signal?: AbortSignal;
}
export type CommitMaskImageResult =
  | { status: 'committed'; layerId: string }
  | { status: 'aborted' | 'missing' | 'locked' | 'stale' | 'unsupported' | 'busy' };

export interface MaskResultControllerOptions<Owner = symbol> {
  readonly canEdit: (owner?: Owner) => boolean;
  readonly createLayerId: () => string;
  readonly dispatchPrepared: (
    action: WorkbenchAction,
    reducerAccepted: () => boolean,
    mirrorAccepted: () => boolean
  ) => void;
  readonly endBurst: () => void;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly history: History;
  readonly isGestureActive: () => boolean;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
}

/** Converts a guarded object-selection result into a structural mask layer. */
export class MaskResultController<Owner = symbol> {
  constructor(private readonly options: MaskResultControllerOptions<Owner>) {}

  commit(options: CommitMaskImageResultOptions, owner?: Owner): Promise<CommitMaskImageResult> {
    const o = this.options;
    if (!o.canEdit(owner)) {
      return Promise.resolve({ status: 'busy' });
    }
    if (options.signal?.aborted) {
      return Promise.resolve({ status: 'aborted' });
    }
    const document = o.getDocument();
    if (!document) {
      return Promise.resolve({ status: 'missing' });
    }
    const liveLayer = document.layers.find((candidate) => candidate.id === options.guard.layerId);
    if (!liveLayer) {
      return Promise.resolve({ status: 'missing' });
    }
    if (liveLayer.isLocked) {
      return Promise.resolve({ status: 'locked' });
    }
    if (liveLayer.type !== 'raster' && liveLayer.type !== 'control') {
      return Promise.resolve({ status: 'unsupported' });
    }
    const sourceIndex = document.layers.findIndex((candidate) => candidate.id === liveLayer.id);
    if (o.isGestureActive()) {
      return Promise.resolve({ status: 'busy' });
    }
    if (!o.isGuardCurrent(options.guard)) {
      return Promise.resolve({ status: 'stale' });
    }
    if (options.signal?.aborted) {
      return Promise.resolve({ status: 'aborted' });
    }
    if (sourceIndex < 0) {
      return Promise.resolve({ status: 'missing' });
    }
    const names = document.layers.map((layer) => layer.name);
    const layerId = o.createLayerId();
    const layer =
      options.target === 'inpaint_mask'
        ? createInpaintMaskFromImage({
            fill: DEFAULT_INPAINT_MASK_FILL,
            id: layerId,
            image: options.image,
            name: nextInpaintMaskName(names),
            rect: options.rect,
          })
        : createRegionalGuidanceFromImage({
            fill: {
              color: nextRegionalGuidanceFillColor(
                document.layers.filter((candidate) => candidate.type === 'regional_guidance').length
              ),
              style: 'solid',
            },
            id: layerId,
            image: options.image,
            name: nextRegionalGuidanceName(names),
            rect: options.rect,
          });
    const selectedLayerId = document.selectedLayerId;
    const apply = (): void =>
      o.dispatchPrepared(
        { index: sourceIndex, layer, type: 'addCanvasLayer' },
        () => o.getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () => o.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );
    o.endBurst();
    apply();
    o.history.push({
      bytes: 256,
      label: options.target === 'inpaint_mask' ? 'Create inpaint mask from object' : 'Create region from object',
      redo: apply,
      replayFailureAtomic: true,
      undo: () => {
        o.dispatchPrepared(
          { id: selectedLayerId, type: 'setCanvasSelectedLayer' },
          () => o.getReducerDocument()?.selectedLayerId === selectedLayerId,
          () => o.getDocument()?.selectedLayerId === selectedLayerId
        );
        o.dispatchPrepared(
          { ids: [layerId], type: 'removeCanvasLayers' },
          () => o.getReducerDocument()?.layers.some((candidate) => candidate.id === layerId) === false,
          () => o.getDocument()?.layers.some((candidate) => candidate.id === layerId) === false
        );
      },
    });
    return Promise.resolve({ layerId, status: 'committed' });
  }

  dispose(): void {}
}
