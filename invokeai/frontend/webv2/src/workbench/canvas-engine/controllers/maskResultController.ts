import type { CommitMaskImageResult, CommitMaskImageResultOptions } from '@workbench/canvas-engine/capabilities';

import { getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import {
  createInpaintMaskFromImage,
  createRegionalGuidanceFromImage,
  DEFAULT_INPAINT_MASK_FILL,
  nextInpaintMaskName,
  nextRegionalGuidanceFillColor,
  nextRegionalGuidanceName,
} from '@workbench/canvas-engine/document/layerFactories';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';

import type { CanvasMutationContext } from './mutationContext';

import { addLayerStep, guardedResultRefusal, layerEditRefusal, removeLayerStep } from './editSteps';

export type {
  CommitMaskImageResult,
  CommitMaskImageResultOptions,
  MaskImageResultTarget,
} from '@workbench/canvas-engine/capabilities';

export interface MaskResultControllerOptions {
  readonly ctx: Pick<
    CanvasMutationContext,
    | 'applyStep'
    | 'begin'
    | 'canEdit'
    | 'captureInsertionAnchor'
    | 'createLayerId'
    | 'getDocument'
    | 'installPrepared'
    | 'isGestureActive'
    | 'isGuardCurrent'
    | 'preparePixels'
    | 'reserveRaster'
  >;
}

/** Converts a guarded object-selection result into a structural mask layer. */
export class MaskResultController {
  constructor(private readonly options: MaskResultControllerOptions) {}

  commit(options: CommitMaskImageResultOptions, owner?: symbol): Promise<CommitMaskImageResult> {
    const o = this.options;
    if (!o.ctx.canEdit(owner)) {
      return Promise.resolve({ status: 'busy' });
    }
    if (options.signal?.aborted) {
      return Promise.resolve({ status: 'aborted' });
    }
    const document = o.ctx.getDocument();
    if (!document) {
      return Promise.resolve({ status: 'missing' });
    }
    const liveLayer = getDocumentLayer(document, options.guard.layerId);
    if (!liveLayer) {
      return Promise.resolve({ status: 'missing' });
    }
    if (liveLayer.isLocked) {
      return Promise.resolve({ status: 'locked' });
    }
    if (liveLayer.type !== 'raster' && liveLayer.type !== 'control') {
      return Promise.resolve({ status: 'unsupported' });
    }
    if (o.ctx.isGestureActive()) {
      return Promise.resolve({ status: 'busy' });
    }
    if (!o.ctx.isGuardCurrent(options.guard)) {
      return Promise.resolve({ status: 'stale' });
    }
    if (options.signal?.aborted) {
      return Promise.resolve({ status: 'aborted' });
    }
    const names = getDocumentLeaves(document).map((layer) => layer.name);
    const layerId = o.ctx.createLayerId();
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
                getDocumentLeaves(document).filter((candidate) => candidate.type === 'regional_guidance').length
              ),
              style: 'solid',
            },
            id: layerId,
            image: options.image,
            name: nextRegionalGuidanceName(names),
            rect: options.rect,
          });
    const selectedLayerId = document.selectedLayerId;
    const anchor = o.ctx.captureInsertionAnchor(layer.type, liveLayer.id);
    const txn = o.ctx.begin({ historyBytes: HISTORY_ENTRY_OVERHEAD_BYTES, owner });
    if (!('publish' in txn)) {
      return Promise.resolve({ status: layerEditRefusal(txn.status) });
    }
    try {
      const added = () =>
        addLayerStep(o.ctx, layer, anchor, null, { persist: false, previousSelectedLayerId: selectedLayerId });
      const result = txn.publish(
        options.target === 'inpaint_mask' ? 'Create inpaint mask from object' : 'Create region from object',
        added(),
        {
          bytes: HISTORY_ENTRY_OVERHEAD_BYTES,
          heldAssetRefs: collectHistoryMediaRefs(layer),
          redo: () => o.ctx.applyStep(added()),
          undo: () => o.ctx.applyStep(removeLayerStep(layerId, selectedLayerId)),
        }
      );
      return Promise.resolve(
        result.status === 'committed' ? { layerId, status: 'committed' } : { status: guardedResultRefusal(result) }
      );
    } finally {
      txn.end();
    }
  }

  dispose(): void {}
}
