import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { getDocumentLayer, isNodeAbsent } from '@workbench/canvas-engine/document/documentIndex';

import type { CanvasMutationContext, EditPublishResult, EditStep } from './mutationContext';

export type LayerResultContext = Pick<
  CanvasMutationContext,
  'applyStep' | 'installPrepared' | 'preparePixels' | 'reserveRaster'
>;

/** Layer-local pixels a result or its undo step installs. */
export interface LayerPixels {
  readonly pixels: RasterSurface;
  readonly rect: Rect;
}

export const pixelBytes = (rect: Rect): number => Math.max(0, rect.width) * Math.max(0, rect.height) * 4;

/** Swaps `contract` in for its layer and, once the document accepted it, installs the prepared pixels. */
export const replaceLayerStep = (
  ctx: LayerResultContext,
  contract: CanvasLayerContract,
  prepared: PreparedLayerCacheReplacement | null,
  options: { readonly persist: boolean; readonly restore: CanvasLayerContract; readonly notify?: () => void }
): EditStep => ({
  accepted: (document) => getDocumentLayer(document, contract.id) === contract,
  install: prepared ? () => ctx.installPrepared(prepared, options.persist) : undefined,
  mutation: { layer: contract, layerId: contract.id, type: 'replaceCanvasLayer' },
  notify: options.notify,
  rollback: {
    mutation: { layer: options.restore, layerId: contract.id, type: 'replaceCanvasLayer' },
    restored: (document) => getDocumentLayer(document, contract.id) === options.restore,
  },
});

/** Inserts `layer` (selecting it) and installs its prepared pixels; rolls back to the prior selection. */
export const addLayerStep = (
  ctx: LayerResultContext,
  layer: CanvasLayerContract,
  anchor: CanvasNodeInsertionAnchor,
  prepared: PreparedLayerCacheReplacement | null,
  options: { readonly persist: boolean; readonly previousSelectedLayerId: string | null }
): EditStep => ({
  accepted: (document) => getDocumentLayer(document, layer.id) === layer,
  install: prepared ? () => ctx.installPrepared(prepared, options.persist) : undefined,
  mutation: { anchor, layer, type: 'addCanvasLayer' },
  rollback: {
    mutation: removeLayerMutation(layer.id, options.previousSelectedLayerId),
    restored: (document) => isNodeAbsent(document, layer.id),
  },
});

const removeLayerMutation = (layerId: string, selectedLayerId: string | null) =>
  ({ enabledUpdates: [], removeIds: [layerId], selectedLayerId, type: 'applyCanvasLayerStackMutation' }) as const;

/** Removes a layer an edit added and restores the selection it replaced. */
export const removeLayerStep = (layerId: string, selectedLayerId: string | null): EditStep => ({
  accepted: (document) => isNodeAbsent(document, layerId) && document?.selectedLayerId === selectedLayerId,
  mutation: removeLayerMutation(layerId, selectedLayerId),
});

/**
 * Replays a step that installs `snapshot`, reserving the prepared copy first. Throws (leaving the entry in place)
 * when the copy does not fit or the step does not land.
 */
export const replayWithPixels = (
  ctx: LayerResultContext,
  layerId: string,
  snapshot: LayerPixels | null,
  step: (prepared: PreparedLayerCacheReplacement | null) => EditStep
): void => {
  if (!snapshot) {
    ctx.applyStep(step(null));
    return;
  }
  const reservation = ctx.reserveRaster(pixelBytes(snapshot.rect));
  if (!reservation) {
    throw new Error('Not enough raster memory to restore this step.');
  }
  try {
    ctx.applyStep(step(ctx.preparePixels(layerId, snapshot.rect, snapshot.pixels)));
  } finally {
    reservation.release();
  }
};

/** The refusal a failed publication reports to its caller: a step that did not land is stale. */
export const publishRefusal = (
  result: Exclude<EditPublishResult, { status: 'committed' }>
): 'busy' | 'not-ready' | 'over-budget' | 'stale' => {
  switch (result.status) {
    case 'gesture-active':
      return 'busy';
    case 'dispatch-rejected':
    case 'postcondition-failed':
      return 'stale';
    default:
      return result.status;
  }
};
