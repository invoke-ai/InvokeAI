import type { CommitGeneratedImageOptions, CommitGeneratedImageResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasImageRef, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { LayerMutationControllerOptions } from '@workbench/canvas-engine/controllers/layerMutationController';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';

import { getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createControlLayer, nextControlLayerName } from '@workbench/canvas-engine/document/layerFactories';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';

import type { CanvasMutationContext } from './mutationContext';

import {
  addLayerStep,
  guardedResultRefusal,
  layerEditRefusal,
  removeLayerStep,
  replaceLayerStep,
  replayWithPixels,
  rgbaBytes,
} from './editSteps';

export interface GeneratedResultControllerOptions {
  readonly captureCache: LayerMutationControllerOptions['captureCache'];
  readonly clearPreview: (layerId: string) => void;
  readonly ctx: Pick<
    CanvasMutationContext,
    | 'applyStep'
    | 'begin'
    | 'capturePermit'
    | 'captureInsertionAnchor'
    | 'createLayerId'
    | 'getDocument'
    | 'installPrepared'
    | 'isGestureActive'
    | 'isGuardCurrent'
    | 'isPermitCurrent'
    | 'preparePixels'
    | 'reserveRaster'
  >;
  readonly decodeImage: (
    image: CanvasImageRef,
    options: { signal?: AbortSignal; isCurrent?: () => boolean }
  ) => Promise<DecodeImageResult>;
  readonly discardPersisted: (layerId: string) => void;
  readonly getDefaultControlModel: (base: string | null) => string | null;
  readonly getMainModelBase: () => string | null;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
}

/** Publishes guarded workflow/SAM images as replacements or copies. */
export class GeneratedResultController {
  constructor(private readonly options: GeneratedResultControllerOptions) {}

  async commit(options: CommitGeneratedImageOptions, owner?: symbol): Promise<CommitGeneratedImageResult> {
    const o = this.options;
    const permit = o.ctx.capturePermit(owner);
    if (!permit) {
      return { status: 'busy' };
    }
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    const validate = ():
      | { document: CanvasDocumentContractV3; liveLayer: Extract<CanvasLayerContract, { type: 'raster' | 'control' }> }
      | { result: CommitGeneratedImageResult } => {
      if (!o.ctx.isPermitCurrent(permit)) {
        return { result: { status: 'busy' } };
      }
      const document = o.ctx.getDocument();
      if (!document) {
        return { result: { status: 'missing' } };
      }
      const liveLayer = getDocumentLayer(document, options.guard.layerId);
      if (!liveLayer) {
        return { result: { status: 'missing' } };
      }
      if (liveLayer.isLocked) {
        return { result: { status: 'locked' } };
      }
      if (liveLayer.type !== 'raster' && liveLayer.type !== 'control') {
        return { result: { status: 'unsupported' } };
      }
      if (o.ctx.isGestureActive()) {
        return { result: { status: 'busy' } };
      }
      if (!o.ctx.isGuardCurrent(options.guard)) {
        return { result: { status: 'stale' } };
      }
      return { document, liveLayer };
    };
    try {
      const decoded = await o.decodeImage(options.image, {
        isCurrent: () => o.ctx.isPermitCurrent(permit),
        signal: options.signal,
      });
      if (decoded.status !== 'ok') {
        return { status: decoded.status === 'aborted' ? 'aborted' : 'busy' };
      }
      const checked = validate();
      if ('result' in checked) {
        return checked.result;
      }
      const { document, liveLayer } = checked;
      const image = structuredClone(options.image);
      const origin = { ...options.origin };
      const rect = { height: image.height, width: image.width, ...origin };
      const source = { bitmap: image, offset: origin, type: 'paint' } as const;
      const identityTransform: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
      const afterPixels = { pixels: decoded.surface, rect };
      const txn = o.ctx.begin({ historyBytes: rgbaBytes(rect) + HISTORY_ENTRY_OVERHEAD_BYTES, owner });
      if (!('publish' in txn)) {
        return { status: layerEditRefusal(txn.status) };
      }
      try {
        if (!txn.reserveRaster(rgbaBytes(rect))) {
          return { status: 'over-budget' };
        }
        if (options.target === 'replace') {
          const beforePixels = o.captureCache(liveLayer, document, (rect) => txn.growHistory(rgbaBytes(rect)));
          if (beforePixels === 'not-ready' || beforePixels === 'over-budget') {
            return { status: beforePixels };
          }
          if (!beforePixels) {
            return { status: 'stale' };
          }
          const before = structuredClone(liveLayer);
          let after: CanvasLayerContract;
          if (liveLayer.type === 'raster') {
            const { adjustments: _adjustments, ...base } = structuredClone(liveLayer);
            after = { ...base, source, transform: identityTransform };
          } else {
            after = { ...structuredClone(liveLayer), source, transform: identityTransform };
          }
          const clearPreview = (): void => o.clearPreview(liveLayer.id);
          const publishAfter = (): void => {
            o.discardPersisted(liveLayer.id);
            clearPreview();
          };
          const prepared = o.ctx.preparePixels(liveLayer.id, rect, decoded.surface);
          const interrupted = this.interrupted(options, validate);
          if (interrupted) {
            return interrupted;
          }
          const result = txn.publish(
            options.historyLabel ?? 'Replace layer with workflow result',
            replaceLayerStep(o.ctx, after, prepared, { notify: publishAfter, persist: false, restore: liveLayer }),
            {
              bytes: rgbaBytes(beforePixels.rect) + rgbaBytes(rect) + HISTORY_ENTRY_OVERHEAD_BYTES,
              heldAssetRefs: collectHistoryMediaRefs(before, after),
              redo: () =>
                replayWithPixels(o.ctx, liveLayer.id, afterPixels, (prepared) =>
                  replaceLayerStep(o.ctx, after, prepared, { notify: publishAfter, persist: false, restore: before })
                ),
              undo: () =>
                replayWithPixels(o.ctx, liveLayer.id, beforePixels, (prepared) =>
                  replaceLayerStep(o.ctx, before, prepared, {
                    notify: clearPreview,
                    persist: o.needsPixelPersistence(before),
                    restore: after,
                  })
                ),
            }
          );
          return result.status === 'committed'
            ? { layerId: liveLayer.id, status: 'committed' }
            : { status: guardedResultRefusal(result) };
        }
        const layerId = o.ctx.createLayerId();
        const selectedLayerId = document.selectedLayerId;
        const buildControlCopy = (): CanvasLayerContract => {
          const mainBase = o.getMainModelBase();
          return {
            ...createControlLayer(
              options.copyLayerName ?? nextControlLayerName(getDocumentLeaves(document).map((layer) => layer.name)),
              layerId,
              mainBase,
              o.getDefaultControlModel(mainBase)
            ),
            source,
            transform: identityTransform,
          };
        };
        const copy: CanvasLayerContract =
          options.target === 'copy-control'
            ? buildControlCopy()
            : {
                blendMode: 'normal',
                id: layerId,
                isEnabled: true,
                isLocked: false,
                name: options.copyLayerName ?? `${liveLayer.name} workflow result`,
                opacity: 1,
                source,
                transform: identityTransform,
                type: 'raster',
              };
        const anchor = o.ctx.captureInsertionAnchor(copy.type, liveLayer.id);
        const added = (prepared: PreparedLayerCacheReplacement | null) =>
          addLayerStep(o.ctx, copy, anchor, prepared, { persist: false, previousSelectedLayerId: selectedLayerId });
        const prepared = o.ctx.preparePixels(layerId, rect, decoded.surface);
        const interrupted = this.interrupted(options, validate);
        if (interrupted) {
          return interrupted;
        }
        const result = txn.publish(
          options.target === 'copy-control'
            ? 'Copy workflow result to control layer'
            : 'Copy workflow result to raster layer',
          added(prepared),
          {
            bytes: rgbaBytes(rect) + HISTORY_ENTRY_OVERHEAD_BYTES,
            heldAssetRefs: collectHistoryMediaRefs(copy),
            redo: () => replayWithPixels(o.ctx, layerId, afterPixels, added),
            undo: () => o.ctx.applyStep(removeLayerStep(layerId, selectedLayerId)),
          }
        );
        return result.status === 'committed'
          ? { layerId, status: 'committed' }
          : { status: guardedResultRefusal(result) };
      } finally {
        txn.end();
      }
    } catch (error) {
      if (options.signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        return { status: 'aborted' };
      }
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  }

  dispose(): void {}

  /** Preparation may run long enough for the caller to abort or the source to change; check both last. */
  private interrupted(
    options: CommitGeneratedImageOptions,
    validate: () => { result: CommitGeneratedImageResult } | object
  ): CommitGeneratedImageResult | null {
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    const checked = validate();
    return 'result' in checked ? checked.result : null;
  }
}
