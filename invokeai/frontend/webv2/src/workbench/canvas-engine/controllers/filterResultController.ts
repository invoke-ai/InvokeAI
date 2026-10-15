import type { CommitRasterFilterOptions, CommitRasterFilterResult } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasControlLayerContract,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { LayerMutationControllerOptions } from '@workbench/canvas-engine/controllers/layerMutationController';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createControlLayer } from '@workbench/canvas-engine/document/layerFactories';
import { LayerFilterOutputDimensionError } from '@workbench/canvas-engine/filterError';
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

export type {
  CommitRasterFilterOptions,
  CommitRasterFilterResult,
  RasterFilterCommitTarget,
  RasterFilterSettings,
} from '@workbench/canvas-engine/capabilities';

export interface FilterResultControllerOptions {
  readonly captureCache: LayerMutationControllerOptions['captureCache'];
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
    options: {
      signal?: AbortSignal;
      isCurrent?: () => boolean;
      scaleToImage?: boolean;
      validateDecoded?: (width: number, height: number) => void;
    }
  ) => Promise<DecodeImageResult>;
  readonly discardPersisted: (layerId: string) => void;
  readonly getDefaultControlModel: (base: string | null) => string | null;
  readonly getMainModelBase: () => string | null;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
}

/** Publishes guarded filter results as replacements or independent copies. */
export class FilterResultController {
  constructor(private readonly options: FilterResultControllerOptions) {}

  async commit(options: CommitRasterFilterOptions, owner?: symbol): Promise<CommitRasterFilterResult> {
    const o = this.options;
    const permit = o.ctx.capturePermit(owner);
    if (!permit) {
      return { status: 'busy' };
    }
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    try {
      const decoded = await o.decodeImage(options.image, {
        isCurrent: () => o.ctx.isPermitCurrent(permit),
        scaleToImage: false,
        signal: options.signal,
        validateDecoded: (width, height) => {
          if (
            options.requireExactImageDimensions &&
            (width !== options.image.width || height !== options.image.height)
          ) {
            throw new LayerFilterOutputDimensionError(
              options.filter?.type ?? 'decoded_filter',
              { height, width },
              { height: options.image.height, width: options.image.width, x: options.rect.x, y: options.rect.y }
            );
          }
        },
      });
      if (decoded.status !== 'ok') {
        return { status: decoded.status === 'aborted' ? 'aborted' : 'busy' };
      }
      const pixels = decoded.surface;
      const document = o.ctx.getDocument();
      const liveLayer = getDocumentLayer(document, options.guard.layerId);
      if (!document || !liveLayer) {
        return { status: 'missing' };
      }
      if (liveLayer.isLocked) {
        return { status: 'locked' };
      }
      if (liveLayer.type !== 'raster' && liveLayer.type !== 'control') {
        return { status: 'unsupported' };
      }
      if (!o.ctx.isPermitCurrent(permit) || o.ctx.isGestureActive()) {
        return { status: 'busy' };
      }
      if (!o.ctx.isGuardCurrent(options.guard)) {
        return { status: 'stale' };
      }
      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }
      const image = structuredClone(options.image);
      const rect = { ...options.rect };
      const paintSource = { bitmap: image, offset: { x: rect.x, y: rect.y }, type: 'paint' } as const;
      const afterPixels = { pixels, rect };
      const txn = o.ctx.begin({ historyBytes: rgbaBytes(rect) + HISTORY_ENTRY_OVERHEAD_BYTES, owner });
      if (!('publish' in txn)) {
        return { status: layerEditRefusal(txn.status) };
      }
      try {
        if (!txn.reserveRaster(rgbaBytes(rect))) {
          return { status: 'over-budget' };
        }
        if (options.mode === 'replace') {
          const beforePixels = o.captureCache(liveLayer, document, (rect) => txn.growHistory(rgbaBytes(rect)));
          if (beforePixels === 'not-ready' || beforePixels === 'over-budget') {
            return { status: beforePixels };
          }
          if (!beforePixels) {
            return { status: 'stale' };
          }
          const before = structuredClone(liveLayer);
          const after: CanvasLayerContract =
            liveLayer.type === 'raster'
              ? (() => {
                  const { adjustments: _adjustments, ...base } = liveLayer;
                  return structuredClone({ ...base, filter: options.filter, source: paintSource });
                })()
              : structuredClone({ ...liveLayer, filter: options.filter, source: paintSource });
          const discard = (): void => o.discardPersisted(liveLayer.id);
          const prepared = o.ctx.preparePixels(liveLayer.id, rect, pixels);
          const interrupted = this.interrupted(options);
          if (interrupted) {
            return interrupted;
          }
          const result = txn.publish(
            'Replace layer with filter result',
            replaceLayerStep(o.ctx, after, prepared, { notify: discard, persist: false, restore: liveLayer }),
            {
              bytes: rgbaBytes(beforePixels.rect) + rgbaBytes(rect) + HISTORY_ENTRY_OVERHEAD_BYTES,
              heldAssetRefs: collectHistoryMediaRefs(before, after),
              redo: () =>
                replayWithPixels(o.ctx, liveLayer.id, afterPixels, (prepared) =>
                  replaceLayerStep(o.ctx, after, prepared, { notify: discard, persist: false, restore: before })
                ),
              undo: () =>
                replayWithPixels(o.ctx, liveLayer.id, beforePixels, (prepared) =>
                  replaceLayerStep(o.ctx, before, prepared, {
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
        const selectedLayerId = document.selectedLayerId;
        const layerId = o.ctx.createLayerId();
        let copy: CanvasLayerContract;
        if (options.target === 'control') {
          const buildControlBase = (): CanvasControlLayerContract => {
            const mainBase = o.getMainModelBase();
            return createControlLayer(
              `${liveLayer.name} filtered`,
              layerId,
              mainBase,
              o.getDefaultControlModel(mainBase)
            );
          };
          const base = liveLayer.type === 'control' ? structuredClone(liveLayer) : buildControlBase();
          copy = {
            ...base,
            filter: options.filter,
            id: layerId,
            name: `${liveLayer.name} filtered`,
            source: paintSource,
            transform: structuredClone(liveLayer.transform),
          };
        } else if (options.target === 'raster' && liveLayer.type === 'control') {
          copy = {
            blendMode: liveLayer.blendMode,
            filter: options.filter,
            id: layerId,
            isEnabled: true,
            isLocked: false,
            name: `${liveLayer.name} filtered`,
            opacity: liveLayer.opacity,
            source: paintSource,
            transform: structuredClone(liveLayer.transform),
            type: 'raster',
          };
        } else {
          const { adjustments: _adjustments, ...base } = structuredClone(liveLayer as CanvasRasterLayerContractV2);
          copy = {
            ...base,
            filter: options.filter,
            id: layerId,
            name: `${liveLayer.name} filtered`,
            source: paintSource,
            type: 'raster',
          };
        }
        const anchor = o.ctx.captureInsertionAnchor(copy.type, liveLayer.id);
        const added = (prepared: PreparedLayerCacheReplacement | null) =>
          addLayerStep(o.ctx, copy, anchor, prepared, { persist: false, previousSelectedLayerId: selectedLayerId });
        const prepared = o.ctx.preparePixels(layerId, rect, pixels);
        const interrupted = this.interrupted(options);
        if (interrupted) {
          return interrupted;
        }
        const result = txn.publish('Copy layer filter result', added(prepared), {
          bytes: rgbaBytes(rect) + HISTORY_ENTRY_OVERHEAD_BYTES,
          heldAssetRefs: collectHistoryMediaRefs(copy),
          redo: () => replayWithPixels(o.ctx, layerId, afterPixels, added),
          undo: () => o.ctx.applyStep(removeLayerStep(layerId, selectedLayerId)),
        });
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
  private interrupted(options: CommitRasterFilterOptions): CommitRasterFilterResult | null {
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    return this.options.ctx.isGuardCurrent(options.guard) ? null : { status: 'stale' };
  }
}
