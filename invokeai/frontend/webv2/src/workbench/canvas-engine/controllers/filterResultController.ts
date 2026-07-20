import type { CommitRasterFilterOptions, CommitRasterFilterResult } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { CapturedLayerCache } from '@workbench/canvas-engine/controllers/layerMutationController';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { createControlLayer } from '@workbench/canvas-engine/document/layerFactories';
import { LayerFilterOutputDimensionError } from '@workbench/canvas-engine/filterError';

import type { CanvasMutationContext } from './mutationContext';

export type {
  CommitRasterFilterOptions,
  CommitRasterFilterResult,
  RasterFilterCommitTarget,
  RasterFilterSettings,
} from '@workbench/canvas-engine/capabilities';

export interface FilterResultControllerOptions {
  readonly captureCache: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => CapturedLayerCache;
  readonly ctx: CanvasMutationContext;
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
      const liveLayer = document?.layers.find((candidate) => candidate.id === options.guard.layerId);
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
      o.ctx.endBurst();
      const image = structuredClone(options.image);
      const rect = { ...options.rect };
      const paintSource = { bitmap: image, offset: { x: rect.x, y: rect.y }, type: 'paint' } as const;
      if (options.mode === 'replace') {
        const beforePixels = o.captureCache(liveLayer, document);
        if (!beforePixels || beforePixels === 'not-ready') {
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
        const afterPixels = { pixels, rect };
        const publish = (
          contract: CanvasLayerContract,
          prepared: PreparedLayerCacheReplacement,
          publishOptions: { discardPersistence: boolean; persist: boolean }
        ): void => {
          o.ctx.dispatchPrepared(
            { layer: contract, layerId: liveLayer.id, type: 'replaceCanvasLayer' },
            () => o.ctx.getReducerDocument()?.layers.find((candidate) => candidate.id === liveLayer.id) === contract,
            () => o.ctx.getDocument()?.layers.find((candidate) => candidate.id === liveLayer.id) === contract
          );
          if (publishOptions.discardPersistence) {
            try {
              o.discardPersisted(liveLayer.id);
            } catch {
              /* Ancillary after commit. */
            }
          }
          o.ctx.installPrepared(prepared, publishOptions.persist);
        };
        const apply = (
          contract: CanvasLayerContract,
          snapshot: { pixels: RasterSurface; rect: Rect },
          publishOptions: { discardPersistence: boolean; persist: boolean }
        ): void => publish(contract, o.ctx.preparePixels(liveLayer.id, snapshot.rect, snapshot.pixels), publishOptions);
        publish(after, o.ctx.preparePixels(liveLayer.id, rect, pixels), { discardPersistence: true, persist: false });
        o.ctx.history.push({
          bytes: beforePixels.rect.width * beforePixels.rect.height * 4 + rect.width * rect.height * 4 + 256,
          label: 'Replace layer with filter result',
          redo: () => apply(after, afterPixels, { discardPersistence: true, persist: false }),
          replayFailureAtomic: true,
          undo: () =>
            apply(before, beforePixels, { discardPersistence: false, persist: o.needsPixelPersistence(before) }),
        });
        return { layerId: liveLayer.id, status: 'committed' };
      }
      const sourceIndex = document.layers.findIndex((candidate) => candidate.id === liveLayer.id);
      if (sourceIndex < 0) {
        return { status: 'missing' };
      }
      const selectedLayerId = document.selectedLayerId;
      const layerId = o.ctx.createLayerId();
      let copy: CanvasLayerContract;
      if (options.target === 'control') {
        const base =
          liveLayer.type === 'control'
            ? structuredClone(liveLayer)
            : createControlLayer(`${liveLayer.name} filtered`, layerId, o.getMainModelBase());
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
      const apply = (): void => {
        const prepared = o.ctx.preparePixels(layerId, rect, pixels);
        o.ctx.dispatchPrepared(
          { index: sourceIndex, layer: copy, type: 'addCanvasLayer' },
          () => o.ctx.getReducerDocument()?.layers.some((candidate) => candidate === copy) === true,
          () => o.ctx.getDocument()?.layers.some((candidate) => candidate === copy) === true
        );
        o.ctx.installPrepared(prepared, false);
      };
      apply();
      o.ctx.history.push({
        bytes: rect.width * rect.height * 4 + 256,
        label: 'Copy layer filter result',
        redo: apply,
        replayFailureAtomic: true,
        undo: () => {
          o.ctx.dispatchPrepared(
            { id: selectedLayerId, type: 'setCanvasSelectedLayer' },
            () => o.ctx.getReducerDocument()?.selectedLayerId === selectedLayerId,
            () => o.ctx.getDocument()?.selectedLayerId === selectedLayerId
          );
          o.ctx.dispatchPrepared(
            { ids: [layerId], type: 'removeCanvasLayers' },
            () => o.ctx.getReducerDocument()?.layers.some((candidate) => candidate.id === layerId) === false,
            () => o.ctx.getDocument()?.layers.some((candidate) => candidate.id === layerId) === false
          );
        },
      });
      return { layerId, status: 'committed' };
    } catch (error) {
      if (options.signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        return { status: 'aborted' };
      }
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  }

  dispose(): void {}
}
