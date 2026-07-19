import type {
  CommitRasterFilterOptions,
  CommitRasterFilterResult,
  LayerExportGuard,
} from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { CapturedLayerCache } from '@workbench/canvas-engine/controllers/layerMutationController';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { History } from '@workbench/canvas-engine/history/history';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createControlLayer } from '@workbench/canvas-engine/document/layerFactories';
import { LayerFilterOutputDimensionError } from '@workbench/canvas-engine/filterError';

export type {
  CommitRasterFilterOptions,
  CommitRasterFilterResult,
  RasterFilterCommitTarget,
  RasterFilterSettings,
} from '@workbench/canvas-engine/capabilities';

export interface FilterResultControllerOptions<Permit, Owner = symbol> {
  readonly captureCache: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => CapturedLayerCache;
  readonly capturePermit: (owner?: Owner) => Permit | null;
  readonly createLayerId: () => string;
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
  readonly dispatchPrepared: (
    action: CanvasProjectMutation,
    reducerAccepted: () => boolean,
    mirrorAccepted: () => boolean
  ) => void;
  readonly endBurst: () => void;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getMainModelBase: () => string | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly history: History;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement, persist?: boolean) => void;
  readonly isGestureActive: () => boolean;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly isPermitCurrent: (permit: Permit) => boolean;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => PreparedLayerCacheReplacement;
}

/** Publishes guarded filter results as replacements or independent copies. */
export class FilterResultController<Permit, Owner = symbol> {
  constructor(private readonly options: FilterResultControllerOptions<Permit, Owner>) {}

  async commit(options: CommitRasterFilterOptions, owner?: Owner): Promise<CommitRasterFilterResult> {
    const o = this.options;
    const permit = o.capturePermit(owner);
    if (!permit) {
      return { status: 'busy' };
    }
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    try {
      const decoded = await o.decodeImage(options.image, {
        isCurrent: () => o.isPermitCurrent(permit),
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
      const document = o.getDocument();
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
      if (!o.isPermitCurrent(permit) || o.isGestureActive()) {
        return { status: 'busy' };
      }
      if (!o.isGuardCurrent(options.guard)) {
        return { status: 'stale' };
      }
      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }
      o.endBurst();
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
          o.dispatchPrepared(
            { layer: contract, layerId: liveLayer.id, type: 'replaceCanvasLayer' },
            () => o.getReducerDocument()?.layers.find((candidate) => candidate.id === liveLayer.id) === contract,
            () => o.getDocument()?.layers.find((candidate) => candidate.id === liveLayer.id) === contract
          );
          if (publishOptions.discardPersistence) {
            try {
              o.discardPersisted(liveLayer.id);
            } catch {
              /* Ancillary after commit. */
            }
          }
          o.installPrepared(prepared, publishOptions.persist);
        };
        const apply = (
          contract: CanvasLayerContract,
          snapshot: { pixels: RasterSurface; rect: Rect },
          publishOptions: { discardPersistence: boolean; persist: boolean }
        ): void => publish(contract, o.preparePixels(liveLayer.id, snapshot.rect, snapshot.pixels), publishOptions);
        publish(after, o.preparePixels(liveLayer.id, rect, pixels), { discardPersistence: true, persist: false });
        o.history.push({
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
      const layerId = o.createLayerId();
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
        const prepared = o.preparePixels(layerId, rect, pixels);
        o.dispatchPrepared(
          { index: sourceIndex, layer: copy, type: 'addCanvasLayer' },
          () => o.getReducerDocument()?.layers.some((candidate) => candidate === copy) === true,
          () => o.getDocument()?.layers.some((candidate) => candidate === copy) === true
        );
        o.installPrepared(prepared, false);
      };
      apply();
      o.history.push({
        bytes: rect.width * rect.height * 4 + 256,
        label: 'Copy layer filter result',
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
