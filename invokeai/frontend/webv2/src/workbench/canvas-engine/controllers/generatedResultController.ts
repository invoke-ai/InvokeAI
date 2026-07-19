import type { CommitGeneratedImageOptions, CommitGeneratedImageResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2, CanvasImageRef, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CapturedLayerCache } from '@workbench/canvas-engine/controllers/layerMutationController';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect } from '@workbench/canvas-engine/types';

import { createControlLayer, nextControlLayerName } from '@workbench/canvas-engine/document/layerFactories';

import type { CanvasMutationContext } from './mutationContext';

export interface GeneratedResultControllerOptions {
  readonly captureCache: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => CapturedLayerCache;
  readonly clearPreview: (layerId: string) => void;
  readonly ctx: CanvasMutationContext;
  readonly decodeImage: (
    image: CanvasImageRef,
    options: { signal?: AbortSignal; isCurrent?: () => boolean }
  ) => Promise<DecodeImageResult>;
  readonly discardPersisted: (layerId: string) => void;
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
      | { document: CanvasDocumentContractV2; liveLayer: Extract<CanvasLayerContract, { type: 'raster' | 'control' }> }
      | { result: CommitGeneratedImageResult } => {
      if (!o.ctx.isPermitCurrent(permit)) {
        return { result: { status: 'busy' } };
      }
      const document = o.ctx.getDocument();
      if (!document) {
        return { result: { status: 'missing' } };
      }
      const liveLayer = document.layers.find((layer) => layer.id === options.guard.layerId);
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
      const publishSnapshot = (
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
            /* Ancillary after reducer acceptance. */
          }
        }
        try {
          o.clearPreview(liveLayer.id);
        } catch {
          /* Transient preview is ancillary. */
        }
        o.ctx.installPrepared(prepared, publishOptions.persist);
      };
      const applySnapshot = (
        contract: CanvasLayerContract,
        snapshot: { pixels: RasterSurface; rect: Rect },
        publishOptions: { discardPersistence: boolean; persist: boolean }
      ): void =>
        publishSnapshot(contract, o.ctx.preparePixels(liveLayer.id, snapshot.rect, snapshot.pixels), publishOptions);
      if (options.target === 'replace') {
        const beforePixels = o.captureCache(liveLayer, document);
        if (!beforePixels || beforePixels === 'not-ready') {
          return { status: 'stale' };
        }
        if (options.signal?.aborted) {
          return { status: 'aborted' };
        }
        const before = structuredClone(liveLayer);
        let after: CanvasLayerContract;
        if (liveLayer.type === 'raster') {
          const { adjustments: _adjustments, ...base } = structuredClone(liveLayer);
          after = { ...base, source, transform: identityTransform };
        } else {
          after = { ...structuredClone(liveLayer), source, transform: identityTransform };
        }
        const afterPixels = { pixels: decoded.surface, rect };
        const prepared = o.ctx.preparePixels(liveLayer.id, rect, decoded.surface);
        if (options.signal?.aborted) {
          return { status: 'aborted' };
        }
        const finalCheck = validate();
        if ('result' in finalCheck) {
          return finalCheck.result;
        }
        o.ctx.endBurst();
        publishSnapshot(after, prepared, { discardPersistence: true, persist: false });
        o.ctx.history.push({
          bytes: beforePixels.rect.width * beforePixels.rect.height * 4 + rect.width * rect.height * 4 + 256,
          label: options.historyLabel ?? 'Replace layer with workflow result',
          redo: () => applySnapshot(after, afterPixels, { discardPersistence: true, persist: false }),
          replayFailureAtomic: true,
          undo: () =>
            applySnapshot(before, beforePixels, {
              discardPersistence: false,
              persist: o.needsPixelPersistence(before),
            }),
        });
        return { layerId: liveLayer.id, status: 'committed' };
      }
      const sourceIndex = document.layers.findIndex((candidate) => candidate.id === liveLayer.id);
      if (sourceIndex < 0) {
        return { status: 'missing' };
      }
      const layerId = o.ctx.createLayerId();
      const selectedLayerId = document.selectedLayerId;
      const copy: CanvasLayerContract =
        options.target === 'copy-control'
          ? {
              ...createControlLayer(
                options.copyLayerName ?? nextControlLayerName(document.layers.map((layer) => layer.name)),
                layerId,
                o.getMainModelBase()
              ),
              source,
              transform: identityTransform,
            }
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
      const publishCopy = (prepared: PreparedLayerCacheReplacement): void => {
        o.ctx.dispatchPrepared(
          { index: sourceIndex, layer: copy, type: 'addCanvasLayer' },
          () => o.ctx.getReducerDocument()?.layers.some((candidate) => candidate === copy) === true,
          () => o.ctx.getDocument()?.layers.some((candidate) => candidate === copy) === true
        );
        o.ctx.installPrepared(prepared, false);
      };
      const applyCopy = (): void => publishCopy(o.ctx.preparePixels(layerId, rect, decoded.surface));
      const prepared = o.ctx.preparePixels(layerId, rect, decoded.surface);
      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }
      const finalCheck = validate();
      if ('result' in finalCheck) {
        return finalCheck.result;
      }
      o.ctx.endBurst();
      publishCopy(prepared);
      o.ctx.history.push({
        bytes: rect.width * rect.height * 4 + 256,
        label:
          options.target === 'copy-control'
            ? 'Copy workflow result to control layer'
            : 'Copy workflow result to raster layer',
        redo: applyCopy,
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
