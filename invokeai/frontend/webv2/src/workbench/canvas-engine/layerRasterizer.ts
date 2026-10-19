import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { RasterizationJob } from '@workbench/canvas-engine/controllers/rasterController';
import type { FontLoader } from '@workbench/canvas-engine/render/fontLoader';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { RasterizeResult } from '@workbench/canvas-engine/render/rasterizers';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { isSupportedExportSource } from '@workbench/canvas-engine/layerExportGuards';

/** How a rasterization ended: pixels landed, the world moved, it threw, or it was cancelled. */
export type LayerRasterizationOutcome = 'published' | 'stale' | 'error' | 'aborted';

/** The registry of in-flight rasterizations, keyed by layer. */
export interface RasterizationJobRegistry {
  getDocumentGeneration(): number;
  get(layerId: string): RasterizationJob | undefined;
  install(layerId: string, job: RasterizationJob): void;
  finish(layerId: string, job: RasterizationJob): void;
  cancel(layerId: string): void;
}

export interface CreateLayerRasterizerDeps {
  readonly layerCache: LayerCacheStore;
  readonly jobs: RasterizationJobRegistry;
  readonly thumbnails: {
    setVersion(layerId: string, version: number): void;
    setStatus(layerId: string, status: 'ready' | 'error'): void;
  };
  readonly fontLoader: Pick<FontLoader, 'ensure' | 'resolveFamily'>;
  readonly createSurface: (width: number, height: number) => RasterSurface;
  readonly rasterize: (
    source: CanvasLayerSourceContract,
    document: CanvasDocumentContractV3,
    scratch: RasterSurface,
    signal: AbortSignal
  ) => Promise<RasterizeResult>;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly hasCanvasState: () => boolean;
  readonly isDisposed: () => boolean;
  readonly invalidateLayerCache: (layerId: string) => void;
  readonly invalidateLayerRender: (layerId: string) => void;
  readonly reportError: (message: 'Layer thumbnail rasterization failed', layerId: string, error: unknown) => void;
}

export interface LayerRasterizer {
  /**
   * Starts or joins rasterization for the same source, cache version and document generation; repeated frame
   * requests do not restart it.
   */
  getOrStartLayerRasterization(
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV3,
    signal?: AbortSignal
  ): Promise<LayerRasterizationOutcome>;
}

/**
 * Rasterize into scratch, publishing only if source value, cache version, document generation, installed job and
 * lifecycle still match. Post-success also checks cache-entry identity to distinguish recreation at version zero;
 * font callbacks run before job installation. Text renders available fonts immediately and invalidates when the
 * real face loads.
 */
export const createLayerRasterizer = (deps: CreateLayerRasterizerDeps): LayerRasterizer => {
  const { jobs, layerCache } = deps;

  const getOrStartLayerRasterization = (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV3,
    signal?: AbortSignal
  ): Promise<LayerRasterizationOutcome> => {
    if (signal?.aborted) {
      return Promise.resolve('aborted');
    }
    if (deps.isDisposed() || !deps.hasCanvasState()) {
      return Promise.resolve('stale');
    }
    const liveSource = renderableSourceOf(layer);
    if (!liveSource || !isSupportedExportSource(liveSource)) {
      return Promise.resolve('stale');
    }

    const contentRect = getSourceContentRect(layer, document);
    const entry = layerCache.getOrCreateRect(layer.id, contentRect);
    const version = entry.version;
    const documentGeneration = jobs.getDocumentGeneration();
    const source = structuredClone(liveSource);
    const existing = jobs.get(layer.id);
    if (
      existing &&
      existing.version === version &&
      existing.documentGeneration === documentGeneration &&
      areJsonValuesStructurallyEqual(existing.source, source)
    ) {
      if (!signal) {
        return existing.promise;
      }
      // Joining with a signal must be able to cancel the shared job, but the
      // listener has to come off again or a long-lived signal accumulates them.
      const abort = (): void => {
        existing.abortedByCaller = true;
        existing.controller.abort(signal.reason);
      };
      signal.addEventListener('abort', abort, { once: true });
      return existing.promise.finally(() => signal.removeEventListener('abort', abort));
    }
    jobs.cancel(layer.id);

    const controller = new AbortController();
    const renderedFontFamily = source.type === 'text' ? deps.fontLoader.resolveFamily(source) : undefined;
    if (source.type === 'text') {
      deps.fontLoader.ensure(
        source,
        () => {
          const currentLayer = getDocumentLayer(deps.getDocument(), layer.id);
          if (
            deps.isDisposed() ||
            !deps.hasCanvasState() ||
            !currentLayer ||
            jobs.getDocumentGeneration() !== documentGeneration ||
            layerCache.version(layer.id) !== version ||
            !areJsonValuesStructurallyEqual(renderableSourceOf(currentLayer), source)
          ) {
            return;
          }
          deps.invalidateLayerCache(layer.id);
          deps.invalidateLayerRender(layer.id);
        },
        controller.signal
      );
    }

    const scratch = deps.createSurface(contentRect.width, contentRect.height);
    let settleJob!: (result: LayerRasterizationOutcome) => void;
    const promise = new Promise<LayerRasterizationOutcome>((resolve) => {
      settleJob = resolve;
    });
    const job: RasterizationJob = { controller, documentGeneration, promise, source, version };
    const abort = (): void => {
      job.abortedByCaller = true;
      controller.abort(signal?.reason);
    };
    signal?.addEventListener('abort', abort, { once: true });
    jobs.install(layer.id, job);
    void (async () => {
      try {
        const result = await deps.rasterize(source, document, scratch, controller.signal);
        const currentLayer = getDocumentLayer(deps.getDocument(), layer.id);
        const currentEntry = layerCache.get(layer.id);
        if (
          deps.isDisposed() ||
          !deps.hasCanvasState() ||
          jobs.get(layer.id) !== job ||
          jobs.getDocumentGeneration() !== documentGeneration ||
          !currentLayer ||
          !currentEntry ||
          currentEntry.version !== version ||
          !areJsonValuesStructurallyEqual(renderableSourceOf(currentLayer), source)
        ) {
          return 'stale';
        }

        const publishedEntry = layerCache.publishRasterized(layer.id, result.rect, result.surface, renderedFontFamily);
        deps.thumbnails.setVersion(layer.id, publishedEntry.version);
        deps.thumbnails.setStatus(layer.id, 'ready');
        deps.invalidateLayerRender(layer.id);
        return 'published';
      } catch (error) {
        if (job.abortedByCaller) {
          return 'aborted';
        }
        const currentLayer = getDocumentLayer(deps.getDocument(), layer.id);
        if (
          deps.isDisposed() ||
          !deps.hasCanvasState() ||
          jobs.get(layer.id) !== job ||
          jobs.getDocumentGeneration() !== documentGeneration ||
          !currentLayer ||
          layerCache.version(layer.id) !== version ||
          !areJsonValuesStructurallyEqual(renderableSourceOf(currentLayer), source)
        ) {
          // Discard failures from obsolete work rather than reporting errors for replaced layers.
          return 'stale';
        }
        deps.thumbnails.setStatus(layer.id, 'error');
        try {
          deps.reportError('Layer thumbnail rasterization failed', layer.id, error);
        } catch {
          // Diagnostics must not turn a handled thumbnail failure into a rejection.
        }
        return 'error';
      } finally {
        signal?.removeEventListener('abort', abort);
        jobs.finish(layer.id, job);
      }
    })().then(settleJob, () => settleJob('stale'));
    return promise;
  };

  return { getOrStartLayerRasterization };
};
