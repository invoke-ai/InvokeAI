import type {
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { RasterizationJob } from '@workbench/canvas-engine/controllers/rasterController';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { RasterizeResult } from '@workbench/canvas-engine/render/rasterizers';

import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { isDeeplyEqual, isSupportedExportSource } from '@workbench/canvas-engine/layerExportGuards';
import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { textFontString } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';

/** How a rasterization ended: pixels landed, the world moved, it threw, or it was cancelled. */
export type LayerRasterizationOutcome = 'published' | 'stale' | 'error' | 'aborted';

/** The image a source references, whose decoded bitmap the job holds a reference to. */
const sourceImageName = (source: CanvasLayerSourceContract): string | null => {
  if (source.type === 'image') {
    return source.image.imageName;
  }
  if (source.type === 'paint') {
    return source.bitmap?.imageName ?? null;
  }
  return null;
};

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
  readonly fontLoader: { ensure(font: string, onReady: () => void): void };
  readonly createSurface: (width: number, height: number) => RasterSurface;
  readonly rasterize: (
    source: CanvasLayerSourceContract,
    document: CanvasDocumentContractV2,
    scratch: RasterSurface,
    signal: AbortSignal
  ) => Promise<RasterizeResult>;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly hasCanvasState: () => boolean;
  readonly isDisposed: () => boolean;
  readonly invalidateLayerCache: (layerId: string) => void;
  readonly invalidateLayerRender: (layerId: string) => void;
  readonly trackPublishedLayerImage: (layer: CanvasLayerContract) => void;
  readonly releaseBitmapIfUnreferenced: (imageName: string) => void;
  readonly reportError: (message: 'Layer thumbnail rasterization failed', layerId: string, error: unknown) => void;
}

export interface LayerRasterizer {
  /**
   * Starts — or joins — the rasterization of one layer, resolving with how it
   * ended. Safe to call every frame: a job that already describes the same
   * source, cache version and document generation is shared, not restarted.
   */
  getOrStartLayerRasterization(
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV2,
    signal?: AbortSignal
  ): Promise<LayerRasterizationOutcome>;
}

/**
 * Rasterizes a layer's source into its cache.
 *
 * Rasterizing is slow and the document is live, so a job never draws straight
 * into the cache: pixels land in a scratch surface, and are copied across only
 * if the job still describes the world it started in — same source (compared by
 * value, since a reducer pass replaces objects wholesale), same cache version,
 * same document generation, same job still installed, engine not disposed. Any
 * of those moving means the result describes a layer that no longer exists in
 * that form, and it is dropped as `'stale'` rather than published over whatever
 * replaced it.
 *
 * The same check appears three times and is deliberately not shared. The
 * post-success one holds the cache entry it is about to write to, so it can
 * distinguish a deleted cache from a fresh one at version 0 — the two are
 * indistinguishable through `version()` alone. The font-load one runs before
 * any job is installed, so it has no job identity to compare.
 *
 * Text is the one source that can rasterize correctly and still be wrong: the
 * font may not have loaded yet. Rather than block, it rasterizes with whatever
 * is available and re-invalidates once the real font arrives.
 */
export const createLayerRasterizer = (deps: CreateLayerRasterizerDeps): LayerRasterizer => {
  const { jobs, layerCache } = deps;

  const getOrStartLayerRasterization = (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV2,
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
      isDeeplyEqual(existing.source, source)
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

    if (source.type === 'text') {
      deps.fontLoader.ensure(textFontString(source), () => {
        const currentLayer = deps.getDocument()?.layers.find((candidate) => candidate.id === layer.id);
        if (
          deps.isDisposed() ||
          !deps.hasCanvasState() ||
          !currentLayer ||
          jobs.getDocumentGeneration() !== documentGeneration ||
          layerCache.version(layer.id) !== version ||
          !isDeeplyEqual(renderableSourceOf(currentLayer), source)
        ) {
          return;
        }
        deps.invalidateLayerCache(layer.id);
        deps.invalidateLayerRender(layer.id);
      });
    }

    const scratch = deps.createSurface(contentRect.width, contentRect.height);
    const controller = new AbortController();
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
    let published = false;
    void (async () => {
      try {
        const result = await deps.rasterize(source, document, scratch, controller.signal);
        const currentLayer = deps.getDocument()?.layers.find((candidate) => candidate.id === layer.id);
        const currentEntry = layerCache.get(layer.id);
        if (
          deps.isDisposed() ||
          !deps.hasCanvasState() ||
          jobs.get(layer.id) !== job ||
          jobs.getDocumentGeneration() !== documentGeneration ||
          !currentLayer ||
          !currentEntry ||
          currentEntry.version !== version ||
          !isDeeplyEqual(renderableSourceOf(currentLayer), source)
        ) {
          return 'stale';
        }

        if (currentEntry.surface.width !== result.rect.width || currentEntry.surface.height !== result.rect.height) {
          currentEntry.surface.resize(result.rect.width, result.rect.height);
        }
        const ctx = currentEntry.surface.ctx;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, result.rect.width, result.rect.height);
        if (!isEmpty(result.rect)) {
          ctx.drawImage(result.surface.canvas, 0, 0);
        }
        currentEntry.rect = { ...result.rect };
        const publishedEntry = layerCache.publishPixels(layer.id);
        if (!publishedEntry) {
          return 'stale';
        }
        deps.trackPublishedLayerImage(currentLayer);
        published = true;
        deps.thumbnails.setVersion(layer.id, publishedEntry.version);
        deps.thumbnails.setStatus(layer.id, 'ready');
        deps.invalidateLayerRender(layer.id);
        return 'published';
      } catch (error) {
        if (job.abortedByCaller) {
          return 'aborted';
        }
        const currentLayer = deps.getDocument()?.layers.find((candidate) => candidate.id === layer.id);
        if (
          deps.isDisposed() ||
          !deps.hasCanvasState() ||
          jobs.get(layer.id) !== job ||
          jobs.getDocumentGeneration() !== documentGeneration ||
          !currentLayer ||
          layerCache.version(layer.id) !== version ||
          !isDeeplyEqual(renderableSourceOf(currentLayer), source)
        ) {
          // The failure describes a layer that has already moved on; reporting it
          // would surface an error about work nobody is waiting for.
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
        // A job that did not publish still holds the decoded bitmap it pulled in.
        const imageName = sourceImageName(source);
        if (!published && imageName) {
          deps.releaseBitmapIfUnreferenced(imageName);
        }
      }
    })().then(settleJob, () => settleJob('stale'));
    return promise;
  };

  return { getOrStartLayerRasterization };
};
