import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { RasterizationJob } from '@workbench/canvas-engine/controllers/rasterController';
import type { FontLoader } from '@workbench/canvas-engine/render/fontLoader';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
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
  /** Runs `callback` after `delayMs`, returning a cancel; defaults to the global timer. */
  readonly setTimer?: (callback: () => void, delayMs: number) => () => void;
}

/** The first wait before a failed source may rasterize again for display, doubling per repeated failure up to the cap. */
export const RASTERIZATION_RETRY_BASE_MS = 3_000;
export const RASTERIZATION_RETRY_MAX_MS = 60_000;

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
  /**
   * Whether the last rasterization of `layer`'s current source, cache version and document generation failed, so its
   * stale cache cannot become ready by waiting. A success or any change to the layer clears it.
   */
  hasFailed(layer: CanvasLayerContract): boolean;
  /**
   * Whether {@link hasFailed} and the failure's retry backoff is still running. The backoff starts at
   * {@link RASTERIZATION_RETRY_BASE_MS} and doubles per repeated failure up to {@link RASTERIZATION_RETRY_MAX_MS}; its
   * end invalidates the layer's render, so a displayed layer retries on its own.
   */
  isBackingOff(layer: CanvasLayerContract): boolean;
  /** Cancels pending retry timers. */
  dispose(): void;
}

/** What a failed rasterization read; a later failure of the same key is the same failure. */
interface RasterizationFailure {
  readonly version: number;
  readonly documentGeneration: number;
  readonly source: CanvasLayerSourceContract;
  /** Consecutive failures of this key. */
  readonly attempts: number;
  /** Ends the backoff early; null once it ended. */
  cancelRetry: (() => void) | null;
}

/**
 * Rasterize into scratch, publishing only if source value, cache version, document generation, installed job and
 * lifecycle still match. Post-success also checks cache-entry identity to distinguish recreation at version zero;
 * font callbacks run before job installation. Text renders available fonts immediately and invalidates when the
 * real face loads.
 */
export const createLayerRasterizer = (deps: CreateLayerRasterizerDeps): LayerRasterizer => {
  const { jobs, layerCache } = deps;
  const setTimer =
    deps.setTimer ??
    ((callback: () => void, delayMs: number) => {
      const handle = setTimeout(callback, delayMs);
      return () => clearTimeout(handle);
    });
  // Keyed by cache entry so an evicted or deleted layer's record goes with it.
  const failures = new WeakMap<LayerCacheEntry, RasterizationFailure>();
  const backoffs = new Set<RasterizationFailure>();
  const endBackoff = (failure: RasterizationFailure | undefined): void => {
    failure?.cancelRetry?.();
    if (failure) {
      failure.cancelRetry = null;
      backoffs.delete(failure);
    }
  };
  const isSameFailure = (
    failure: RasterizationFailure | undefined,
    version: number,
    documentGeneration: number,
    source: CanvasLayerSourceContract | null
  ): boolean =>
    !!failure &&
    failure.version === version &&
    failure.documentGeneration === documentGeneration &&
    areJsonValuesStructurallyEqual(failure.source, source);

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

        endBackoff(failures.get(currentEntry));
        failures.delete(currentEntry);
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
        // Retrying the same source fails the same way; only its first failure is worth a notice.
        const failedEntry = layerCache.peek(layer.id);
        const previous = failedEntry ? failures.get(failedEntry) : undefined;
        const repeated = !!failedEntry && isSameFailure(previous, version, documentGeneration, source);
        endBackoff(previous);
        if (failedEntry && !deps.isDisposed()) {
          const attempts = repeated ? previous!.attempts + 1 : 1;
          const failure: RasterizationFailure = { attempts, cancelRetry: null, documentGeneration, source, version };
          failure.cancelRetry = setTimer(
            () => {
              failure.cancelRetry = null;
              backoffs.delete(failure);
              deps.invalidateLayerRender(layer.id);
            },
            Math.min(RASTERIZATION_RETRY_BASE_MS * 2 ** (attempts - 1), RASTERIZATION_RETRY_MAX_MS)
          );
          backoffs.add(failure);
          failures.set(failedEntry, failure);
        }
        if (!repeated) {
          try {
            deps.reportError('Layer thumbnail rasterization failed', layer.id, error);
          } catch {
            // Diagnostics must not turn a handled thumbnail failure into a rejection.
          }
        }
        return 'error';
      } finally {
        signal?.removeEventListener('abort', abort);
        jobs.finish(layer.id, job);
      }
    })().then(settleJob, () => settleJob('stale'));
    return promise;
  };

  /** The failure recorded for `layer`'s current key, while its cache still waits for pixels. */
  const currentFailure = (layer: CanvasLayerContract): RasterizationFailure | undefined => {
    const entry = layerCache.peek(layer.id);
    const failure = entry ? failures.get(entry) : undefined;
    return entry?.stale &&
      isSameFailure(failure, entry.version, jobs.getDocumentGeneration(), renderableSourceOf(layer))
      ? failure
      : undefined;
  };
  const hasFailed = (layer: CanvasLayerContract): boolean => !!currentFailure(layer);
  const isBackingOff = (layer: CanvasLayerContract): boolean => !!currentFailure(layer)?.cancelRetry;

  const dispose = (): void => {
    for (const failure of backoffs) {
      endBackoff(failure);
    }
  };

  return { dispose, getOrStartLayerRasterization, hasFailed, isBackingOff };
};
