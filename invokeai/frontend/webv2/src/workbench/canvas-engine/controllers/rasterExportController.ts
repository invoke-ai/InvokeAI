import type {
  ExportBakedLayerBlobResult,
  ExportBakedLayerPixelsOptions,
  ExportLayerPixelsOptions,
  LayerExportGuard,
} from '@workbench/canvas-engine/capabilities';
import type {
  CanvasAdjustmentsContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { RasterReadLease } from '@workbench/canvas-engine/rasterTransactions';
import type { CanvasTextSource } from '@workbench/canvas-engine/render/fontLoader';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';

import { lookupDocumentLayer, lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { hasSpatialAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import {
  type AdjustmentRefusal,
  adjustmentRefusal,
  adjustSurfaceExactly,
  type SpatialRunner,
} from '@workbench/canvas-engine/render/exactAdjustments';

/**
 * `not-ready`: the pixels are still loading or moved while they were read, so trying again can succeed. `unreadable`:
 * rasterizing the layer's current source failed (an image that is gone or does not decode), so trying again cannot
 * succeed until the source changes.
 */
export type ExportLayerPixelsResult =
  | ({ status: 'ok' } & RasterReadLease)
  | {
      status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'unreadable' | 'over-budget' | 'aborted';
    }
  | AdjustmentRefusal;

export interface RasterExportControllerOptions {
  readonly backend: RasterBackend;
  readonly captureGuard: (layer: CanvasLayerContract, entry: LayerCacheEntry) => LayerExportGuard;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly getOrStartRasterization: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV3,
    signal?: AbortSignal
  ) => Promise<'published' | 'stale' | 'error' | 'aborted'>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly isRasterizing: (layer: CanvasLayerContract) => boolean;
  readonly isSupportedSource: (source: CanvasLayerSourceContract) => boolean;
  readonly layers: LayerCacheStore;
  readonly reserve?: (
    bytes: number
  ) =>
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
  /**
   * The display's current, exact adjusted copy of `surface` when it is the layer's live cache, or null; never builds
   * one. Bakes draw it instead of reserving and adjusting a layer-sized copy.
   */
  readonly peekAdjustedCacheSurface?: (
    layerId: string,
    surface: RasterSurface,
    adjustments: CanvasAdjustmentsContract
  ) => RasterSurface | null;
  /** Evaluates spatial adjustment runs; without it a spatial stack refuses `error`. */
  readonly runSpatial?: SpatialRunner;
  /** Keeps a layer resident and untrimmed for the lifetime of a read lease. */
  readonly pin: (layerId: string) => { release(): void };
  /** Resolves custom font bytes before an operation whose pixels leave the editor. */
  readonly waitForFont?: (source: CanvasTextSource, signal?: AbortSignal) => Promise<string>;
  /** Invalidates cached pixels rendered with a fallback family after output readiness recovers. */
  readonly invalidateLayerCache?: (layerId: string) => void;
}

interface ReservedExportLayerPixels {
  readonly result: ExportLayerPixelsResult;
  release(): void;
}

const noReservedPixels = (result: ExportLayerPixelsResult): ReservedExportLayerPixels => ({
  release: () => undefined,
  result,
});

/** Owns cache-backed, transformed, and encoded layer export primitives. */
export class RasterExportController {
  constructor(private readonly options: RasterExportControllerOptions) {}

  /**
   * An owned, layer-local adjusted copy of the leased pixels, releasing the lease. A spatial stack is evaluated
   * exactly (reusing the display's copy when it is exact for these pixels) or refused; its job yields, so the lease's
   * guard is checked again afterwards.
   */
  private async adjustLease(
    result: Extract<ExportLayerPixelsResult, { status: 'ok' }>,
    shouldApply: boolean,
    signal?: AbortSignal
  ): Promise<ExportLayerPixelsResult> {
    const layer = result.guard.layer;
    // Identity-aware: an emptied or all-disabled stack must not reserve or copy.
    if (!shouldApply || layer.type !== 'raster' || isIdentityAdjustments(layer.adjustments)) {
      return result;
    }
    const reservation = this.options.reserve?.(result.rect.width * result.rect.height * 8);
    if (reservation?.status === 'over-budget') {
      result.release();
      return { status: 'over-budget' };
    }
    let released = false;
    const release = (): void => {
      if (released) {
        return;
      }
      released = true;
      if (reservation?.status === 'ok') {
        reservation.lease.release();
      }
    };
    try {
      const { backend } = this.options;
      const displayed = hasSpatialAdjustments(layer.adjustments)
        ? (this.options.peekAdjustedCacheSurface?.(layer.id, result.surface, layer.adjustments!) ?? null)
        : null;
      let surface: RasterSurface;
      if (displayed) {
        // A copy: the display repaints its own surface in place when the stack changes.
        surface = backend.createSurface(result.rect.width, result.rect.height);
        surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
        surface.ctx.drawImage(displayed.canvas, 0, 0);
      } else {
        const adjusted = await adjustSurfaceExactly(
          (width, height) => backend.createSurface(width, height),
          result.surface,
          result.rect,
          layer.adjustments,
          { run: this.options.runSpatial, signal }
        );
        if (adjusted.status !== 'ok') {
          release();
          return adjusted.status === 'aborted'
            ? { status: 'aborted' }
            : adjustmentRefusal(this.options.getDocument(), layer.id, adjusted.status);
        }
        surface = adjusted.surface;
      }
      if (!this.options.isGuardCurrent(result.guard)) {
        release();
        return { status: 'not-ready' };
      }
      return { ...result, release, surface };
    } catch (error) {
      release();
      throw error;
    } finally {
      // The owned copy no longer reads the live cache; only its guard still describes it.
      result.release();
    }
  }

  async rasterize(layerId: string, options: ExportLayerPixelsOptions = {}): Promise<ExportLayerPixelsResult> {
    const document = this.options.getDocument();
    if (!document) {
      return { status: 'missing' };
    }
    const layer = lookupDocumentLayer(document, layerId);
    const source = layer ? renderableSourceOf(layer) : null;
    if (!layer || !source) {
      return { status: 'missing' };
    }
    if (!options.includeDisabled && !lookupDocumentLeaf(document, layerId)?.contributionEnabled) {
      return { status: 'disabled' };
    }
    if (!this.options.isSupportedSource(source)) {
      return { status: 'unsupported' };
    }
    if (source.type === 'text' && this.options.waitForFont) {
      let readyFontFamily: string;
      try {
        readyFontFamily = await this.options.waitForFont(source, options.signal);
      } catch {
        return { status: options.signal?.aborted ? 'aborted' : 'not-ready' };
      }
      const cachedText = this.options.layers.peek(layerId);
      if (
        source.fontRef &&
        cachedText &&
        !cachedText.stale &&
        !isEmpty(cachedText.rect) &&
        cachedText.renderedFontFamily !== readyFontFamily
      ) {
        (this.options.invalidateLayerCache ?? this.options.layers.invalidate)(layerId);
      }
    }
    const liveEntry = this.options.layers.get(layerId);
    if (liveEntry && !liveEntry.stale && !this.options.isRasterizing(layer) && !isEmpty(liveEntry.rect)) {
      return this.adjustLease(this.lease(layer, liveEntry), options.applyAdjustments === true, options.signal);
    }
    const contentRect = getSourceContentRect(layer, document);
    if (isEmpty(contentRect)) {
      return { status: 'empty' };
    }
    const reservation = this.options.reserve?.(contentRect.width * contentRect.height * 8);
    if (reservation?.status === 'over-budget') {
      return { status: 'over-budget' };
    }
    // The pin outlives the rasterization: the lease keeps the published pixels resident until its release.
    const pin = this.options.pin(layerId);
    let leased = false;
    try {
      const rasterized = await this.options.getOrStartRasterization(layer, document, options.signal);
      if (rasterized !== 'published') {
        return { status: rasterized === 'aborted' ? 'aborted' : rasterized === 'error' ? 'unreadable' : 'not-ready' };
      }
      const currentDocument = this.options.getDocument();
      const currentLayer = currentDocument ? lookupDocumentLayer(currentDocument, layerId) : null;
      const entry = this.options.layers.get(layerId);
      if (!currentLayer || !entry || entry.stale) {
        return { status: 'not-ready' };
      }
      const currentSource = renderableSourceOf(currentLayer);
      if (!currentSource) {
        return { status: 'missing' };
      }
      if (!options.includeDisabled && !lookupDocumentLeaf(currentDocument!, layerId)?.contributionEnabled) {
        return { status: 'disabled' };
      }
      if (!this.options.isSupportedSource(currentSource)) {
        return { status: 'unsupported' };
      }
      if (isEmpty(entry.rect)) {
        return { status: 'empty' };
      }
      const lease = this.lease(currentLayer, entry, pin);
      leased = true;
      return this.adjustLease(lease, options.applyAdjustments === true, options.signal);
    } finally {
      if (!leased) {
        pin.release();
      }
      if (reservation?.status === 'ok') {
        reservation.lease.release();
      }
    }
  }

  /** A lease on the live cache; the pin is taken last so nothing can fail while it is unowned. */
  private lease(
    layer: CanvasLayerContract,
    entry: LayerCacheEntry,
    pin?: { release(): void }
  ): Extract<ExportLayerPixelsResult, { status: 'ok' }> {
    const guard = this.options.captureGuard(layer, entry);
    const held = pin ?? this.options.pin(layer.id);
    return { guard, rect: { ...entry.rect }, release: () => held.release(), status: 'ok', surface: entry.surface };
  }

  private async reserveBaked(
    layerId: string,
    options: ExportBakedLayerPixelsOptions = {}
  ): Promise<ReservedExportLayerPixels> {
    const leased = await this.rasterize(layerId, { ...options, applyAdjustments: false });
    if (leased.status !== 'ok') {
      return noReservedPixels(leased);
    }
    let raw = leased;
    const layer = raw.guard.layer;
    const matrix = fromTRS(
      { x: layer.transform.x, y: layer.transform.y },
      layer.transform.rotation,
      layer.transform.scaleX,
      layer.transform.scaleY
    );
    const rect = roundOut(transformBounds(matrix, raw.rect));
    if (isEmpty(rect)) {
      raw.release();
      return noReservedPixels({ status: 'empty' });
    }
    // The layer's own stack applies to layer-local pixels before the transform resamples them, as the display does,
    // so bakes keep the displayed grain, detail and color for scaled and rotated layers too. The display's exact
    // adjusted copy of the leased live cache is drawn as is (before anything yields); otherwise a layer-local copy is
    // reserved and adjusted, spatial entries included, or the bake refuses.
    if (options.applyAdjustments !== false && layer.type === 'raster' && !isIdentityAdjustments(layer.adjustments)) {
      const displayed = this.options.peekAdjustedCacheSurface?.(layer.id, raw.surface, layer.adjustments!) ?? null;
      const adjusted = displayed ? { ...raw, surface: displayed } : await this.adjustLease(raw, true, options.signal);
      if (adjusted.status !== 'ok') {
        return noReservedPixels(adjusted);
      }
      raw = adjusted;
    }
    const reservation = this.options.reserve?.(rect.width * rect.height * 4);
    if (reservation?.status === 'over-budget') {
      raw.release();
      return noReservedPixels({ status: 'over-budget' });
    }
    let released = false;
    const release = (): void => {
      if (released) {
        return;
      }
      released = true;
      if (reservation?.status === 'ok') {
        reservation.lease.release();
      }
    };
    try {
      const surface = this.options.backend.createSurface(rect.width, rect.height);
      const ctx = surface.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - rect.x, matrix.f - rect.y);
      ctx.drawImage(raw.surface.canvas, raw.rect.x, raw.rect.y);
      return { release, result: { guard: raw.guard, rect, release, status: 'ok', surface } };
    } catch (error) {
      release();
      throw error;
    } finally {
      raw.release();
    }
  }

  async baked(layerId: string, options: ExportBakedLayerPixelsOptions = {}): Promise<ExportLayerPixelsResult> {
    const reserved = await this.reserveBaked(layerId, options);
    return reserved.result;
  }

  async blob(layerId: string, options: ExportBakedLayerPixelsOptions = {}): Promise<ExportBakedLayerBlobResult> {
    const reserved = await this.reserveBaked(layerId, options);
    try {
      const result = reserved.result;
      if (result.status !== 'ok') {
        return result;
      }
      const blob = await this.options.backend.encodeSurface(result.surface, 'image/png');
      if (!this.options.isGuardCurrent(result.guard)) {
        return { status: 'not-ready' };
      }
      return { blob, guard: result.guard, rect: result.rect, status: 'ok' };
    } finally {
      reserved.release();
    }
  }

  dispose(): void {}
}
