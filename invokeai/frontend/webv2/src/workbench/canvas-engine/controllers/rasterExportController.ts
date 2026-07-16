import type {
  ExportBakedLayerBlobResult,
  ExportBakedLayerPixelsOptions,
  ExportLayerPixelsOptions,
  LayerExportGuard,
} from '@workbench/canvas-engine/api';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerContract, CanvasLayerSourceContract } from '@workbench/types';

import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { applyAdjustments } from '@workbench/canvas-engine/render/adjustments';

export type ExportLayerPixelsResult =
  | {
      status: 'ok';
      surface: ReturnType<RasterBackend['createSurface']>;
      rect: Rect;
      guard: LayerExportGuard;
      release(): void;
    }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' | 'aborted' };

export interface RasterExportControllerOptions {
  readonly backend: RasterBackend;
  readonly captureGuard: (layer: CanvasLayerContract, entry: LayerCacheEntry) => LayerExportGuard;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getOrStartRasterization: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV2,
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
  readonly pin?: (layerId: string) => { release(): void };
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

  private applyAdjustments(
    result: Extract<ExportLayerPixelsResult, { status: 'ok' }>,
    shouldApply: boolean
  ): ExportLayerPixelsResult {
    const layer = result.guard.layer;
    if (!shouldApply || layer.type !== 'raster' || !layer.adjustments) {
      return result;
    }
    const reservation = this.options.reserve?.(result.rect.width * result.rect.height * 8);
    if (reservation?.status === 'over-budget') {
      result.release();
      return { status: 'over-budget' };
    }
    const pinLease = this.options.pin?.(result.guard.layerId);
    let released = false;
    const release = (): void => {
      if (released) {
        return;
      }
      released = true;
      result.release();
      pinLease?.release();
      if (reservation?.status === 'ok') {
        reservation.lease.release();
      }
    };
    try {
      const surface = this.options.backend.createSurface(result.rect.width, result.rect.height);
      const ctx = surface.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, result.rect.width, result.rect.height);
      ctx.drawImage(result.surface.canvas, 0, 0);
      const imageData = ctx.getImageData(0, 0, result.rect.width, result.rect.height);
      applyAdjustments(imageData, layer.adjustments);
      ctx.putImageData(imageData, 0, 0);
      return { ...result, release, surface };
    } catch (error) {
      release();
      throw error;
    }
  }

  async rasterize(layerId: string, options: ExportLayerPixelsOptions = {}): Promise<ExportLayerPixelsResult> {
    const document = this.options.getDocument();
    if (!document) {
      return { status: 'missing' };
    }
    const layer = document.layers.find((candidate) => candidate.id === layerId);
    const source = layer ? renderableSourceOf(layer) : null;
    if (!layer || !source) {
      return { status: 'missing' };
    }
    if (!options.includeDisabled && !layer.isEnabled) {
      return { status: 'disabled' };
    }
    if (!this.options.isSupportedSource(source)) {
      return { status: 'unsupported' };
    }
    const liveEntry = this.options.layers.get(layerId);
    if (liveEntry && !liveEntry.stale && !this.options.isRasterizing(layer) && !isEmpty(liveEntry.rect)) {
      return this.applyAdjustments(
        {
          guard: this.options.captureGuard(layer, liveEntry),
          release: () => undefined,
          rect: liveEntry.rect,
          status: 'ok',
          surface: liveEntry.surface,
        },
        options.applyAdjustments === true
      );
    }
    const contentRect = getSourceContentRect(layer, document);
    if (isEmpty(contentRect)) {
      return { status: 'empty' };
    }
    const reservation = this.options.reserve?.(contentRect.width * contentRect.height * 8);
    if (reservation?.status === 'over-budget') {
      return { status: 'over-budget' };
    }
    const pinLease = this.options.pin?.(layerId);
    try {
      const rasterized = await this.options.getOrStartRasterization(layer, document, options.signal);
      if (rasterized !== 'published') {
        return { status: rasterized === 'aborted' ? 'aborted' : 'not-ready' };
      }
    } finally {
      pinLease?.release();
      if (reservation?.status === 'ok') {
        reservation.lease.release();
      }
    }
    const currentDocument = this.options.getDocument();
    const currentLayer = currentDocument?.layers.find((candidate) => candidate.id === layerId);
    const entry = this.options.layers.get(layerId);
    if (!currentLayer || !entry || entry.stale) {
      return { status: 'not-ready' };
    }
    const currentSource = renderableSourceOf(currentLayer);
    if (!currentSource) {
      return { status: 'missing' };
    }
    if (!options.includeDisabled && !currentLayer.isEnabled) {
      return { status: 'disabled' };
    }
    if (!this.options.isSupportedSource(currentSource)) {
      return { status: 'unsupported' };
    }
    if (isEmpty(entry.rect)) {
      return { status: 'empty' };
    }
    return this.applyAdjustments(
      {
        guard: this.options.captureGuard(currentLayer, entry),
        release: () => undefined,
        rect: entry.rect,
        status: 'ok',
        surface: entry.surface,
      },
      options.applyAdjustments === true
    );
  }

  private async reserveBaked(
    layerId: string,
    options: ExportBakedLayerPixelsOptions = {}
  ): Promise<ReservedExportLayerPixels> {
    const raw = await this.rasterize(layerId, { ...options, applyAdjustments: false });
    if (raw.status !== 'ok') {
      return noReservedPixels(raw);
    }
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
    const appliesAdjustments = options.applyAdjustments !== false && layer.type === 'raster' && !!layer.adjustments;
    const reservation = this.options.reserve?.(rect.width * rect.height * (appliesAdjustments ? 8 : 4));
    if (reservation?.status === 'over-budget') {
      raw.release();
      return noReservedPixels({ status: 'over-budget' });
    }
    const pinLease = this.options.pin?.(layerId);
    let released = false;
    const release = (): void => {
      if (released) {
        return;
      }
      released = true;
      pinLease?.release();
      if (reservation?.status === 'ok') {
        reservation.lease.release();
      }
      raw.release();
    };
    try {
      const surface = this.options.backend.createSurface(rect.width, rect.height);
      const ctx = surface.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - rect.x, matrix.f - rect.y);
      ctx.drawImage(raw.surface.canvas, raw.rect.x, raw.rect.y);
      if (appliesAdjustments && layer.type === 'raster' && layer.adjustments) {
        const imageData = ctx.getImageData(0, 0, rect.width, rect.height);
        applyAdjustments(imageData, layer.adjustments);
        ctx.putImageData(imageData, 0, 0);
      }
      return { release, result: { guard: raw.guard, rect, release, status: 'ok', surface } };
    } catch (error) {
      release();
      throw error;
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
