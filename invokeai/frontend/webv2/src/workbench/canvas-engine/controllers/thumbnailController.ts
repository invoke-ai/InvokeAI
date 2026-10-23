import type { LayerThumbnailRequestResult } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasAdjustmentsContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';

import { lookupDocumentLayer } from '@workbench/canvas-engine/document-model/documentModel';
import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { applyAdjustments, isIdentityAdjustments, noiseGrainSpan } from '@workbench/canvas-engine/render/adjustments';
import { renderControlTransparency } from '@workbench/canvas-engine/render/controlTransparency';
import { colorizeMask } from '@workbench/canvas-engine/render/maskFill';
import { fitThumbnailSize } from '@workbench/canvas-engine/render/thumbnail';

export interface ThumbnailControllerOptions {
  readonly backend: RasterBackend;
  readonly projectId: string;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly getActiveProjectId: () => string | null;
  readonly getEntry: (layerId: string) => LayerCacheEntry | undefined;
  readonly getCheckerboard: () => RasterSurface;
  readonly getMaskPattern: (style: string, color: string) => RasterSurface | null;
  readonly isDisposed: () => boolean;
  readonly isSupportedSource: (source: CanvasLayerSourceContract) => boolean;
  /** The display's current adjusted copy of a raster layer's cache, or null; never builds a layer-sized surface. */
  readonly peekAdjustedSurface: (layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null;
  readonly rasterize: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV3
  ) => Promise<'published' | 'stale' | 'error'>;
  readonly setStatus: (layerId: string, status: 'loading' | 'ready' | 'error' | null) => void;
  readonly reportError: (layerId: string, error: unknown) => void;
  readonly reserve?: (
    bytes: number
  ) =>
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
  readonly pin?: (layerId: string) => { release(): void };
}

/**
 * Bounded fallback without a display copy: the stack adjusts the downscaled pixels, its noise on the thumbnail's own
 * grid. A downscale box-averages about 1/scale × 1/scale pixels, which leaves grain spanning `noiseGrainSpan` pixels
 * `scale × span` of its σ once blocks are larger than the grain (exact for white noise), and keeps it when they are
 * smaller; the grain itself shrinks to `size × scale` thumbnail pixels, at least one. An approximation: exact only in
 * those limits, and it ignores how resampling actually weights pixels.
 */
const atThumbnailScale = (adjustments: CanvasAdjustmentsContract, scale: number): CanvasAdjustmentsContract =>
  scale >= 1
    ? adjustments
    : adjustments.map((entry) =>
        entry.type === 'noise'
          ? {
              ...entry,
              amount: entry.amount * Math.min(1, scale * noiseGrainSpan(entry.size)),
              size: Math.max(1, entry.size * scale),
            }
          : entry
      );

/** Owns bounded thumbnail rendering and lazy rasterization status. */
export class ThumbnailController {
  private disposed = false;

  constructor(private readonly deps: ThumbnailControllerOptions) {}

  draw(layerId: string, target: HTMLCanvasElement, maxSize: number): boolean {
    if (this.disposed) {
      return false;
    }
    const entry = this.deps.getEntry(layerId);
    const document = this.deps.getDocument();
    const layer = document ? lookupDocumentLayer(document, layerId) : null;
    if (!entry?.hasPublishedPixels || !layer) {
      return false;
    }
    const { height, width } = fitThumbnailSize(entry.surface.width, entry.surface.height, maxSize);
    const context = target.getContext('2d');
    if (width === 0 || height === 0 || !context) {
      return false;
    }
    const reservation = this.deps.reserve?.(width * height * 12);
    if (reservation?.status === 'over-budget') {
      return false;
    }
    const pinLease = this.deps.pin?.(layerId);
    try {
      target.width = width;
      target.height = height;
      context.clearRect(0, 0, width, height);
      const thumbnail = this.deps.backend.createSurface(width, height);
      thumbnail.ctx.setTransform(1, 0, 0, 1, 0, 0);
      thumbnail.ctx.clearRect(0, 0, width, height);
      thumbnail.ctx.globalAlpha = 1;
      thumbnail.ctx.globalCompositeOperation = 'source-over';
      const adjusted = layer.type === 'raster' && !isIdentityAdjustments(layer.adjustments) ? layer.adjustments : null;
      // Downscale the display's adjusted copy when it is current, so the thumbnail shows the canvas's grain and color.
      const displayed = adjusted ? this.deps.peekAdjustedSurface(layer, entry) : null;
      thumbnail.ctx.drawImage((displayed ?? entry.surface).canvas, 0, 0, width, height);
      const checker = context.createPattern(this.deps.getCheckerboard().canvas as CanvasImageSource, 'repeat');
      if (checker) {
        context.fillStyle = checker;
        context.fillRect(0, 0, width, height);
      }
      let display = thumbnail;
      if (adjusted && !displayed) {
        const pixels = thumbnail.ctx.getImageData(0, 0, width, height);
        applyAdjustments(pixels, atThumbnailScale(adjusted, width / entry.surface.width), { x: 0, y: 0 });
        thumbnail.ctx.putImageData(pixels, 0, 0);
      } else if (layer.type === 'control' && layer.withTransparencyEffect) {
        display = renderControlTransparency(this.deps.backend, thumbnail, width, height);
      } else if (layer.type === 'inpaint_mask' || layer.type === 'regional_guidance') {
        const { fill } = layer.mask;
        display = colorizeMask(
          this.deps.backend,
          thumbnail,
          width,
          height,
          fill,
          this.deps.getMaskPattern(fill.style, fill.color)
        );
      }
      context.globalAlpha = layer.opacity;
      context.drawImage(display.canvas as CanvasImageSource, 0, 0);
      return true;
    } finally {
      pinLease?.release();
      if (reservation?.status === 'ok') {
        reservation.lease.release();
      }
    }
  }

  async request(layerId: string): Promise<LayerThumbnailRequestResult> {
    if (this.disposed || this.deps.isDisposed() || this.deps.getActiveProjectId() !== this.deps.projectId) {
      this.deps.setStatus(layerId, null);
      return this.disposed || this.deps.isDisposed() ? 'missing' : 'stale';
    }
    const document = this.deps.getDocument();
    const layer = document ? lookupDocumentLayer(document, layerId) : null;
    if (!document || !layer) {
      this.deps.setStatus(layerId, null);
      return 'missing';
    }
    const source = renderableSourceOf(layer);
    if (!source || !this.deps.isSupportedSource(source)) {
      this.deps.setStatus(layerId, null);
      return 'unsupported';
    }
    const entry = this.deps.getEntry(layerId);
    if (entry?.hasPublishedPixels && !entry.stale) {
      this.deps.setStatus(layerId, 'ready');
      return 'ready';
    }
    const rect = getSourceContentRect(layer, document);
    const reservation = this.deps.reserve?.(rect.width * rect.height * 8);
    if (reservation?.status === 'over-budget') {
      this.deps.setStatus(layerId, null);
      return 'over-budget';
    }
    const pinLease = this.deps.pin?.(layerId);
    this.deps.setStatus(layerId, 'loading');
    try {
      const result = await this.deps.rasterize(layer, document);
      return result === 'published' ? 'ready' : result;
    } catch (error) {
      this.deps.setStatus(layerId, 'error');
      this.deps.reportError(layerId, error);
      return 'error';
    } finally {
      pinLease?.release();
      if (reservation?.status === 'ok') {
        reservation.lease.release();
      }
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}
