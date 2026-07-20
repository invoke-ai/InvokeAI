import type { LayerThumbnailRequestResult } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';

import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { applyAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { renderControlTransparency } from '@workbench/canvas-engine/render/controlTransparency';
import { colorizeMask } from '@workbench/canvas-engine/render/maskFill';
import { fitThumbnailSize } from '@workbench/canvas-engine/render/thumbnail';

export interface ThumbnailControllerOptions {
  readonly backend: RasterBackend;
  readonly projectId: string;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getActiveProjectId: () => string | null;
  readonly getEntry: (layerId: string) => LayerCacheEntry | undefined;
  readonly getCheckerboard: () => RasterSurface;
  readonly getMaskPattern: (style: string, color: string) => RasterSurface | null;
  readonly isDisposed: () => boolean;
  readonly isSupportedSource: (source: CanvasLayerSourceContract) => boolean;
  readonly rasterize: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV2
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

/** Owns bounded thumbnail rendering and lazy rasterization status. */
export class ThumbnailController {
  private disposed = false;

  constructor(private readonly deps: ThumbnailControllerOptions) {}

  draw(layerId: string, target: HTMLCanvasElement, maxSize: number): boolean {
    if (this.disposed) {
      return false;
    }
    const entry = this.deps.getEntry(layerId);
    const layer = this.deps.getDocument()?.layers.find((candidate) => candidate.id === layerId);
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
      thumbnail.ctx.drawImage(entry.surface.canvas, 0, 0, width, height);
      const checker = context.createPattern(this.deps.getCheckerboard().canvas as CanvasImageSource, 'repeat');
      if (checker) {
        context.fillStyle = checker;
        context.fillRect(0, 0, width, height);
      }
      let display = thumbnail;
      if (layer.type === 'raster' && !isIdentityAdjustments(layer.adjustments)) {
        const pixels = thumbnail.ctx.getImageData(0, 0, width, height);
        applyAdjustments(pixels, layer.adjustments);
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
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
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
