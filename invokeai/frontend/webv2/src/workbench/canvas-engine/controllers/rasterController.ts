import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type {
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/types';

export type DecodeImageResult =
  | { status: 'ok'; surface: RasterSurface; decodedWidth: number; decodedHeight: number }
  | { status: 'aborted' | 'stale' };

export interface RasterizationJob {
  abortedByCaller?: boolean;
  controller: AbortController;
  version: number;
  documentGeneration: number;
  source: CanvasLayerSourceContract;
  promise: Promise<'published' | 'stale' | 'error' | 'aborted'>;
}

import {
  createAdjustedSurfaceCache,
  type AdjustedSurfaceCache,
} from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createDecodedBitmapPool, type DecodedBitmapPool } from '@workbench/canvas-engine/render/decodedBitmapPool';
import {
  createDerivedSurfaceCache,
  type DerivedSurfaceCache,
} from '@workbench/canvas-engine/render/derivedSurfaceCache';
import { createLayerCacheStore, DEFAULT_CACHE_BUDGET_BYTES } from '@workbench/canvas-engine/render/layerCache';

import { RasterMemoryBudgetController } from './rasterMemoryBudgetController';

export interface RasterControllerOptions {
  readonly backend: RasterBackend;
  readonly diagnostics: CanvasDiagnostics;
  readonly onVersionChange?: (layerId: string) => void;
  readonly getDocument?: () => CanvasDocumentContractV2 | null;
  readonly getLayerImageName?: (layer: CanvasLayerContract) => string | null;
  readonly imageResolver?: (imageName: string, signal?: AbortSignal) => Promise<Blob>;
}

export class RasterController {
  readonly layers: LayerCacheStore;
  readonly derived: DerivedSurfaceCache;
  readonly adjustments: AdjustedSurfaceCache;
  readonly memory: RasterMemoryBudgetController;
  readonly bitmaps: DecodedBitmapPool;
  private readonly jobs = new Map<string, RasterizationJob>();
  private readonly activeJobs = new Set<RasterizationJob>();
  private readonly trackedImages = new Map<string, string>();
  private readonly mirroredImages = new Map<string, string>();
  private readonly thumbnailKeys = new Map<string, string>();
  private documentGeneration = 0;
  private disposed = false;
  private readonly getDocument: () => CanvasDocumentContractV2 | null;
  private readonly getLayerImageName: (layer: CanvasLayerContract) => string | null;
  private readonly backend: RasterBackend;
  private readonly imageResolver: ((imageName: string, signal?: AbortSignal) => Promise<Blob>) | null;

  constructor(options: RasterControllerOptions) {
    this.backend = options.backend;
    this.memory = new RasterMemoryBudgetController({ budgetBytes: DEFAULT_CACHE_BUDGET_BYTES });
    this.bitmaps = createDecodedBitmapPool({ onBytesChange: (bytes) => this.memory.setDecodedBytes(bytes) });
    this.layers = createLayerCacheStore(options.backend, { onVersionChange: options.onVersionChange });
    this.getDocument = options.getDocument ?? (() => null);
    this.getLayerImageName = options.getLayerImageName ?? (() => null);
    this.imageResolver = options.imageResolver ?? null;
    this.derived = createDerivedSurfaceCache(options.diagnostics);
    this.adjustments = createAdjustedSurfaceCache(options.backend, this.derived);
  }

  async decodeImage(
    image: CanvasImageRef,
    options: {
      signal?: AbortSignal;
      isCurrent?: () => boolean;
      scaleToImage?: boolean;
      validateDecoded?: (width: number, height: number) => void;
    } = {}
  ): Promise<DecodeImageResult> {
    if (!this.imageResolver) {
      throw new Error('RasterController requires an image resolver to decode images.');
    }
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    const blob = await this.imageResolver(image.imageName, options.signal);
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    if (options.isCurrent && !options.isCurrent()) {
      return { status: 'stale' };
    }
    const bitmap = await this.backend.createImageBitmap(blob);
    if (options.signal?.aborted || (options.isCurrent && !options.isCurrent())) {
      bitmap.close();
      return { status: options.signal?.aborted ? 'aborted' : 'stale' };
    }
    try {
      options.validateDecoded?.(bitmap.width, bitmap.height);
      const surface = this.backend.createSurface(image.width, image.height);
      surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
      surface.ctx.clearRect(0, 0, image.width, image.height);
      if (options.scaleToImage === false) {
        surface.ctx.drawImage(bitmap, 0, 0);
      } else {
        surface.ctx.drawImage(bitmap, 0, 0, image.width, image.height);
      }
      return { decodedHeight: bitmap.height, decodedWidth: bitmap.width, status: 'ok', surface };
    } finally {
      bitmap.close();
    }
  }

  async decodeBlob(
    blob: Blob,
    dimensions?: { width: number; height: number; scale?: boolean }
  ): Promise<{ surface: RasterSurface; decodedWidth: number; decodedHeight: number }> {
    const bitmap = await this.backend.createImageBitmap(blob);
    try {
      const width = dimensions?.width ?? bitmap.width;
      const height = dimensions?.height ?? bitmap.height;
      const surface = this.backend.createSurface(width, height);
      surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
      surface.ctx.clearRect(0, 0, width, height);
      if (dimensions?.scale) {
        surface.ctx.drawImage(bitmap, 0, 0, width, height);
      } else {
        surface.ctx.drawImage(bitmap, 0, 0);
      }
      return { decodedHeight: bitmap.height, decodedWidth: bitmap.width, surface };
    } finally {
      bitmap.close();
    }
  }

  getAdjustedSurface(layer: CanvasLayerContract, entry: LayerCacheEntry): RasterSurface | null {
    return layer.type === 'raster' ? this.adjustments.get(layer.id, entry, layer.adjustments) : null;
  }

  deleteDerivedSurfaces(layerId: string): void {
    this.adjustments.delete(layerId);
    this.derived.deleteLayer(layerId);
  }

  getDocumentGeneration(): number {
    return this.documentGeneration;
  }

  invalidateDocument(): void {
    this.documentGeneration += 1;
    this.cancelAllRasterization();
  }

  getRasterizationJob(layerId: string): RasterizationJob | undefined {
    return this.jobs.get(layerId);
  }

  installRasterizationJob(layerId: string, job: RasterizationJob): void {
    this.jobs.set(layerId, job);
    this.activeJobs.add(job);
  }

  finishRasterizationJob(layerId: string, job: RasterizationJob): void {
    if (this.jobs.get(layerId) === job) {
      this.jobs.delete(layerId);
    }
    this.activeJobs.delete(job);
  }

  cancelRasterization(layerId: string): void {
    const job = this.jobs.get(layerId);
    if (!job) {
      return;
    }
    this.jobs.delete(layerId);
    job.controller.abort();
  }

  cancelAllRasterization(): void {
    const jobs = [...this.jobs.values()];
    this.jobs.clear();
    for (const job of jobs) {
      job.controller.abort();
    }
  }

  hasActiveSourceImage(imageName: string): boolean {
    for (const job of this.activeJobs) {
      if (job.source.type === 'image' && job.source.image.imageName === imageName) {
        return true;
      }
    }
    return false;
  }

  getTrackedImage(layerId: string): string | undefined {
    return this.trackedImages.get(layerId);
  }
  setTrackedImage(layerId: string, imageName: string): void {
    this.trackedImages.set(layerId, imageName);
  }
  deleteTrackedImage(layerId: string): void {
    this.trackedImages.delete(layerId);
  }
  trackedImageIds(): string[] {
    return [...this.trackedImages.keys()];
  }
  hasTrackedImage(imageName: string): boolean {
    return [...this.trackedImages.values()].includes(imageName);
  }
  clearTrackedImages(): void {
    this.trackedImages.clear();
  }

  getMirroredImage(layerId: string): string | undefined {
    return this.mirroredImages.get(layerId);
  }
  setMirroredImage(layerId: string, imageName: string): void {
    this.mirroredImages.set(layerId, imageName);
  }
  deleteMirroredImage(layerId: string): void {
    this.mirroredImages.delete(layerId);
  }
  mirroredImageNames(): string[] {
    return [...this.mirroredImages.values()];
  }
  clearMirroredImages(): void {
    this.mirroredImages.clear();
  }

  getThumbnailKey(layerId: string): string | undefined {
    return this.thumbnailKeys.get(layerId);
  }
  setThumbnailKey(layerId: string, key: string): void {
    this.thumbnailKeys.set(layerId, key);
  }
  deleteThumbnailKey(layerId: string): void {
    this.thumbnailKeys.delete(layerId);
  }
  clearThumbnailKeys(): void {
    this.thumbnailKeys.clear();
  }

  releaseBitmapIfUnreferenced(imageName: string): void {
    // Decoded bitmaps are lease-owned and close automatically after the final
    // rasterizer releases them. Retained for callers that also update tracking.
    void imageName;
  }

  untrackLayerImage(layerId: string): void {
    const imageName = this.getTrackedImage(layerId);
    if (!imageName) {
      return;
    }
    this.deleteTrackedImage(layerId);
    this.releaseBitmapIfUnreferenced(imageName);
  }

  trackPublishedLayerImage(layer: CanvasLayerContract): void {
    const previous = this.getTrackedImage(layer.id);
    const current = this.getLayerImageName(layer);
    if (current) {
      this.setTrackedImage(layer.id, current);
    } else {
      this.deleteTrackedImage(layer.id);
    }
    if (previous && previous !== current) {
      this.releaseBitmapIfUnreferenced(previous);
    }
  }

  dropLayer(layerId: string): void {
    this.cancelRasterization(layerId);
    this.untrackLayerImage(layerId);
    this.layers.delete(layerId);
    this.deleteDerivedSurfaces(layerId);
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.cancelAllRasterization();
    this.clearTrackedImages();
    this.clearMirroredImages();
    this.clearThumbnailKeys();
    this.layers.dispose();
    this.bitmaps.dispose();
    this.adjustments.dispose();
    this.memory.dispose();
  }
}
