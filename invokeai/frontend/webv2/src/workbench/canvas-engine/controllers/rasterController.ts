import type {
  CanvasAdjustmentsContract,
  CanvasDocumentContractV3,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';

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
import { createGroupSurfaceCache, type GroupSurfaceCache } from '@workbench/canvas-engine/render/groupSurfaceCache';
import { createLayerCacheStore, DEFAULT_CACHE_BUDGET_BYTES } from '@workbench/canvas-engine/render/layerCache';

import { RasterMemoryBudgetController } from './rasterMemoryBudgetController';

export interface RasterControllerOptions {
  readonly backend: RasterBackend;
  readonly diagnostics: CanvasDiagnostics;
  readonly budgetBytes?: number;
  readonly onVersionChange?: (layerId: string) => void;
  readonly getDocument?: () => CanvasDocumentContractV3 | null;
  readonly imageResolver?: (imageName: string, signal?: AbortSignal) => Promise<Blob>;
  /** Layers whose live pixels nothing else can reconstruct: unpersisted paint and open edit sessions. */
  readonly isLayerHeld?: (layerId: string) => boolean;
  /** Layers whose live cache already includes their adjustments (an open pixel transaction), drawn raw. */
  readonly isAdjustmentBaked?: (layerId: string) => boolean;
}

/** Access clocks captured before a frame draws; artifacts read after them are that frame's working set. */
export interface RasterFrameUsage {
  readonly derivedTick: number;
  readonly groupTick: number;
}

export interface RasterBudgetResult {
  readonly evictedBaseLayerIds: readonly string[];
  readonly overageBytes: number;
}

/** Owns every raster allocation class, their accounting, residency pins, and working-set eviction. */
export class RasterController {
  readonly layers: LayerCacheStore;
  readonly derived: DerivedSurfaceCache;
  readonly adjustments: AdjustedSurfaceCache;
  readonly groups: GroupSurfaceCache;
  readonly memory: RasterMemoryBudgetController;
  readonly bitmaps: DecodedBitmapPool;
  private readonly diagnostics: CanvasDiagnostics;
  private readonly isLayerHeld: (layerId: string) => boolean;
  private readonly isAdjustmentBaked: (layerId: string) => boolean;
  private readonly jobs = new Map<string, RasterizationJob>();
  private readonly activeJobs = new Set<RasterizationJob>();
  private readonly thumbnailKeys = new Map<string, string>();
  private documentGeneration = 0;
  private disposed = false;
  private readonly getDocument: () => CanvasDocumentContractV3 | null;
  private readonly backend: RasterBackend;
  private readonly imageResolver: ((imageName: string, signal?: AbortSignal) => Promise<Blob>) | null;

  constructor(options: RasterControllerOptions) {
    this.backend = options.backend;
    this.diagnostics = options.diagnostics;
    this.isLayerHeld = options.isLayerHeld ?? (() => false);
    this.isAdjustmentBaked = options.isAdjustmentBaked ?? (() => false);
    const memory = new RasterMemoryBudgetController({ budgetBytes: options.budgetBytes ?? DEFAULT_CACHE_BUDGET_BYTES });
    this.memory = memory;
    this.bitmaps = createDecodedBitmapPool({ onBytesChange: (bytes) => memory.setCategoryBytes('decoded', bytes) });
    this.layers = createLayerCacheStore(options.backend, {
      onBytesChange: (bytes) => memory.setCategoryBytes('base', bytes),
      onVersionChange: options.onVersionChange,
    });
    this.getDocument = options.getDocument ?? (() => null);
    this.imageResolver = options.imageResolver ?? null;
    this.derived = createDerivedSurfaceCache(options.diagnostics, (bytes) => memory.setCategoryBytes('derived', bytes));
    this.adjustments = createAdjustedSurfaceCache(options.backend, this.derived, (layerId, version) =>
      this.layers.damageSince(layerId, version)
    );
    this.groups = createGroupSurfaceCache({
      createSurface: (width, height) => options.backend.createSurface(width, height),
      damageSince: (layerId, version) => this.layers.damageSince(layerId, version),
      diagnostics: options.diagnostics,
      getAdjustedSurface: (layer, entry) => this.getAdjustedSurface(layer, entry),
      getCacheEntry: (layerId) => this.layers.get(layerId),
      onBytesChange: (bytes) => memory.setCategoryBytes('group', bytes),
    });
  }

  /** Pinned or held layers survive eviction; their bytes become accounted overage instead. */
  isProtected(layerId: string): boolean {
    return this.memory.isPinned(layerId) || this.isLayerHeld(layerId);
  }

  beginFrame(): RasterFrameUsage {
    return { derivedTick: this.derived.tick(), groupTick: this.groups.tick() };
  }

  /**
   * Reclaims, in order, derived and group artifacts the frame did not use, then unprotected base caches outside
   * the working set. Whatever the working set still needs stays resident as reported overage.
   */
  enforceBudget(workingSetLayerIds: ReadonlySet<string>, usage: RasterFrameUsage): RasterBudgetResult {
    const excess = (): number => this.memory.snapshot().overageBytes;
    if (excess() === 0) {
      return { evictedBaseLayerIds: [], overageBytes: 0 };
    }
    this.derived.evict(
      Math.max(0, this.derived.byteSize() - excess()),
      (_layerId, lastUsed) => lastUsed > usage.derivedTick
    );
    if (excess() > 0) {
      this.groups.evict(Math.max(0, this.groups.byteSize() - excess()), usage.groupTick);
    }
    const evictedBaseLayerIds =
      excess() > 0
        ? this.layers.evict(
            (layerId) => workingSetLayerIds.has(layerId) || this.isProtected(layerId),
            Math.max(0, this.layers.byteSize() - excess())
          )
        : [];
    for (const layerId of evictedBaseLayerIds) {
      this.deleteDerivedSurfaces(layerId);
    }
    const overageBytes = excess();
    if (overageBytes > 0) {
      this.diagnostics.add('rasterOverageBytes', overageBytes);
    }
    return { evictedBaseLayerIds, overageBytes };
  }

  /** Drops every derived and group surface and each base cache nothing pins or holds. */
  releaseReconstructible(): void {
    this.derived.dispose();
    this.groups.clear();
    for (const layerId of this.layers.evict((candidate) => this.isProtected(candidate), 0)) {
      this.deleteDerivedSurfaces(layerId);
    }
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
    return layer.type === 'raster' && !this.isAdjustmentBaked(layer.id)
      ? this.adjustments.get(layer.id, entry, layer.adjustments)
      : null;
  }

  /** The adjusted copy of a layer's live cache (built and memoized on a miss), when `surface` is that cache and unbaked. */
  getAdjustedCacheSurface(
    layerId: string,
    surface: RasterSurface,
    adjustments: CanvasAdjustmentsContract
  ): RasterSurface | null {
    const entry = this.layers.peek(layerId);
    return entry?.surface === surface && !this.isAdjustmentBaked(layerId)
      ? this.adjustments.get(layerId, entry, adjustments)
      : null;
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

  dropLayer(layerId: string): void {
    this.cancelRasterization(layerId);
    this.layers.delete(layerId);
    this.deleteDerivedSurfaces(layerId);
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.cancelAllRasterization();
    this.clearThumbnailKeys();
    this.layers.dispose();
    this.derived.dispose();
    this.groups.clear();
    this.bitmaps.dispose();
    this.memory.dispose();
  }
}
