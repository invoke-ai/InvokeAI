import type { LayerExportGuard } from './capabilities';
import type { CanvasStateContractV3 } from './contracts';
import type { RasterSurface } from './render/raster';
import type { Rect } from './types';

/**
 * Owned access to one layer's pixels. The source stays pinned (resident, untrimmed) until `release`, which is
 * idempotent; `guard` tells whether the pixels still describe the live layer.
 */
export interface RasterReadLease {
  readonly surface: RasterSurface;
  /** Layer-local extent of `surface`. */
  readonly rect: Rect;
  readonly guard: LayerExportGuard;
  release(): void;
}

export interface CanvasDetachedLayerSurface {
  readonly rect: Rect;
  readonly surface: RasterSurface;
}

export interface CanvasRasterSnapshot {
  readonly canvas: CanvasStateContractV3;
  readonly documentGeneration: number;
  readonly emptyLayerIds: ReadonlySet<string>;
  readonly layerSurfaces: ReadonlyMap<string, CanvasDetachedLayerSurface>;
  release(): void;
}

export type CaptureRasterSnapshotResult =
  | { status: 'ok'; snapshot: CanvasRasterSnapshot }
  | { status: 'stale' | 'aborted' | 'not-ready' | 'over-budget' };

export interface CanvasCompositeExecutorDeps {
  backend: {
    createSurface(width: number, height: number): RasterSurface;
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  reserve?(
    bytes: number
  ):
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
  uploadImage(blob: Blob): Promise<{ imageName: string; width: number; height: number }>;
}
