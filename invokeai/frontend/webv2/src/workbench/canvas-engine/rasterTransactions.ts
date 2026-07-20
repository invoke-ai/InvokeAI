import type { CanvasStateContractV2 } from './contracts';
import type { RasterSurface } from './render/raster';
import type { Rect } from './types';

export interface CanvasDetachedLayerSurface {
  readonly rect: Rect;
  readonly surface: RasterSurface;
}

export interface CanvasRasterSnapshot {
  readonly canvas: CanvasStateContractV2;
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
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect }>;
  reserve?(
    bytes: number
  ):
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
  uploadImage(blob: Blob): Promise<{ imageName: string; width: number; height: number }>;
}
