import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { DecodedBitmapPool } from '@workbench/canvas-engine/render/decodedBitmapPool';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

/** Resolves persisted image names to decode blobs; application networking stays behind this injectable seam. */
export type ImageResolver = (imageName: string, signal?: AbortSignal) => Promise<Blob>;

/** A text source whose family may be replaced by an injected custom-font alias. */
export type RasterizeTextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;

/**
 * Pixels and their layer-local content rect. Image/shape/text/gradient start at zero; paint uses its persisted
 * offset.
 */
export interface RasterizeResult {
  surface: RasterSurface;
  rect: Rect;
}

/** Dependencies shared by the source rasterizers. */
export interface RasterizeDeps {
  /** Surface + bitmap factory seam. */
  backend: RasterBackend;
  /** Fetches image blobs by name for decoding. */
  resolver: ImageResolver;
  /** Cancels pending image resolution and prevents decode/cache publication. */
  signal?: AbortSignal;
  /** Holds the decoded-bitmap cache (keyed by image name). */
  store: LayerCacheStore;
  /** Coalesces concurrent decodes and owns decoded pixels only for the duration of a rasterization. */
  bitmapPool?: DecodedBitmapPool;
  /** Resolves a stable persisted face to the runtime family used for this render. */
  resolveFontFamily?: (source: RasterizeTextSource) => string;
  /**
   * Document pixel size. Layers are content-sized, so this only backs the
   * legacy default for gradients that predate the explicit extent field (they
   * were document-sized by construction and must render identically).
   */
  documentSize: { width: number; height: number };
}
