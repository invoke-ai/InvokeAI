/**
 * Shared vocabulary for the source rasterizers.
 *
 * Type-only module (no runtime), so it can be imported by both the dispatch
 * (`rasterizeSource`) and the per-source rasterizers without any import
 * cycle. Zero React, zero side effects.
 */

import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

/**
 * Resolves a persisted image asset (referenced by name in the document) to a
 * `Blob` for decoding. The DOM implementation (deriving a URL and fetching)
 * ships with the React shell task; the engine only depends on this seam so it
 * stays node-testable.
 */
export type ImageResolver = (imageName: string) => Promise<Blob>;

/**
 * The result of rasterizing a source: the surface holding its pixels plus the
 * content `rect` those pixels occupy in the layer's LOCAL coordinate space. For
 * origin-anchored sources (image / shape / text / gradient) the rect origin is
 * `(0, 0)`; a `paint` source places its bitmap at the persisted `offset`.
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
  /** Holds the decoded-bitmap cache (keyed by image name). */
  store: LayerCacheStore;
  /**
   * Document pixel size. Layers are content-sized, so this only backs the
   * legacy default for gradients that predate the explicit extent field (they
   * were document-sized by construction and must render identically).
   */
  documentSize: { width: number; height: number };
}
