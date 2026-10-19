/**
 * Trim grown paint caches to visible alpha before persistence so erased padding does not become phantom layer
 * bounds. Full-surface synchronous readback runs in debounced flush, outside the stroke hot path, and scales with
 * cache extent.
 */

import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';

import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { alphaBounds } from '@workbench/canvas-engine/render/alphaBounds';

/**
 * `deferred` — something else owns these pixels; retry later. `emptied` — none
 * visible, cache now zero-rect. `kept` — already tight, or no cache. `trimmed` —
 * cropped smaller.
 */
export type PaintCacheTrim = 'deferred' | 'emptied' | 'kept' | 'trimmed';

export interface TrimPaintCacheDeps {
  readonly layers: LayerCacheStore;
  /** A gesture, session or rasterization owns the pixels; persistence retries once it ends. */
  readonly isLayerBusy: (layerId: string) => boolean;
  /** A read lease holds the surface; persist it untrimmed rather than reshaping pixels it reads. */
  readonly isLayerPinned: (layerId: string) => boolean;
}

export const trimPaintCacheToAlpha = (deps: TrimPaintCacheDeps, layerId: string): PaintCacheTrim => {
  // `peek`, not `get`: a persistence-side probe must not reorder the LRU.
  const entry = deps.layers.peek(layerId);
  if (!entry || isEmpty(entry.rect)) {
    return 'kept';
  }
  // The unpublished/stale guard is correctness, not thrift: the rasterizer sizes the
  // entry from the persisted rect BEFORE its async decode fills it, so scanning there
  // would read a blank surface and clear a good bitmap on every document load.
  if (!entry.hasPublishedPixels || entry.stale || deps.isLayerBusy(layerId)) {
    return 'deferred';
  }
  if (deps.isLayerPinned(layerId)) {
    return 'kept';
  }
  const bounds = alphaBounds(entry.surface.ctx.getImageData(0, 0, entry.rect.width, entry.rect.height));
  if (isEmpty(bounds)) {
    deps.layers.shrinkToRect(layerId, { height: 0, width: 0, x: entry.rect.x, y: entry.rect.y });
    return 'emptied';
  }
  if (bounds.width === entry.rect.width && bounds.height === entry.rect.height) {
    return 'kept';
  }
  // `bounds` is surface-local; the cache rect is layer-local.
  deps.layers.shrinkToRect(layerId, {
    height: bounds.height,
    width: bounds.width,
    x: entry.rect.x + bounds.x,
    y: entry.rect.y + bounds.y,
  });
  return 'trimmed';
};
