/**
 * Trims a paint layer's raster cache back to the pixels it can actually show.
 *
 * The cache only ever grows (strokes chunk-pad their extent; the eraser grows it
 * too), and persistence bakes that extent into the layer's bitmap dimensions — which
 * is what the move outline, the transform frame and fit-to-content read. So a fully
 * erased layer kept a draggable rectangle around nothing.
 *
 * Runs from the debounced persistence flush, which already reads the whole surface
 * to encode it, so the alpha scan never touches the paint hot path.
 *
 * Zero React, zero import-time side effects.
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
  /** Injected by the engine, so this module stays free of engine knowledge. */
  readonly isLayerBusy: (layerId: string) => boolean;
}

/**
 * Shrinks `layerId`'s cache to its non-transparent bounds, or collapses it to a zero
 * rect when it has none.
 */
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
