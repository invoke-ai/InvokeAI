/**
 * Trims a paint layer's raster cache back to the pixels it can actually show.
 *
 * The cache only ever GROWS: a stroke chunk-pads its extent so a long drag
 * reallocates once per chunk crossed rather than once per pointer batch, and the
 * eraser grows it on the same path even though `destination-out` can only remove
 * alpha. Nothing ever shrank it, so the extent recorded the high-water mark of
 * everywhere the user's pointer had been — and persistence baked that extent into
 * the layer's bitmap dimensions, which is what the move outline, the transform
 * frame and fit-to-content all read. A fully erased layer therefore kept a
 * full-size, draggable, transformable rectangle around nothing.
 *
 * This is the convergence step. It runs from the debounced persistence flush,
 * which already reads the whole surface to encode it, so the alpha scan is
 * proportionally cheap and never touches the paint hot path.
 *
 * Every guard below is load-bearing; see {@link trimPaintCacheToAlpha}.
 *
 * Zero React, zero import-time side effects.
 */

import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';

import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { alphaBounds } from '@workbench/canvas-engine/render/alphaBounds';

/**
 * What a trim attempt did.
 *
 * - `deferred` — something else owns or frames these pixels right now; try later.
 * - `emptied` — no visible pixels at all; the cache is now zero-rect.
 * - `kept` — nothing to do (already tight, or no cache to trim).
 * - `trimmed` — the cache was cropped to a smaller extent.
 */
export type PaintCacheTrim = 'deferred' | 'emptied' | 'kept' | 'trimmed';

export interface TrimPaintCacheDeps {
  readonly layers: LayerCacheStore;
  /**
   * True while something other than persistence owns or frames this layer's
   * pixels. Supplied by the engine so this module stays free of engine knowledge.
   */
  readonly isLayerBusy: (layerId: string) => boolean;
}

/**
 * Shrinks `layerId`'s cache to the bounds of its non-transparent pixels, or
 * collapses it to a zero rect when it has none.
 *
 * The guards, and why each one matters:
 *
 * - Reads through `peek`, not `get`: a persistence-side probe must not reorder the
 *   LRU and make an unrelated layer the next eviction candidate.
 * - `!hasPublishedPixels || stale` defers. This is the correctness guard, not an
 *   optimization: the layer rasterizer creates the cache entry sized to the
 *   persisted content rect BEFORE its async decode fills it, so a scan in that
 *   window would read a blank surface and conclude a perfectly good bitmap was
 *   empty — clearing it on every document load.
 * - `isLayerBusy` defers for an open gesture, transform/text session, floating
 *   selection or in-flight rasterization. A transform frame in particular is
 *   expressed relative to this rect, so shrinking underneath it would make the
 *   frame jump and Apply would bake against a rect the user never framed.
 */
export const trimPaintCacheToAlpha = (deps: TrimPaintCacheDeps, layerId: string): PaintCacheTrim => {
  const entry = deps.layers.peek(layerId);
  if (!entry || isEmpty(entry.rect)) {
    return 'kept';
  }
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
  // `bounds` is in surface coordinates; the cache rect is layer-local.
  deps.layers.shrinkToRect(layerId, {
    height: bounds.height,
    width: bounds.width,
    x: entry.rect.x + bounds.x,
    y: entry.rect.y + bounds.y,
  });
  return 'trimmed';
};
