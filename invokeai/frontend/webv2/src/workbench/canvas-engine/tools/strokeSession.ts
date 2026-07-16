/**
 * The per-gesture paint session shared by the brush and eraser tools.
 *
 * A session is created on pointer-down against a resolved target layer's cache
 * surface and lives until commit (pointer-up) or cancel (Esc / pointercancel).
 * It owns the hot path the plan pins as invariant: coalesced points accumulate,
 * and on each batch the full freehand outline is filled into a scratch surface
 * at **full alpha** and composited into the layer cache at the stroke's opacity —
 * so overlapping segments within one stroke never darken (the "stroke buffer"
 * approach). Brush composites `source-over` in the fill color; eraser composites
 * `destination-out`. When the target raster layer's transparency is LOCKED, the
 * brush composites `source-atop` instead, so colour only lands on already-opaque
 * pixels and the layer's alpha channel is never grown (legacy "lock transparent
 * pixels"). The eraser is refused entirely on a transparency-locked layer (it
 * would alter alpha), handled by the tool before a session is created.
 *
 * ## Per-frame restore/recapture
 *
 * The cache is the live preview, so it must show `before ∪ stroke@opacity` each
 * frame. To recomposite without compounding opacity, every frame first restores
 * the previously-painted region from the captured "before" pixels, then
 * recaptures the pristine "before" for the (monotonically growing) dirty region,
 * then composites the whole accumulated stroke once. This keeps `beforeImageData`
 * exactly equal to the pre-stroke pixels over the final dirty rect — which is
 * what commit hands to history — and makes cancel a single `putImageData`.
 *
 * Everything flows through the {@link RasterSurface} `ctx` seam, so this runs
 * unchanged on the node test stub. Zero React, zero dispatch on the move path.
 */

import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { PlacedSurface, PointerInput, Rect } from '@workbench/canvas-engine/types';
import type { CanvasLayerContract } from '@workbench/types';

import { strokeToPath, type StrokeSamplePoint } from '@workbench/canvas-engine/freehand';
import { intersect, isEmpty, roundOut, union } from '@workbench/canvas-engine/math/rect';

import type { StrokeCommittedEvent, ToolContext } from './tool';

/** Everything a stroke session needs, resolved by the owning tool on pointer-down. */
export interface StrokeSessionConfig {
  ctx: ToolContext;
  /** The layer being painted into (its cache grows with the stroke). */
  layerId: string;
  /** Base stroke diameter (document units). */
  size: number;
  /** Per-stroke opacity in [0, 1]. */
  opacity: number;
  /** Freehand thinning; 0 disables pressure sensitivity. */
  thinning: number;
  /** Fill color (brush only; ignored for the eraser). */
  color: string;
  /**
   * Cache composite operation: `source-over` (brush), `destination-out` (eraser),
   * or `source-atop` (transparency-locked brush — colour only where the layer is
   * already opaque, alpha never grows).
   */
  composite: 'source-over' | 'destination-out' | 'source-atop';
  tool: 'brush' | 'eraser';
  /** Set only when this gesture auto-created its paint layer (for the composed history entry). */
  createdLayer?: { layer: CanvasLayerContract; index: number } | null;
  /**
   * The bounded selection mask to clip the stroke to (resolved once by the tool
   * on pointer-down when a selection exists), as a placed surface in document
   * (= layer-local) space. When set, the paint region is intersected with the
   * mask bounds and the scratch stroke is masked (`destination-in`) before
   * compositing, so pixels outside the selection are never written and the cache
   * only grows within the selection. Absent ⇒ the no-selection hot path.
   */
  clipMask?: PlacedSurface | null;
}

/** The imperative handle a tool drives across a gesture. */
export interface StrokeSession {
  /** Appends coalesced samples and repaints the accumulated stroke. */
  addPoints(inputs: readonly PointerInput[]): void;
  /** Finalizes the stroke and returns the completed edit for its owner to publish. */
  commit(): StrokeCommittedEvent | null;
  /** Restores the pre-stroke pixels and drops the session without an event. */
  cancel(): void;
}

const toSample = (input: PointerInput): StrokeSamplePoint => ({
  pressure: input.pressure,
  x: input.documentPoint.x,
  y: input.documentPoint.y,
});

/**
 * Cache/scratch growth is snapped OUTWARD to this pixel grid. Without it, an
 * outward brush drag extends the paint region by a few pixels on every batch, so
 * the cache (and the scratch) would reallocate + full-copy on every pointer-move.
 * Snapping to a coarse chunk grid means successive small extensions land inside
 * the current padded extent, so growth happens at most once per chunk crossed —
 * O(stroke / CHUNK) reallocations instead of O(batches).
 */
const GROWTH_CHUNK = 64;

/** Rounds a rect OUTWARD to the {@link GROWTH_CHUNK} grid (integer, chunk-aligned). */
const padToChunk = (r: Rect): Rect => {
  const x = Math.floor(r.x / GROWTH_CHUNK) * GROWTH_CHUNK;
  const y = Math.floor(r.y / GROWTH_CHUNK) * GROWTH_CHUNK;
  const right = Math.ceil((r.x + r.width) / GROWTH_CHUNK) * GROWTH_CHUNK;
  const bottom = Math.ceil((r.y + r.height) / GROWTH_CHUNK) * GROWTH_CHUNK;
  return { height: bottom - y, width: right - x, x, y };
};

/** Creates a paint session that grows the target layer's content-sized cache. */
export const createStrokeSession = (config: StrokeSessionConfig): StrokeSession => {
  const { clipMask, color, composite, createdLayer, ctx, layerId, opacity, size, thinning, tool } = config;

  const layers: LayerCacheStore = ctx.layers;
  // A per-frame scratch surface for the filled stroke, sized to the paint region.
  let stroke: RasterSurface | null = null;

  const points: StrokeSamplePoint[] = [];
  // `beforeImageData` holds the pristine (pre-stroke) pixels of `accumRect`, in
  // LAYER-LOCAL coordinates — so it stays valid across a cache growth-realloc
  // (which only shifts the surface origin, not the layer-local geometry).
  let beforeImageData: ImageData | null = null;
  let accumRect: Rect | null = null;

  /** Ensures the scratch surface is at least `w`×`h`. */
  const ensureStroke = (w: number, h: number): RasterSurface => {
    if (!stroke) {
      stroke = ctx.backend.createSurface(w, h);
    } else if (stroke.width < w || stroke.height < h) {
      stroke.resize(Math.max(stroke.width, w), Math.max(stroke.height, h));
    }
    return stroke;
  };

  const paint = (last: boolean): void => {
    if (points.length === 0) {
      return;
    }
    const { bounds, path } = strokeToPath(points, { last, size, thinning }, ctx.createPath2D);
    let dirty: Rect | null = roundOut(bounds);
    // Selection clip: only the region inside the selection can ever change, so
    // bound the dirty/growth region to the mask extent (and skip empty results).
    if (clipMask) {
      dirty = intersect(dirty, clipMask.rect);
    }
    if (!dirty || isEmpty(dirty)) {
      return;
    }
    // Accumulate the dirty union, then round it OUTWARD to a coarse chunk grid.
    // Chunk-padding is the allocation-light seam the plan pins as invariant: an
    // extending drag now grows the cache/scratch at most once per chunk it crosses
    // instead of on every pointer batch. The padded extent may exceed the true
    // content bounds — fine: it is an internal cache extent, so the flush just
    // encodes a slightly larger (mostly-transparent) PNG at a consistent offset,
    // and the reported dirty/before/after all use this same padded region so every
    // consumer stays coherent. When a selection clips the stroke, clamp the padded
    // region back to the mask so growth still never escapes the selection.
    let region = padToChunk(accumRect ? roundOut(union(accumRect, dirty)) : dirty);
    if (clipMask) {
      const clamped = intersect(region, clipMask.rect);
      if (clamped) {
        region = clamped;
      }
    }
    if (isEmpty(region)) {
      return;
    }

    // Grow the cache to cover the (chunk-padded, layer-local) region, preserving
    // existing pixels. Because `region` is chunk-aligned and monotonically growing,
    // `entry.surface` is reallocated at most once per chunk boundary the stroke
    // crosses — not once per batch — keeping the hot path allocation-light. The
    // surface is resized in place (identity preserved).
    const entry = layers.growToRect(layerId, region);
    const target = entry.surface;
    const targetCtx = target.ctx;
    // Surface origin in layer-local space: surface(sx,sy) ↔ local(ox+sx, oy+sy).
    const ox = entry.rect.x;
    const oy = entry.rect.y;

    // 1. Restore the region painted last frame back to pristine "before" pixels,
    //    so recompositing the (larger) stroke doesn't compound its opacity.
    if (accumRect && beforeImageData) {
      targetCtx.putImageData(beforeImageData, accumRect.x - ox, accumRect.y - oy);
    }
    // 2. Recapture the pristine "before" for the grown region (in surface coords).
    beforeImageData = targetCtx.getImageData(region.x - ox, region.y - oy, region.width, region.height);
    accumRect = region;

    // 3. Render the whole accumulated stroke into the scratch surface at full
    //    alpha — one filled polygon, so no self-overlap darkening. The scratch is
    //    region-local: translate the (layer-local) path by -region.origin.
    const scratch = ensureStroke(region.width, region.height);
    const strokeCtx = scratch.ctx;
    strokeCtx.setTransform(1, 0, 0, 1, -region.x, -region.y);
    strokeCtx.clearRect(region.x, region.y, region.width, region.height);
    strokeCtx.globalCompositeOperation = 'source-over';
    strokeCtx.globalAlpha = 1;
    strokeCtx.fillStyle = color;
    strokeCtx.fill(path);

    // 3b. Selection clip: keep only the stroke pixels inside the selection mask.
    //     The mask is a placed surface; draw it at (maskOrigin - regionOrigin) in
    //     the scratch's region-local space.
    if (clipMask) {
      strokeCtx.setTransform(1, 0, 0, 1, 0, 0);
      strokeCtx.globalCompositeOperation = 'destination-in';
      strokeCtx.globalAlpha = 1;
      strokeCtx.drawImage(clipMask.surface.canvas, clipMask.rect.x - region.x, clipMask.rect.y - region.y);
    }

    // 4. Composite the scratch stroke into the cache over the region at the stroke
    //    opacity, using the tool's blend (brush over / eraser out). Both are in
    //    surface coords (region translated by the surface origin).
    targetCtx.save();
    targetCtx.setTransform(1, 0, 0, 1, 0, 0);
    targetCtx.beginPath();
    targetCtx.rect(region.x - ox, region.y - oy, region.width, region.height);
    targetCtx.clip();
    targetCtx.globalAlpha = opacity;
    targetCtx.globalCompositeOperation = composite;
    targetCtx.drawImage(scratch.canvas, region.x - ox, region.y - oy);
    targetCtx.restore();

    // The cache pixels changed THIS frame — bump the version so a version-keyed
    // dependent (the memoized adjusted-surface cache) recomputes over the live
    // stroke. Without this the compositor would keep serving the pre-stroke
    // adjusted surface, and the live stroke would be invisible on an adjusted
    // raster layer until pointer-up (and jump on rect growth). This does NOT
    // touch `thumbnailVersion` (only `notifyLayerPainted`/rasterize do), so
    // thumbnails don't churn mid-stroke.
    layers.publishPixels(layerId);
    ctx.invalidate({ layers: [layerId] });
  };

  return {
    addPoints: (inputs) => {
      for (const input of inputs) {
        points.push(toSample(input));
      }
      paint(false);
    },
    cancel: () => {
      const entry = layers.get(layerId);
      if (entry && accumRect && beforeImageData) {
        entry.surface.ctx.putImageData(beforeImageData, accumRect.x - entry.rect.x, accumRect.y - entry.rect.y);
        // Bump the version so the adjusted-surface memo (which recomputed over the
        // live stroke) re-derives from the RESTORED pixels — otherwise an adjusted
        // raster layer would keep showing the cancelled stroke's adjusted preview.
        layers.publishPixels(layerId);
        ctx.invalidate({ layers: [layerId] });
      }
    },
    commit: () => {
      paint(true);
      const entry = layers.get(layerId);
      if (!entry || !accumRect || !beforeImageData) {
        return null;
      }
      const afterImageData = entry.surface.ctx.getImageData(
        accumRect.x - entry.rect.x,
        accumRect.y - entry.rect.y,
        accumRect.width,
        accumRect.height
      );
      return {
        afterImageData,
        beforeImageData,
        dirtyRect: accumRect,
        layerId,
        tool,
        ...(createdLayer ? { createdLayer } : {}),
      };
    },
  };
};
