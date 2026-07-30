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
 * ## Per-frame restore/recomposite
 *
 * The cache is the live preview, so it must show `before ∪ stroke@opacity` each
 * frame. To recomposite without compounding opacity, every frame first restores
 * the painted region from the captured "before" pixels, then composites the
 * whole accumulated stroke once. This keeps `beforeImageData` exactly equal to
 * the pre-stroke pixels over the final dirty rect — which is what commit hands
 * to history — and makes cancel a single `putImageData`.
 *
 * The "before" snapshot is EXTENDED rather than recaptured (see
 * {@link extendBefore}). Re-reading the whole region every frame was by far the
 * most expensive thing this module did — a full-region `getImageData` costs
 * roughly ten times the rest of the frame put together, and it grows with the
 * area painted so far, which is exactly why covering a large area with a big
 * brush degraded the longer you painted. Since the stroke is never composited
 * outside the accumulated rect, the pixels the region gains are still pristine
 * and can simply be read as strips and stitched onto the snapshot we already
 * hold, leaving per-frame readback proportional to new area rather than total.
 *
 * Everything flows through the {@link RasterSurface} `ctx` seam, so this runs
 * unchanged on the node test stub. Zero React, zero dispatch on the move path.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { PlacedSurface, PointerInput, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { strokeToPath, type StrokeSamplePoint } from '@workbench/canvas-engine/freehand';
import { expand, intersect, isEmpty, roundOut, union } from '@workbench/canvas-engine/math/rect';
import { getPressureBands } from '@workbench/canvas-engine/pressureBands';

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
  /**
   * Whether pen pressure modulates alpha along the stroke. Costs a full-region scratch
   * refill per frame (see the band fill below), so it is opt-in.
   */
  pressureOpacity: boolean;
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
  /**
   * A document-space rectangle the stroke may not paint outside — the legacy
   * "clip strokes to bbox" setting, resolved ONCE per gesture like `clipMask`.
   *
   * Unlike the selection mask this needs no compositing pass: the scratch is
   * region-local and the region is clamped to this rect, so path geometry beyond
   * it falls outside the scratch surface and is never drawn. A mask needs the
   * extra `destination-in` only because it is an arbitrary SHAPE within its rect.
   *
   * Absent/`null` ⇒ unclipped (the default, and the hot path).
   */
  clipRect?: Rect | null;
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

/**
 * Upper bound on the growth chunk. Reallocating the cache surface costs roughly
 * the same as a full-region readback — it is dominated by re-allocating the
 * backing canvas, not by preserving the pixels — so a wide brush wants a coarse
 * grid to cross fewer boundaries.
 */
const MAX_GROWTH_CHUNK = 512;

/**
 * Fraction of the brush diameter used as the growth chunk, above the
 * {@link GROWTH_CHUNK} floor.
 *
 * Scaling with the brush keeps the padding a constant FRACTION of the stroke
 * rather than a constant number of pixels. A fixed 64px grid means a 1200px
 * brush reallocates every 64px of travel, and each reallocation costs about as
 * much as a whole frame; scaled, it reallocates ~5× less often for the same
 * relative overhead. Small brushes keep the fine grid, so their history patches
 * (sized from this padded rect) stay small.
 */
const GROWTH_CHUNK_RATIO = 0.25;

/** The growth-grid pitch for a stroke of diameter `size`. */
const growthChunk = (size: number): number =>
  Math.min(
    MAX_GROWTH_CHUNK,
    Math.max(GROWTH_CHUNK, Math.round((size * GROWTH_CHUNK_RATIO) / GROWTH_CHUNK) * GROWTH_CHUNK)
  );

/** Rounds a rect OUTWARD to the growth grid for `chunk` (integer, chunk-aligned). */
const padToChunk = (r: Rect, chunk: number): Rect => {
  const x = Math.floor(r.x / chunk) * chunk;
  const y = Math.floor(r.y / chunk) * chunk;
  const right = Math.ceil((r.x + r.width) / chunk) * chunk;
  const bottom = Math.ceil((r.y + r.height) / chunk) * chunk;
  return { height: bottom - y, width: right - x, x, y };
};

/** The 2D context flavour a {@link RasterSurface} exposes. */
type SurfaceContext = RasterSurface['ctx'];

/**
 * Extra slack (document units) around the changed geometry, covering the
 * antialiased edge and the outward rounding of the bounds themselves.
 */
const CHANGE_MARGIN = 2;

/**
 * Bounds of the vertices where two outline rings disagree, or `null` when they
 * are identical.
 *
 * This decides how little of the cache a frame has to rewrite, and it is
 * answered from the geometry itself rather than inferred from the input.
 * Bounding it by "a brush radius around the newest samples" is the intuitive
 * approach and it is wrong: the ring is
 * `left ++ endCap ++ reverse(right) ++ startCap`, so extending the stroke
 * appends to `left`, rewrites `endCap`, and — because the right side is
 * REVERSED — shifts the index of every vertex after it. Comparing the rings
 * directly costs one linear scan and assumes nothing about how far a new sample
 * can reach.
 *
 * The scan takes the longest common prefix and the longest common suffix; what
 * lies between is everything that moved, appeared or vanished. One vertex of
 * slack is added on each side, because a vertex is the control point of the
 * quadratic spanning its two neighbouring edge midpoints, so its influence on
 * the rendered curve reaches one vertex either way.
 */
const changedVertexBounds = (previous: readonly Vec2[], current: readonly Vec2[]): Rect | null => {
  const shortest = Math.min(previous.length, current.length);
  let head = 0;
  while (head < shortest && previous[head]!.x === current[head]!.x && previous[head]!.y === current[head]!.y) {
    head++;
  }
  let tail = 0;
  while (
    tail < shortest - head &&
    previous[previous.length - 1 - tail]!.x === current[current.length - 1 - tail]!.x &&
    previous[previous.length - 1 - tail]!.y === current[current.length - 1 - tail]!.y
  ) {
    tail++;
  }
  const from = Math.max(0, head - 1);
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  const cover = (ring: readonly Vec2[]): void => {
    for (let i = from; i < ring.length && i <= ring.length - tail; i++) {
      const p = ring[i]!;
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
  };
  cover(previous);
  cover(current);
  if (minX > maxX) {
    return null;
  }
  return { height: maxY - minY, width: maxX - minX, x: minX, y: minY };
};

/** Whether two rects are identical. */
const sameRect = (a: Rect, b: Rect): boolean =>
  a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;

/**
 * Grows the pristine "before" snapshot from `previous` to cover `region`,
 * reading ONLY the newly-added area from the surface.
 *
 * The stroke has never been composited outside `previous` — every frame
 * composites within the accumulated rect — so the surface still holds pristine
 * pixels there and they can be read directly. Carrying the existing snapshot
 * forward by copy, instead of re-reading the whole region, is what keeps the
 * per-frame cost proportional to new area rather than total painted area.
 */
const extendBefore = (
  surfaceCtx: SurfaceContext,
  existing: ImageData | null,
  previous: Rect | null,
  region: Rect,
  originX: number,
  originY: number
): ImageData => {
  if (!existing || !previous || isEmpty(previous)) {
    return surfaceCtx.getImageData(region.x - originX, region.y - originY, region.width, region.height);
  }
  const grown = surfaceCtx.createImageData(region.width, region.height);

  /** Blits `source` (covering `rect`) into `grown` at its place in `region`. */
  const blit = (source: ImageData, rect: Rect): void => {
    const offsetX = rect.x - region.x;
    const offsetY = rect.y - region.y;
    const rowBytes = rect.width * 4;
    for (let row = 0; row < rect.height; row++) {
      const from = row * rowBytes;
      grown.data.set(source.data.subarray(from, from + rowBytes), ((row + offsetY) * region.width + offsetX) * 4);
    }
  };

  // Carry the pixels we already hold forward untouched...
  blit(existing, previous);
  // ...and read only the frame the growth exposed. The stroke has never been
  // composited out here, so the surface still holds pristine pixels.
  for (const strip of surroundingStrips(previous, region)) {
    blit(surfaceCtx.getImageData(strip.x - originX, strip.y - originY, strip.width, strip.height), strip);
  }
  return grown;
};

/**
 * Decomposes `outer` minus `inner` into up to four disjoint rects (top, bottom,
 * left, right), assuming `inner` is contained in `outer`.
 */
const surroundingStrips = (inner: Rect, outer: Rect): Rect[] => {
  const innerBottom = inner.y + inner.height;
  const outerBottom = outer.y + outer.height;
  const candidates: Rect[] = [
    { height: inner.y - outer.y, width: outer.width, x: outer.x, y: outer.y },
    { height: outerBottom - innerBottom, width: outer.width, x: outer.x, y: innerBottom },
    { height: inner.height, width: inner.x - outer.x, x: outer.x, y: inner.y },
    {
      height: inner.height,
      width: outer.x + outer.width - (inner.x + inner.width),
      x: inner.x + inner.width,
      y: inner.y,
    },
  ];
  return candidates.filter((r) => !isEmpty(r));
};

/** Creates a paint session that grows the target layer's content-sized cache. */
export const createStrokeSession = (config: StrokeSessionConfig): StrokeSession => {
  const {
    clipMask,
    clipRect,
    color,
    composite,
    createdLayer,
    ctx,
    layerId,
    opacity,
    pressureOpacity,
    size,
    thinning,
    tool,
  } = config;

  const layers: LayerCacheStore = ctx.layers;
  // A per-frame scratch surface for the filled stroke, sized to the paint region.
  let stroke: RasterSurface | null = null;

  const points: StrokeSamplePoint[] = [];
  // `beforeImageData` holds the pristine (pre-stroke) pixels of `accumRect`, in
  // LAYER-LOCAL coordinates — so it stays valid across a cache growth-realloc
  // (which only shifts the surface origin, not the layer-local geometry).
  let beforeImageData: ImageData | null = null;
  let accumRect: Rect | null = null;
  let previousPolygon: Vec2[] | null = null;
  const chunk = growthChunk(size);

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
    const { bounds, path, polygon } = strokeToPath(points, { last, size, thinning }, ctx.createPath2D);
    let dirty: Rect | null = roundOut(bounds);
    // Selection clip: only the region inside the selection can ever change, so
    // bound the dirty/growth region to the mask extent (and skip empty results).
    if (clipMask) {
      dirty = intersect(dirty, clipMask.rect);
    }
    if (dirty && clipRect) {
      dirty = intersect(dirty, clipRect);
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
    let region = padToChunk(accumRect ? roundOut(union(accumRect, dirty)) : dirty, chunk);
    if (clipMask) {
      const clamped = intersect(region, clipMask.rect);
      if (clamped) {
        region = clamped;
      }
    }
    if (clipRect) {
      // Clamping the region IS the bbox clip: the scratch is sized to the region
      // and the path is drawn translated into it, so anything beyond the rect
      // lands outside the surface and never reaches the cache.
      const clamped = intersect(region, clipRect);
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

    // 1. Extend the pristine "before" snapshot over whatever the region just
    //    gained, reading only the newly-exposed frame rather than re-reading
    //    everything painted so far.
    const previousRect = accumRect;
    if (!beforeImageData || !previousRect || !sameRect(previousRect, region)) {
      beforeImageData = extendBefore(targetCtx, beforeImageData, previousRect, region, ox, oy);
    }
    accumRect = region;

    // 2. Work out where the silhouette actually moved since the last frame, and
    //    confine the rest of the frame to it. Outside this band the cache already
    //    holds the correct composite and the shape has not changed, so redoing it
    //    would only cost time — and rewriting the whole accumulated region every
    //    frame is what made a long stroke get slower the longer it got.
    const moved = previousPolygon ? changedVertexBounds(previousPolygon, polygon) : null;
    const touched = previousRect && moved ? intersect(roundOut(expand(moved, CHANGE_MARGIN)), region) : region;
    // No intersection means the change fell outside the paintable region (a clip
    // can do that); fall back to the whole region rather than skip the frame.
    const changed = touched && !isEmpty(touched) ? touched : region;
    previousPolygon = polygon;

    // 3. Restore that band to pristine "before" pixels, so recompositing the
    //    stroke over it doesn't compound its opacity. `putImageData`'s dirty-rect
    //    arguments write only this window out of the full-region snapshot. On the
    //    opening frame nothing has been composited yet, so there is nothing to undo.
    if (previousRect) {
      targetCtx.putImageData(
        beforeImageData,
        region.x - ox,
        region.y - oy,
        changed.x - region.x,
        changed.y - region.y,
        changed.width,
        changed.height
      );
    }

    // 4. Render the accumulated stroke into the scratch surface at full alpha.
    //    The path is filled in ONE pass, so overlapping parts of a stroke union
    //    rather than compounding — filling it in pieces across frames would let
    //    the antialiased edges of successive batches blend into each other and
    //    leave seams. The scratch is region-local: translate the (layer-local)
    //    path by -region.origin.
    //
    //    Only the changed band is refreshed. The scratch persists between
    //    frames, and outside that band the silhouette is by definition the same
    //    one already sitting there — so re-clearing and re-filling the whole
    //    region was redrawing pixels to their existing values. A region change
    //    resizes the scratch, which clears it, so that case still refills whole.
    //
    //    Pressure-opacity forces the whole region to refresh instead. Bands are painted as
    //    replace, so a band straddling the changed strip would be half-punched by a partial
    //    refill and lose its earlier half. Refilling whole keeps the band sequence intact,
    //    at the cost of the per-frame saving above — which is why it stays opt-in.
    const scratchCleared = !previousRect || !sameRect(previousRect, region);
    const refresh = scratchCleared || pressureOpacity ? region : changed;
    const scratch = ensureStroke(region.width, region.height);
    const strokeCtx = scratch.ctx;
    strokeCtx.setTransform(1, 0, 0, 1, -region.x, -region.y);
    strokeCtx.save();
    strokeCtx.beginPath();
    strokeCtx.rect(refresh.x, refresh.y, refresh.width, refresh.height);
    strokeCtx.clip();
    strokeCtx.clearRect(refresh.x, refresh.y, refresh.width, refresh.height);
    strokeCtx.fillStyle = color;

    if (pressureOpacity) {
      // Each band replaces its own footprint rather than blending into it: punch the band's
      // outline out of whatever is already there, then fill it at the band's alpha. Blending
      // would compound alpha wherever bands overlap — the exact darkening the single
      // full-alpha fill below exists to prevent. Replacing means the later (newer) band wins
      // in the overlap, which is the pressure the user is applying now.
      for (const band of getPressureBands(points)) {
        const bandPath = strokeToPath(band.points, { last, size, thinning }, ctx.createPath2D).path;

        strokeCtx.globalCompositeOperation = 'destination-out';
        strokeCtx.globalAlpha = 1;
        strokeCtx.fill(bandPath);
        strokeCtx.globalCompositeOperation = 'source-over';
        strokeCtx.globalAlpha = band.alpha;
        strokeCtx.fill(bandPath);
      }
    } else {
      strokeCtx.globalCompositeOperation = 'source-over';
      strokeCtx.globalAlpha = 1;
      strokeCtx.fill(path);
    }

    strokeCtx.restore();

    // 4b. Selection clip: keep only the stroke pixels inside the selection mask.
    //     The mask is a placed surface; draw it at (maskOrigin - regionOrigin) in
    //     the scratch's region-local space. Confined to the same band — masking
    //     already-masked pixels a second time would multiply their alpha again.
    if (clipMask) {
      strokeCtx.setTransform(1, 0, 0, 1, 0, 0);
      strokeCtx.save();
      strokeCtx.beginPath();
      strokeCtx.rect(refresh.x - region.x, refresh.y - region.y, refresh.width, refresh.height);
      strokeCtx.clip();
      strokeCtx.globalCompositeOperation = 'destination-in';
      strokeCtx.globalAlpha = 1;
      strokeCtx.drawImage(clipMask.surface.canvas, clipMask.rect.x - region.x, clipMask.rect.y - region.y);
      strokeCtx.restore();
    }

    // 5. Composite the scratch stroke into the cache at the stroke opacity, using
    //    the tool's blend (brush over / eraser out), clipped to the band this
    //    batch changed. The clip also confines the whole-canvas `drawImage` to
    //    the region, which the scratch may exceed after a resize.
    targetCtx.save();
    targetCtx.setTransform(1, 0, 0, 1, 0, 0);
    targetCtx.beginPath();
    targetCtx.rect(changed.x - ox, changed.y - oy, changed.width, changed.height);
    targetCtx.clip();
    targetCtx.globalAlpha = opacity;
    targetCtx.globalCompositeOperation = composite;
    targetCtx.drawImage(scratch.canvas, region.x - ox, region.y - oy);
    targetCtx.restore();

    // The cache pixels changed THIS frame — bump the version, WITH the band they
    // changed in, so a version-keyed dependent (the memoized adjusted-surface
    // cache) recomputes over the live stroke without rebuilding from the whole
    // layer. Without the bump the compositor would keep serving the pre-stroke
    // adjusted surface, and the live stroke would be invisible on an adjusted
    // raster layer until pointer-up (and jump on rect growth). The band is in
    // SURFACE-local coordinates, the space a derived surface is built in. Note
    // it is the derived surfaces that need this — the stroke's own scratch and
    // cache writes are already bounded above. This does NOT
    // touch `thumbnailVersion` (only `notifyLayerPainted`/rasterize do), so
    // thumbnails don't churn mid-stroke.
    layers.publishPixels(layerId, {
      height: changed.height,
      width: changed.width,
      x: changed.x - ox,
      y: changed.y - oy,
    });
    // Report the region as damage so the frame repaints only these pixels
    // instead of re-resampling every doc-sized layer to screen scale — the cost
    // that grows with zoom. `region` is layer-local, which is exactly what the
    // compositor wants: it carries it to the screen through the layer's own
    // matrix, so this stays correct on a moved or scaled layer.
    ctx.invalidate({ damage: { layerId, rect: changed }, layers: [layerId] });
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
        // The restore spans everything the gesture touched, so this reports the
        // whole accumulated rect rather than any one frame's band.
        layers.publishPixels(layerId, {
          height: accumRect.height,
          width: accumRect.width,
          x: accumRect.x - entry.rect.x,
          y: accumRect.y - entry.rect.y,
        });
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
