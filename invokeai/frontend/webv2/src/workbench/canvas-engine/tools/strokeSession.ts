/**
 * Per-gesture brush/eraser sessions accumulate samples and draw one scratch silhouette, compositing at stroke
 * opacity without overlap darkening. Samples append immediately; the accumulated stroke renders at most once per
 * frame and commit renders the final stroke synchronously. Brush uses source-over or transparency-locked source-atop; erase uses
 * destination-out and is refused under transparency lock.
 *
 * Each preview restores affected pristine pixels before compositing the accumulated stroke. Extend
 * before-snapshots only with newly covered strips, keeping readback proportional to added area. Commit records
 * exact before/after pixels; cancel restores before. Move processing never dispatches.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, PlacedSurface, PointerInput, Rect, Vec2 } from '@workbench/canvas-engine/types';

import {
  createSampleDecimator,
  polygonToPath,
  sampleSpacing,
  strokeOutlineFromSamples,
  strokeToPath,
  type StrokeSamplePoint,
} from '@workbench/canvas-engine/freehand';
import { applyToPoint, getScale, identity, invert, multiply } from '@workbench/canvas-engine/math/mat2d';
import { expand, intersect, isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { getPressureBands } from '@workbench/canvas-engine/pressureBands';

import type { StrokeCommittedEvent, StrokeEdit, ToolContext } from './tool';

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
  /** Edge hardness in [0, 1]: 1 keeps the crisp hot path; lower feathers the silhouette. */
  hardness: number;
  /**
   * Composite mode: brush source-over, eraser destination-out, or transparency-locked brush source-atop preserving
   * alpha.
   */
  composite: 'source-over' | 'destination-out' | 'source-atop';
  tool: 'brush' | 'eraser' | 'shape';
  /** Set only when this gesture auto-created its paint layer (for the composed history entry). */
  createdLayer?: { layer: CanvasLayerContract; anchor: CanvasNodeInsertionAnchor } | null;
  /**
   * Optional document-space selection mask captured at pointerdown. Map bounds locally to limit growth, then
   * destination-in clip scratch so unselected pixels remain untouched.
   */
  clipMask?: PlacedSurface | null;
  /**
   * Optional document-space bbox clip captured once per gesture. Region bounds limit scratch coverage; arbitrary
   * selection masks additionally need shape clipping.
   */
  clipRect?: Rect | null;
  /**
   * Local-to-document layer transform; inverse-map document input/clips because cache, damage and history are
   * local. Absence uses identity.
   */
  layerTransform?: Mat2d | null;
  /** The admitted edit; its undo footprint grows before the stroke's before/after pixels do. */
  edit: Pick<StrokeEdit, 'grow'>;
  /** Called when a scheduled frame render throws, before the error propagates; the owner aborts the stroke. */
  onRenderError?: (error: unknown) => void;
  /** Called once when a frame's growth is refused; the session has already restored the layer. */
  onRefused?: () => void;
}

/** The imperative handle a tool drives across a gesture. */
export interface StrokeSession {
  /** Appends coalesced samples and schedules one repaint of the accumulated stroke for the next frame. */
  addPoints(inputs: readonly PointerInput[]): void;
  /**
   * Renders the final stroke and hands it to `publish`. Returns whether it was published; when nothing was painted,
   * growth was refused, or `publish` declined, the layer is restored first.
   */
  commit(publish: (event: StrokeCommittedEvent) => boolean): boolean;
  /** Restores the pre-stroke pixels and extent and drops the session without an event. */
  cancel(): void;
}

/**
 * Outward chunk alignment limits cache/scratch reallocations to crossed chunks rather than every small pointer
 * extension.
 */
const GROWTH_CHUNK = 64;

/** Cap growth chunks; wider brushes need coarser grids to reduce expensive backing-store reallocations. */
const MAX_GROWTH_CHUNK = 512;

/**
 * Scale growth chunks with brush diameter above {@link GROWTH_CHUNK}; wide brushes cross fewer boundaries while
 * small brushes retain compact history patches.
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
 * Extra slack (layer-local units, the space the stroke rasterizes in) around
 * the changed geometry, covering the antialiased edge and the outward rounding
 * of the bounds themselves.
 */
const CHANGE_MARGIN = 2;

/**
 * Find changed outline bounds via common prefix/suffix, accounting for appended sides, rewritten caps and reversed
 * right-side indices. Add one neighboring vertex each side because quadratic controls influence adjacent midpoint
 * segments; input-radius guesses can miss changes.
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
 * Extend pristine snapshots by copying retained data and reading only new strips, where no stroke has yet
 * composited.
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
  // The document→layer-local boundary. The owning tool refuses degenerate
  // (zero-scale) transforms before opening a session; the identity fallback
  // keeps this total for direct callers.
  const toLocal = (config.layerTransform && invert(config.layerTransform)) ?? identity();
  // Brush geometry is authored in document units; in layer space it scales by
  // the inverse (exact under rotation and uniform scale, the geometric mean
  // under non-uniform — the stroke then stretches with the layer's own pixels).
  const localSize = size * getScale(toLocal);
  // Blur the whole silhouette for uniform, noncompounding softness; sub-quarter-pixel sigma uses the crisp banded
  // path.
  const rawSigma = ((1 - Math.min(1, Math.max(0, config.hardness))) * localSize) / 4;
  const featherSigma = rawSigma < 0.25 ? 0 : rawSigma;
  const featherBleed = Math.ceil(featherSigma * 3);
  const clipMaskLocalRect = clipMask ? roundOut(transformBounds(toLocal, clipMask.rect)) : null;
  const clipRectLocal = clipRect ? roundOut(transformBounds(toLocal, clipRect)) : null;
  // Axis-aligned layers get the rect clip for free from the region clamp; a
  // rotated/sheared layer's clamp is only the AABB, so the exact rect must also
  // be cut out of the scratch like the selection mask is.
  const clipRectNeedsMask = !!clipRect && (toLocal.b !== 0 || toLocal.c !== 0);
  const toSample = (input: PointerInput): StrokeSamplePoint => {
    const local = applyToPoint(toLocal, input.documentPoint);
    return { pressure: input.pressure, x: local.x, y: local.y };
  };
  // A per-frame scratch surface for the filled stroke, sized to the paint region.
  let stroke: RasterSurface | null = null;
  // The feathered copy of the silhouette; allocated only when hardness < 1.
  let soft: RasterSurface | null = null;

  // Raw samples feed only the pressure bands; the outline reads the incrementally decimated samples.
  const points: StrokeSamplePoint[] = [];
  const decimator = createSampleDecimator(sampleSpacing(localSize));
  // Below the travel threshold, preserve the first aim point as a round dab rather than a short capsule.
  const tapCollapseLength = Math.max(2, localSize * 0.25);
  let travel = 0;
  let cancelScheduledFrame: (() => void) | null = null;
  let closed = false;
  // Before pixels cover `accumRect` in stable layer-local coordinates, unaffected by backing-store origin shifts.
  let beforeImageData: ImageData | null = null;
  let accumRect: Rect | null = null;
  let previousPolygon: Vec2[] | null = null;
  const chunk = growthChunk(localSize);
  const original = layers.peek(layerId);
  const originalRect = original ? { ...original.rect } : null;
  // Bytes of before/after pixels the admitted edit already covers.
  let admittedFootprint = 0;

  /** Ensures the scratch surface is at least `w`×`h`. */
  const ensureStroke = (w: number, h: number): RasterSurface => {
    if (!stroke) {
      stroke = ctx.backend.createSurface(w, h);
    } else if (stroke.width < w || stroke.height < h) {
      stroke.resize(Math.max(stroke.width, w), Math.max(stroke.height, h));
    }
    return stroke;
  };

  const ensureSoft = (w: number, h: number): RasterSurface => {
    if (!soft) {
      soft = ctx.backend.createSurface(w, h);
    } else if (soft.width < w || soft.height < h) {
      soft.resize(Math.max(soft.width, w), Math.max(soft.height, h));
    }
    return soft;
  };

  /** Renders the accumulated stroke; false when the undo footprint could not grow to cover it. */
  const paint = (last: boolean): boolean => {
    if (points.length === 0) {
      return true;
    }
    const isTap = travel < tapCollapseLength;
    const effective: readonly StrokeSamplePoint[] = isTap ? [points[0]!] : points;
    const { bounds, path, polygon } = polygonToPath(
      strokeOutlineFromSamples(isTap ? effective : decimator.samples(last), { last, size: localSize, thinning }),
      ctx.createPath2D
    );
    let dirty: Rect | null = roundOut(featherBleed > 0 ? expand(bounds, featherBleed) : bounds);
    // Limit dirty/growth bounds to selection coverage and skip empty intersections.
    if (clipMaskLocalRect) {
      dirty = intersect(dirty, clipMaskLocalRect);
    }
    if (dirty && clipRectLocal) {
      dirty = intersect(dirty, clipRectLocal);
    }
    if (!dirty || isEmpty(dirty)) {
      return true;
    }
    // Outward chunk-padding amortizes growth; dirty regions and history use the same padded extent. Clamp padding
    // to selection bounds, and trim visible content during persistence.
    let region = padToChunk(accumRect ? roundOut(union(accumRect, dirty)) : dirty, chunk);
    if (clipMaskLocalRect) {
      const clamped = intersect(region, clipMaskLocalRect);
      if (clamped) {
        region = clamped;
      }
    }
    if (clipRectLocal) {
      // Region-local scratch clips axis-aligned bbox overflow at its surface boundary.
      const clamped = intersect(region, clipRectLocal);
      if (clamped) {
        region = clamped;
      }
    }
    if (isEmpty(region)) {
      return true;
    }
    // Before and after pixels each cover the region; admit them before either is captured.
    const footprint = region.width * region.height * 8;
    if (footprint > admittedFootprint) {
      if (!config.edit.grow(footprint - admittedFootprint)) {
        return false;
      }
      admittedFootprint = footprint;
    }

    // Grow preserving pixels at chunk crossings while retaining surface-object identity.
    const entry = layers.growToRect(layerId, region);
    const target = entry.surface;
    const targetCtx = target.ctx;
    // Surface origin in layer-local space: surface(sx,sy) ↔ local(ox+sx, oy+sy).
    const ox = entry.rect.x;
    const oy = entry.rect.y;

    const previousRect = accumRect;
    if (!beforeImageData || !previousRect || !sameRect(previousRect, region)) {
      beforeImageData = extendBefore(targetCtx, beforeImageData, previousRect, region, ox, oy);
    }
    accumRect = region;

    // Rewrite only changed silhouette bounds; unchanged pixels already hold the correct composite.
    const moved = previousPolygon ? changedVertexBounds(previousPolygon, polygon) : null;
    const touched =
      previousRect && moved ? intersect(roundOut(expand(moved, CHANGE_MARGIN + featherBleed)), region) : region;
    // No intersection means the change fell outside the paintable region (a clip
    // can do that); fall back to the whole region rather than skip the frame.
    const changed = touched && !isEmpty(touched) ? touched : region;
    previousPolygon = polygon;

    // Restore the changed band from pristine pixels before compositing to avoid compounded opacity. The first
    // frame has nothing to restore.
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

    // Fill the accumulated outline once in region-local scratch, refreshing only changed bounds to avoid overlap
    // seams. Resizing clears scratch and requires a full fill. Pressure-alpha also refreshes fully because partial
    // replacement could split a band and lose earlier coverage.
    const scratchCleared = !previousRect || !sameRect(previousRect, region);
    // The blur reads the whole silhouette; a band-only refill would seam.
    const refresh = scratchCleared || pressureOpacity || featherSigma > 0 ? region : changed;
    const scratch = ensureStroke(region.width, region.height);
    const strokeCtx = scratch.ctx;
    strokeCtx.setTransform(1, 0, 0, 1, -region.x, -region.y);
    strokeCtx.save();
    strokeCtx.beginPath();
    strokeCtx.rect(refresh.x, refresh.y, refresh.width, refresh.height);
    strokeCtx.clip();
    strokeCtx.clearRect(refresh.x, refresh.y, refresh.width, refresh.height);
    strokeCtx.fillStyle = color;

    const tapPoint = effective.length === 1 ? effective[0] : undefined;
    const isSubpixelTap =
      tapPoint !== undefined && bounds.width > 0 && bounds.width < 1 && bounds.height > 0 && bounds.height < 1;

    if (isSubpixelTap) {
      // Subpixel paths may quantize to zero in Skia. Represent a tiny dab in its containing pixel with
      // ellipse-area alpha, preserving subpixel coverage rather than inflating brush size.
      strokeCtx.globalCompositeOperation = 'source-over';
      const pressureAlpha = pressureOpacity ? (getPressureBands([tapPoint])[0]?.alpha ?? 1) : 1;
      strokeCtx.globalAlpha = Math.min(1, ((Math.PI * bounds.width * bounds.height) / 4) * pressureAlpha);
      strokeCtx.fillRect(Math.floor(tapPoint.x), Math.floor(tapPoint.y), 1, 1);
    } else if (pressureOpacity) {
      // Pressure bands erase then replace their footprint at current alpha; newer bands win overlaps without
      // compounding.
      for (const band of getPressureBands(effective)) {
        const bandPath = strokeToPath(band.points, { last, size: localSize, thinning }, ctx.createPath2D).path;

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

    // 4a. Feather before the clips — blurring after would bleed masked pixels
    //     back outside the selection. Without `filter` support the stroke stays crisp.
    let paintSurface = scratch;
    if (featherSigma > 0) {
      const softScratch = ensureSoft(region.width, region.height);
      const softCtx = softScratch.ctx;
      softCtx.setTransform(1, 0, 0, 1, 0, 0);
      softCtx.clearRect(0, 0, softScratch.width, softScratch.height);
      softCtx.filter = `blur(${featherSigma}px)`;
      softCtx.drawImage(scratch.canvas, 0, 0);
      softCtx.filter = 'none';
      paintSurface = softScratch;
    }
    const paintCtx = paintSurface.ctx;

    // Mask only the changed scratch band at maskOrigin-regionOrigin; masking retained pixels again would multiply
    // alpha twice.
    if (clipMask) {
      paintCtx.setTransform(1, 0, 0, 1, 0, 0);
      paintCtx.save();
      paintCtx.beginPath();
      paintCtx.rect(refresh.x - region.x, refresh.y - region.y, refresh.width, refresh.height);
      paintCtx.clip();
      paintCtx.globalCompositeOperation = 'destination-in';
      paintCtx.globalAlpha = 1;
      // The mask is document-space; drawing it through the layer inverse keeps
      // its exact shape clipping even on a rotated or scaled layer.
      const maskMatrix = multiply({ a: 1, b: 0, c: 0, d: 1, e: -region.x, f: -region.y }, toLocal);
      paintCtx.setTransform(maskMatrix.a, maskMatrix.b, maskMatrix.c, maskMatrix.d, maskMatrix.e, maskMatrix.f);
      paintCtx.drawImage(clipMask.surface.canvas, clipMask.rect.x, clipMask.rect.y);
      paintCtx.restore();
    }

    // Rotated/sheared bbox AABBs need exact inverse-mapped rectangular clipping beyond the coarse region clamp.
    if (clipRect && clipRectNeedsMask) {
      paintCtx.setTransform(1, 0, 0, 1, 0, 0);
      paintCtx.save();
      paintCtx.beginPath();
      paintCtx.rect(refresh.x - region.x, refresh.y - region.y, refresh.width, refresh.height);
      paintCtx.clip();
      paintCtx.globalCompositeOperation = 'destination-in';
      paintCtx.globalAlpha = 1;
      const rectMatrix = multiply({ a: 1, b: 0, c: 0, d: 1, e: -region.x, f: -region.y }, toLocal);
      paintCtx.setTransform(rectMatrix.a, rectMatrix.b, rectMatrix.c, rectMatrix.d, rectMatrix.e, rectMatrix.f);
      paintCtx.beginPath();
      paintCtx.rect(clipRect.x, clipRect.y, clipRect.width, clipRect.height);
      paintCtx.fill();
      paintCtx.restore();
    }

    // Composite scratch at stroke opacity with tool blend, clipping to changed bounds even if scratch exceeds the
    // region.
    targetCtx.save();
    targetCtx.setTransform(1, 0, 0, 1, 0, 0);
    targetCtx.beginPath();
    targetCtx.rect(changed.x - ox, changed.y - oy, changed.width, changed.height);
    targetCtx.clip();
    targetCtx.globalAlpha = opacity;
    targetCtx.globalCompositeOperation = composite;
    targetCtx.drawImage(paintSurface.canvas, region.x - ox, region.y - oy);
    targetCtx.restore();

    // Bump cache version with surface-local changed bounds so adjusted surfaces refresh live strokes partially.
    // Keep thumbnail versions unchanged until committed paint/rasterization notifications.
    layers.publishPixels(layerId, {
      height: changed.height,
      width: changed.width,
      x: changed.x - ox,
      y: changed.y - oy,
    });
    // Report layer-local damage; compositor transforms it to screen space, avoiding full-layer resampling as zoom
    // increases.
    ctx.invalidate({ damage: { layerId, rect: changed }, layers: [layerId] });
    return true;
  };

  let settled = false;
  /** Puts the pre-stroke pixels back and returns the cache to its pre-stroke extent, once. */
  const restore = (): void => {
    const entry = layers.peek(layerId);
    if (settled || !entry || !accumRect || !beforeImageData) {
      return;
    }
    settled = true;
    entry.surface.ctx.putImageData(beforeImageData, accumRect.x - entry.rect.x, accumRect.y - entry.rect.y);
    if (originalRect) {
      layers.shrinkToRect(layerId, originalRect);
    } else {
      layers.delete(layerId);
    }
    // The whole restored rect is damage, so adjusted caches discard the live stroke preview.
    const restored = layers.peek(layerId);
    if (restored && !isEmpty(restored.rect)) {
      const visible = intersect(accumRect, restored.rect);
      layers.publishPixels(
        layerId,
        visible ? { ...visible, x: visible.x - restored.rect.x, y: visible.y - restored.rect.y } : null
      );
    }
    ctx.invalidate({ layers: [layerId] });
  };

  const append = (sample: StrokeSamplePoint): void => {
    const previous = points[points.length - 1];
    if (previous && travel < tapCollapseLength) {
      travel += Math.hypot(sample.x - previous.x, sample.y - previous.y);
    }
    points.push(sample);
    decimator.push(sample);
  };

  const unschedule = (): void => {
    cancelScheduledFrame?.();
    cancelScheduledFrame = null;
  };

  const renderFrame = (): void => {
    cancelScheduledFrame = null;
    if (closed) {
      return;
    }
    let painted: boolean;
    try {
      painted = paint(false);
    } catch (error) {
      config.onRenderError?.(error);
      throw error;
    }
    if (!painted) {
      closed = true;
      restore();
      config.onRefused?.();
    }
  };

  return {
    addPoints: (inputs) => {
      if (closed) {
        return;
      }
      for (const input of inputs) {
        append(toSample(input));
      }
      if (cancelScheduledFrame) {
        return;
      }
      let ran = false;
      const cancel = ctx.scheduleFrame(() => {
        ran = true;
        renderFrame();
      });
      if (!ran) {
        cancelScheduledFrame = cancel;
      }
    },
    cancel: () => {
      closed = true;
      unschedule();
      restore();
    },
    commit: (publish) => {
      if (closed) {
        return false;
      }
      closed = true;
      unschedule();
      if (!paint(true)) {
        restore();
        return false;
      }
      const entry = layers.get(layerId);
      if (!entry || !accumRect || !beforeImageData) {
        return false;
      }
      const afterImageData = entry.surface.ctx.getImageData(
        accumRect.x - entry.rect.x,
        accumRect.y - entry.rect.y,
        accumRect.width,
        accumRect.height
      );
      const published = publish({
        afterImageData,
        beforeImageData,
        dirtyRect: accumRect,
        layerId,
        tool,
        ...(createdLayer ? { createdLayer } : {}),
      });
      if (published) {
        settled = true;
      } else {
        restore();
      }
      return published;
    },
  };
};
