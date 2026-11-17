/** One scratch silhouette composited at stroke opacity, without overlap darkening, at most once per frame.
 Previews restore pristine pixels first; before-snapshots extend only by newly covered strips. */

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

export interface StrokeSessionConfig {
  ctx: ToolContext;
  layerId: string;
  /** Document units. */
  size: number;
  opacity: number;
  /** 0 disables pressure sensitivity. */
  thinning: number;
  /** Opt-in: costs a full-region scratch refill per frame. */
  pressureOpacity: boolean;
  /** Ignored by the eraser. */
  color: string;
  /** 1 keeps the crisp hot path; lower feathers the silhouette. */
  hardness: number;
  /** `source-atop` is the transparency-locked brush. */
  composite: 'source-over' | 'destination-out' | 'source-atop';
  tool: 'brush' | 'eraser' | 'shape';
  /** Set only when this gesture auto-created its paint layer. */
  createdLayer?: { layer: CanvasLayerContract; anchor: CanvasNodeInsertionAnchor } | null;
  /** Document space, captured at pointerdown. */
  clipMask?: PlacedSurface | null;
  /** Document space, captured once per gesture. */
  clipRect?: Rect | null;
  /** Cache, damage and history are layer-local, so document input is inverse-mapped; absent is identity. */
  layerTransform?: Mat2d | null;
  /** Its undo footprint grows before the stroke's pixels do. */
  edit: Pick<StrokeEdit, 'grow'>;
  /** Called before the error propagates; the owner aborts the stroke. */
  onRenderError?: (error: unknown) => void;
  /** The session has already restored the layer. */
  onRefused?: () => void;
}

export interface StrokeSession {
  addPoints(inputs: readonly PointerInput[]): void;
  /** Returns whether it was published; otherwise the layer is restored first. */
  commit(publish: (event: StrokeCommittedEvent) => boolean): boolean;
  cancel(): void;
}

/** Chunk alignment reallocates cache and scratch only when a stroke crosses a chunk. */
const GROWTH_CHUNK = 64;

const MAX_GROWTH_CHUNK = 512;

/** Wide brushes cross fewer boundaries while small ones keep compact history patches. */
const GROWTH_CHUNK_RATIO = 0.25;

const growthChunk = (size: number): number =>
  Math.min(
    MAX_GROWTH_CHUNK,
    Math.max(GROWTH_CHUNK, Math.round((size * GROWTH_CHUNK_RATIO) / GROWTH_CHUNK) * GROWTH_CHUNK)
  );

const padToChunk = (r: Rect, chunk: number): Rect => {
  const x = Math.floor(r.x / chunk) * chunk;
  const y = Math.floor(r.y / chunk) * chunk;
  const right = Math.ceil((r.x + r.width) / chunk) * chunk;
  const bottom = Math.ceil((r.y + r.height) / chunk) * chunk;
  return { height: bottom - y, width: right - x, x, y };
};

type SurfaceContext = RasterSurface['ctx'];

/** Layer-local slack covering the antialiased edge and outward rounding. */
const CHANGE_MARGIN = 2;

/** Adds one neighbor vertex each side: quadratic controls reach adjacent midpoint segments. */
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

const sameRect = (a: Rect, b: Rect): boolean =>
  a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;

/** Reads only the new strips, where no stroke composited yet. */
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

  const blit = (source: ImageData, rect: Rect): void => {
    const offsetX = rect.x - region.x;
    const offsetY = rect.y - region.y;
    const rowBytes = rect.width * 4;
    for (let row = 0; row < rect.height; row++) {
      const from = row * rowBytes;
      grown.data.set(source.data.subarray(from, from + rowBytes), ((row + offsetY) * region.width + offsetX) * 4);
    }
  };

  blit(existing, previous);
  for (const strip of surroundingStrips(previous, region)) {
    blit(surfaceCtx.getImageData(strip.x - originX, strip.y - originY, strip.width, strip.height), strip);
  }
  return grown;
};

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
  // Owning tools refuse degenerate transforms; the identity fallback keeps direct callers total.
  const toLocal = (config.layerTransform && invert(config.layerTransform)) ?? identity();
  // Exact under rotation and uniform scale, the geometric mean under non-uniform scale.
  const localSize = size * getScale(toLocal);
  // Sub-quarter-pixel sigma uses the crisp banded path.
  const rawSigma = ((1 - Math.min(1, Math.max(0, config.hardness))) * localSize) / 4;
  const featherSigma = rawSigma < 0.25 ? 0 : rawSigma;
  const featherBleed = Math.ceil(featherSigma * 3);
  const clipMaskLocalRect = clipMask ? roundOut(transformBounds(toLocal, clipMask.rect)) : null;
  const clipRectLocal = clipRect ? roundOut(transformBounds(toLocal, clipRect)) : null;
  // A rotated or sheared layer's clamp is only the AABB, so the exact rect is masked like a selection.
  const clipRectNeedsMask = !!clipRect && (toLocal.b !== 0 || toLocal.c !== 0);
  const toSample = (input: PointerInput): StrokeSamplePoint => {
    const local = applyToPoint(toLocal, input.documentPoint);
    return { pressure: input.pressure, x: local.x, y: local.y };
  };
  let stroke: RasterSurface | null = null;
  let soft: RasterSurface | null = null;

  // Raw samples feed only the pressure bands; the outline reads decimated samples.
  const points: StrokeSamplePoint[] = [];
  const decimator = createSampleDecimator(sampleSpacing(localSize));
  // Below the travel threshold, keep the first aim point as a round dab rather than a short capsule.
  const tapCollapseLength = Math.max(2, localSize * 0.25);
  let travel = 0;
  let cancelScheduledFrame: (() => void) | null = null;
  let closed = false;
  // Stable layer-local coordinates, unaffected by backing-store origin shifts.
  let beforeImageData: ImageData | null = null;
  let accumRect: Rect | null = null;
  let previousPolygon: Vec2[] | null = null;
  const chunk = growthChunk(localSize);
  const original = layers.peek(layerId);
  const originalRect = original ? { ...original.rect } : null;
  let admittedFootprint = 0;

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

  /** False when the undo footprint could not grow to cover it. */
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
    if (clipMaskLocalRect) {
      dirty = intersect(dirty, clipMaskLocalRect);
    }
    if (dirty && clipRectLocal) {
      dirty = intersect(dirty, clipRectLocal);
    }
    if (!dirty || isEmpty(dirty)) {
      return true;
    }
    // Dirty regions and history share the chunk-padded extent.
    let region = padToChunk(accumRect ? roundOut(union(accumRect, dirty)) : dirty, chunk);
    if (clipMaskLocalRect) {
      const clamped = intersect(region, clipMaskLocalRect);
      if (clamped) {
        region = clamped;
      }
    }
    if (clipRectLocal) {
      const clamped = intersect(region, clipRectLocal);
      if (clamped) {
        region = clamped;
      }
    }
    if (isEmpty(region)) {
      return true;
    }
    // Admit before and after pixels before either is captured.
    const footprint = region.width * region.height * 8;
    if (footprint > admittedFootprint) {
      if (!config.edit.grow(footprint - admittedFootprint)) {
        return false;
      }
      admittedFootprint = footprint;
    }

    const entry = layers.growToRect(layerId, region);
    const target = entry.surface;
    const targetCtx = target.ctx;
    // surface(sx,sy) ↔ local(ox+sx, oy+sy).
    const ox = entry.rect.x;
    const oy = entry.rect.y;

    const previousRect = accumRect;
    if (!beforeImageData || !previousRect || !sameRect(previousRect, region)) {
      beforeImageData = extendBefore(targetCtx, beforeImageData, previousRect, region, ox, oy);
    }
    accumRect = region;

    const moved = previousPolygon ? changedVertexBounds(previousPolygon, polygon) : null;
    const touched =
      previousRect && moved ? intersect(roundOut(expand(moved, CHANGE_MARGIN + featherBleed)), region) : region;
    // A clip can move the change outside the region; repaint the whole region rather than skip the frame.
    const changed = touched && !isEmpty(touched) ? touched : region;
    previousPolygon = polygon;

    // Restore from pristine pixels first to avoid compounded opacity.
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

    // Partial refills would split a pressure band and lose earlier coverage, so pressure alpha refills fully.
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
      // Subpixel paths can quantize to zero in Skia; draw a tiny dab with ellipse-area alpha instead.
      strokeCtx.globalCompositeOperation = 'source-over';
      const pressureAlpha = pressureOpacity ? (getPressureBands([tapPoint])[0]?.alpha ?? 1) : 1;
      strokeCtx.globalAlpha = Math.min(1, ((Math.PI * bounds.width * bounds.height) / 4) * pressureAlpha);
      strokeCtx.fillRect(Math.floor(tapPoint.x), Math.floor(tapPoint.y), 1, 1);
    } else if (pressureOpacity) {
      // Newer bands win overlaps uncompounded.
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

    // Feather before clipping, or the blur bleeds masked pixels back outside the selection.
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

    // Re-masking retained pixels would double-apply alpha.
    if (clipMask) {
      paintCtx.setTransform(1, 0, 0, 1, 0, 0);
      paintCtx.save();
      paintCtx.beginPath();
      paintCtx.rect(refresh.x - region.x, refresh.y - region.y, refresh.width, refresh.height);
      paintCtx.clip();
      paintCtx.globalCompositeOperation = 'destination-in';
      paintCtx.globalAlpha = 1;
      const maskMatrix = multiply({ a: 1, b: 0, c: 0, d: 1, e: -region.x, f: -region.y }, toLocal);
      paintCtx.setTransform(maskMatrix.a, maskMatrix.b, maskMatrix.c, maskMatrix.d, maskMatrix.e, maskMatrix.f);
      paintCtx.drawImage(clipMask.surface.canvas, clipMask.rect.x, clipMask.rect.y);
      paintCtx.restore();
    }

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

    targetCtx.save();
    targetCtx.setTransform(1, 0, 0, 1, 0, 0);
    targetCtx.beginPath();
    targetCtx.rect(changed.x - ox, changed.y - oy, changed.width, changed.height);
    targetCtx.clip();
    targetCtx.globalAlpha = opacity;
    targetCtx.globalCompositeOperation = composite;
    targetCtx.drawImage(paintSurface.canvas, region.x - ox, region.y - oy);
    targetCtx.restore();

    // Thumbnail versions stay unchanged until committed paint notifications.
    layers.publishPixels(layerId, {
      height: changed.height,
      width: changed.width,
      x: changed.x - ox,
      y: changed.y - oy,
    });
    ctx.invalidate({ damage: { layerId, rect: changed }, layers: [layerId] });
    return true;
  };

  let settled = false;
  /** Idempotent. */
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
    // The whole restored rect is damage, so adjusted caches discard the live preview.
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
