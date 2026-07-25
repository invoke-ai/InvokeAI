/**
 * The floating selection: pixels lifted out of a layer and held in flight.
 *
 * Dragging inside the marching ants cuts the selected pixels off the layer into
 * a transient float that follows the pointer, leaving a hole behind. The float
 * stays live — re-draggable, transformable — until it is committed (baked back
 * into the same layer as ONE undo entry) or cancelled (the hole refilled). This
 * module owns the geometry and the pixel extraction; `FloatingSelectionController`
 * owns the lifecycle, history, and persistence around it.
 *
 * ## Coordinate spaces
 *
 * The distinction that makes this work: **the selection mask is in DOCUMENT
 * space, the layer cache is in LAYER-LOCAL space.** They coincide only when the
 * layer's transform is the identity — which the existing `selectionOps` fill and
 * erase happen to assume. A float cannot assume it: it lands back in the layer
 * cache, so it must be exact under rotation and scale.
 *
 * So the float keeps its pixels and its live transform in **layer-local** space:
 *
 * - The lift maps the mask into layer-local through the inverse layer matrix,
 *   which resamples the MASK (cheap, and only its coverage matters) rather than
 *   the content.
 * - The lifted content is never resampled — a plain blit out and, for a pure
 *   translate at whole pixels, a plain blit back.
 * - The move tool converts its document-space pointer delta into layer-local
 *   before applying it (see {@link documentDeltaToLocal}).
 *
 * Zero React, zero import-time side effects.
 */

import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Mat2d, PlacedSurface, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint, invert, multiply } from '@workbench/canvas-engine/math/mat2d';
import { intersect, isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';

/** A live floating selection. */
export interface FloatingSelection {
  /** The layer the pixels were cut from, and the only layer they can bake into. */
  readonly layerId: string;
  /** The lifted pixels, placed at their lift-time rect in LAYER-LOCAL space. */
  readonly pixels: PlacedSurface;
  /**
   * `pixels` with the layer's display-only effects baked in (control
   * transparency / raster adjustments), or `null` when the layer has none. The
   * compositor draws THIS; the bake writes back the untouched `pixels`, so the
   * effect is never burned into the document. Computed once at lift: the effects
   * are per-pixel, so the float's transform cannot change them, and a change to
   * the layer's display properties cancels the float outright.
   */
  readonly display: RasterSurface | null;
  /** The layer cache's pre-lift pixels over the cut region, for cancel and the undo entry. */
  readonly before: { rect: Rect; data: ImageData };
  /** The selection mask at lift time, in DOCUMENT space — the ants follow it through the float's transform. */
  readonly mask: PlacedSurface;
  /** The float's live transform, in LAYER-LOCAL space. Identity at lift. */
  transform: LayerTransform;
}

/** Inputs to {@link liftSelectedPixels}. */
export interface LiftSelectedPixelsParams {
  backend: RasterBackend;
  /** The layer's cache surface and its layer-local content rect. */
  cache: PlacedSurface;
  /** The selection mask (alpha inside) and its document-space rect. */
  mask: PlacedSurface;
  /** The layer's local→document matrix. */
  layerMatrix: Mat2d;
}

/** What a lift produces, all in LAYER-LOCAL space. */
export interface LiftedPixels {
  /** The masked copy of the layer's content, sized to the cut region. */
  pixels: PlacedSurface;
  /**
   * The selection mask projected into layer-local space over the same region —
   * the stencil the caller punches the hole with, so the cut and the copy agree
   * to the pixel.
   */
  localMask: PlacedSurface;
}

/**
 * Converts a DOCUMENT-space delta into the layer's LOCAL space. Only the linear
 * part of the layer matrix applies — a delta is a vector, not a point, so the
 * translation must not be added.
 */
export const documentDeltaToLocal = (layerMatrix: Mat2d, delta: Vec2): Vec2 => {
  const inverse = invert(layerMatrix);
  if (!inverse) {
    return { x: 0, y: 0 };
  }
  const origin = applyToPoint(inverse, { x: 0, y: 0 });
  const moved = applyToPoint(inverse, delta);
  return { x: moved.x - origin.x, y: moved.y - origin.y };
};

/**
 * The float's transform expressed in DOCUMENT space: `L · F · L⁻¹`, where `L` is
 * the layer matrix and `F` the float's layer-local transform. Used to carry the
 * document-space selection mask along with the float on commit, so the marching
 * ants end up around the moved pixels.
 */
export const floatDocumentMatrix = (layerMatrix: Mat2d, floatMatrix: Mat2d): Mat2d | null => {
  const inverse = invert(layerMatrix);
  return inverse ? multiply(multiply(layerMatrix, floatMatrix), inverse) : null;
};

/**
 * Draws a document-space placed surface into a fresh surface covering `region`
 * in LAYER-LOCAL space, projecting through `inverseLayerMatrix`.
 */
const projectIntoLocal = (
  backend: RasterBackend,
  source: PlacedSurface,
  inverseLayerMatrix: Mat2d,
  region: Rect
): PlacedSurface => {
  const surface = backend.createSurface(region.width, region.height);
  const ctx = surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  // document → layer-local → region-local, so the source's own document-space
  // origin can be passed straight to `drawImage`.
  const toRegion = multiply({ a: 1, b: 0, c: 0, d: 1, e: -region.x, f: -region.y }, inverseLayerMatrix);
  ctx.setTransform(toRegion.a, toRegion.b, toRegion.c, toRegion.d, toRegion.e, toRegion.f);
  ctx.drawImage(source.surface.canvas, source.rect.x, source.rect.y);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return { rect: region, surface };
};

/**
 * Copies the layer's content within the selection into a detached surface, and
 * returns the layer-local stencil that produced it. Returns `null` when the
 * selection does not overlap the layer's content — there would be nothing to
 * float.
 *
 * This does NOT modify the cache: punching the hole is the caller's step, done
 * with the returned `localMask` so the two agree exactly.
 */
export const liftSelectedPixels = ({
  backend,
  cache,
  layerMatrix,
  mask,
}: LiftSelectedPixelsParams): LiftedPixels | null => {
  const inverseLayerMatrix = invert(layerMatrix);
  if (!inverseLayerMatrix || isEmpty(cache.rect) || isEmpty(mask.rect)) {
    return null;
  }
  // The selection's extent in layer-local space, clipped to what the layer holds.
  const maskLocalBounds = roundOut(transformBounds(inverseLayerMatrix, mask.rect));
  const region = intersect(maskLocalBounds, cache.rect);
  if (!region || isEmpty(region)) {
    return null;
  }

  const localMask = projectIntoLocal(backend, mask, inverseLayerMatrix, region);

  const surface = backend.createSurface(region.width, region.height);
  const ctx = surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.drawImage(cache.surface.canvas, cache.rect.x - region.x, cache.rect.y - region.y);
  // Keep only what the selection covers.
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(localMask.surface.canvas, 0, 0);
  ctx.globalCompositeOperation = 'source-over';

  return { localMask, pixels: { rect: region, surface } };
};

/**
 * Re-places a document-space mask through `matrix`, returning a fresh placed
 * surface — how the selection outline follows a committed float.
 */
export const transformPlacedMask = (
  backend: RasterBackend,
  mask: PlacedSurface,
  matrix: Mat2d
): PlacedSurface | null => {
  const rect = roundOut(transformBounds(matrix, mask.rect));
  if (isEmpty(rect)) {
    return null;
  }
  const surface = backend.createSurface(rect.width, rect.height);
  const ctx = surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  const placed = multiply({ a: 1, b: 0, c: 0, d: 1, e: -rect.x, f: -rect.y }, matrix);
  ctx.setTransform(placed.a, placed.b, placed.c, placed.d, placed.e, placed.f);
  ctx.drawImage(mask.surface.canvas, mask.rect.x, mask.rect.y);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return { rect, surface };
};
