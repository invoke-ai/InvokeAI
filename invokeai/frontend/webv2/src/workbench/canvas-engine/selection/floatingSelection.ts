/**
 * Owns float geometry and extraction; the controller owns lifecycle/history/persistence. Document-space masks map
 * through the inverse layer transform into local space, resampling coverage rather than content. Pixels and live
 * transforms stay layer-local; whole-pixel translations preserve exact blits, and pointer deltas convert via
 * {@link documentDeltaToLocal}.
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
   * Effect-baked display copy; raw `pixels` alone are written back. Per-pixel effects are computed at lift and
   * unaffected by transforms; appearance changes cancel the float.
   */
  readonly display: RasterSurface | null;
  /** The layer cache's pre-lift pixels over the cut region, for cancel and the undo entry. */
  readonly before: { rect: Rect; data: ImageData };
  /** The selection mask at lift time, in DOCUMENT space — the ants follow it through the float's transform. */
  readonly mask: PlacedSurface;
  /** The float's live transform, in LAYER-LOCAL space. Identity at lift. */
  transform: LayerTransform;
}

/** A lift took the pixels, was refused admission (and reported it), or found nothing it could lift. */
export type FloatLiftResult = 'lifted' | 'refused' | 'unavailable';

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

/** Projects a placed document surface through the inverse layer matrix into a fresh local-region surface. */
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

/** The layer-local region a lift takes: the selection's local extent clipped to the layer content, or null. */
export const liftRegion = (cacheRect: Rect, layerMatrix: Mat2d, maskRect: Rect): Rect | null => {
  const inverseLayerMatrix = invert(layerMatrix);
  if (!inverseLayerMatrix || isEmpty(cacheRect) || isEmpty(maskRect)) {
    return null;
  }
  const region = intersect(roundOut(transformBounds(inverseLayerMatrix, maskRect)), cacheRect);
  return region && !isEmpty(region) ? region : null;
};

/**
 * Copies selected layer content over {@link liftRegion} and returns its exact local stencil, or null without
 * overlap. Does not mutate cache; callers cut with the same stencil.
 */
export const liftSelectedPixels = ({
  backend,
  cache,
  layerMatrix,
  mask,
}: LiftSelectedPixelsParams): LiftedPixels | null => {
  const region = liftRegion(cache.rect, layerMatrix, mask.rect);
  const inverseLayerMatrix = invert(layerMatrix);
  if (!region || !inverseLayerMatrix) {
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

/** Repositions the document mask through a matrix so committed selection outlines follow the float. */
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
