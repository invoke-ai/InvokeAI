/** Policy and pure transforms shared by control and raster-image pixel editing. */

import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { bakeMatrix, IDENTITY_TRANSFORM, type LayerTransform } from '@workbench/canvas-engine/transform/transformMath';

export type PixelEditRejectedReason = 'disabled' | 'locked' | 'not-ready' | 'unsupported';
export type PixelEditDecision =
  | { status: 'direct' }
  | { status: 'materialize' }
  | { status: 'rejected'; reason: PixelEditRejectedReason };

export type PixelEditableLayer = CanvasControlLayerContract | CanvasRasterLayerContractV2;

export interface DecidePixelEditInput {
  layer: PixelEditableLayer;
  hasSourceContent: boolean;
  isCacheReady: boolean;
}

const isIdentity = (transform: LayerTransform): boolean =>
  transform.x === 0 &&
  transform.y === 0 &&
  transform.scaleX === 1 &&
  transform.scaleY === 1 &&
  transform.rotation === 0;

const isRasterizable = (layer: PixelEditableLayer): boolean =>
  layer.type === 'raster'
    ? layer.source.type === 'image' || layer.source.type === 'paint'
    : layer.source.type !== 'shape' || layer.source.kind !== 'polygon';

export const isLayerPixelEditEligible = (layer: CanvasLayerContract | undefined): boolean =>
  !!layer &&
  !layer.isLocked &&
  layer.isEnabled &&
  ((layer.type === 'raster' && layer.source.type === 'paint') || (layer.type === 'control' && isRasterizable(layer)));

export const decidePixelEdit = ({ hasSourceContent, isCacheReady, layer }: DecidePixelEditInput): PixelEditDecision => {
  if (layer.isLocked) {
    return { reason: 'locked', status: 'rejected' };
  }
  if (!layer.isEnabled) {
    return { reason: 'disabled', status: 'rejected' };
  }
  if (!isRasterizable(layer)) {
    return { reason: 'unsupported', status: 'rejected' };
  }
  if (hasSourceContent && !isCacheReady) {
    return { reason: 'not-ready', status: 'rejected' };
  }
  if (layer.source.type === 'paint' && isIdentity(layer.transform)) {
    return { status: 'direct' };
  }
  return { status: 'materialize' };
};

export const buildMaterializedPixelLayer = (layer: PixelEditableLayer, rect: Rect): PixelEditableLayer => {
  const materialized: PixelEditableLayer = {
    ...structuredClone(layer),
    source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
    transform: { ...IDENTITY_TRANSFORM },
  };
  if (materialized.type === 'raster') {
    // Raster adjustments are applied before the layer transform by the normal
    // compositor. Materialization bakes those displayed pixels so the eraser
    // edits exactly what the user sees; retaining the metadata would apply the
    // same adjustments a second time to the resulting paint source.
    delete materialized.adjustments;
  }
  return materialized;
};

export interface BakePixelEditSurfaceInput {
  backend: RasterBackend;
  source: RasterSurface;
  sourceRect: Rect;
  transform: LayerTransform;
}

export const bakePixelEditSurface = ({
  backend,
  source,
  sourceRect,
  transform,
}: BakePixelEditSurfaceInput): { rect: Rect; surface: RasterSurface } => {
  const matrix = bakeMatrix(transform);
  const rect = roundOut(transformBounds(matrix, sourceRect));
  const surface = backend.createSurface(rect.width, rect.height);
  const ctx = surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  // Materialization commits the layer transform into the document's native
  // pixel grid. Match the compositor's canonical 1× policy (nearest-neighbor)
  // so starting an edit does not blur otherwise untouched transformed pixels.
  // Viewport downscaling may still smooth this canonical result afterwards.
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - rect.x, matrix.f - rect.y);
  ctx.drawImage(source.canvas, sourceRect.x, sourceRect.y);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return { rect, surface };
};
