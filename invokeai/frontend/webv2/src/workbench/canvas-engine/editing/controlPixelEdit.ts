import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/types';

import { roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { bakeMatrix, IDENTITY_TRANSFORM, type LayerTransform } from '@workbench/canvas-engine/transform/transformMath';

export type ControlPixelEditRejectedReason = 'disabled' | 'locked' | 'not-ready' | 'unsupported';
export type ControlPixelEditDecision =
  | { status: 'direct' }
  | { status: 'materialize' }
  | { status: 'rejected'; reason: ControlPixelEditRejectedReason };

export interface DecideControlPixelEditInput {
  layer: CanvasControlLayerContract;
  hasSourceContent: boolean;
  isCacheReady: boolean;
}

const isIdentity = (transform: LayerTransform): boolean =>
  transform.x === 0 &&
  transform.y === 0 &&
  transform.scaleX === 1 &&
  transform.scaleY === 1 &&
  transform.rotation === 0;

const isRasterizable = (layer: CanvasControlLayerContract): boolean =>
  layer.source.type !== 'shape' || layer.source.kind !== 'polygon';

export const isLayerPixelEditEligible = (layer: CanvasLayerContract | undefined): boolean =>
  !!layer &&
  !layer.isLocked &&
  layer.isEnabled &&
  ((layer.type === 'raster' && layer.source.type === 'paint') || layer.type === 'control');

export const decideControlPixelEdit = ({
  hasSourceContent,
  isCacheReady,
  layer,
}: DecideControlPixelEditInput): ControlPixelEditDecision => {
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

export const buildMaterializedControlLayer = (
  layer: CanvasControlLayerContract,
  rect: Rect
): CanvasControlLayerContract => ({
  ...structuredClone(layer),
  source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
  transform: { ...IDENTITY_TRANSFORM },
});

export interface BakeControlPixelEditSurfaceInput {
  backend: RasterBackend;
  source: RasterSurface;
  sourceRect: Rect;
  transform: LayerTransform;
}

export const bakeControlPixelEditSurface = ({
  backend,
  source,
  sourceRect,
  transform,
}: BakeControlPixelEditSurfaceInput): { rect: Rect; surface: RasterSurface } => {
  const matrix = bakeMatrix(transform);
  const rect = roundOut(transformBounds(matrix, sourceRect));
  const surface = backend.createSurface(rect.width, rect.height);
  const ctx = surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.imageSmoothingEnabled = true;
  ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - rect.x, matrix.f - rect.y);
  ctx.drawImage(source.canvas, sourceRect.x, sourceRect.y);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return { rect, surface };
};
