/** Policy and pure transforms shared by control and raster-image pixel editing, and selection Fill/Erase targets. */

import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { rebaseControlOriginal } from '@workbench/canvas-engine/document/controlDerivation';
import { isLayerTransparencyLocked, isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';
import { isEmptyPolygonShape, isMaskLayer } from '@workbench/canvas-engine/document/sources';
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
  /** Ancestor-effective state; the leaf's own flags are always applied. */
  contributionEnabled?: boolean;
  effectiveLocked?: boolean;
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
    : layer.source.type !== 'shape' || !isEmptyPolygonShape(layer.source);

/** Whether the leaf, with its ancestors applied, can take a direct pixel edit. */
export const isLeafPixelEditEligible = (leaf: SemanticLeaf | null | undefined): boolean =>
  isLeafEditable(leaf) &&
  ((leaf.layer.type === 'raster' && leaf.layer.source.type === 'paint') ||
    (leaf.layer.type === 'control' && isRasterizable(leaf.layer)));

/**
 * `leafEditRefusal` for edits that write pixels (brush, eraser, shapes, selection Fill/Erase), which also refuse a leaf
 * hidden on the canvas only: its pixels would change unseen. Moves and transforms keep `leafEditRefusal`. It lives
 * with the pixel-edit policy rather than `document/layerEligibility.ts` because only lazily loaded editing code uses
 * it, and that module ships in the editor's initial chunk.
 */
export const leafPaintRefusal = (
  leaf: SemanticLeaf
): 'layer-display-hidden' | 'layer-hidden' | 'layer-locked' | null =>
  !leaf.contributionEnabled
    ? 'layer-hidden'
    : leaf.documentHidden
      ? 'layer-display-hidden'
      : leaf.effectiveLocked
        ? 'layer-locked'
        : null;

/** Why selection Fill or Erase cannot edit the selection; `no-target` when nothing is selected. */
export type SelectionPixelEditRefusal =
  | 'no-target'
  | 'unsupported'
  | 'layer-unpaintable'
  | NonNullable<ReturnType<typeof leafPaintRefusal>>
  | 'layer-alpha-locked';

/**
 * Whether the brush, the eraser and selection Fill/Erase edit `layer`'s own pixels: a paint or image raster, a mask or
 * region, or a control layer with pixels. A control layer whose shape is an empty polygon has none and is refused
 * (`layer-unpaintable`); any other layer is not edited in place.
 */
export const editsPixelsInPlace = (layer: CanvasLayerContract): boolean =>
  isMaskLayer(layer) ||
  (layer.type === 'raster'
    ? layer.source.type === 'paint' || layer.source.type === 'image'
    : layer.type === 'control' && isRasterizable(layer));

/**
 * Why selection Fill or Erase cannot edit the selection, refusing as the brush and eraser do; null when it can.
 * `selected` is the selected leaf, `'group'` for a selected group, or null when nothing is selected. Pixels still
 * loading are not predicted: the edit itself waits for them. Callers look the selection up, so this module takes no
 * document lookup: one would place it, through the document model, in the editor's initial chunk.
 */
export const selectionPixelEditRefusal = (
  selected: SemanticLeaf | 'group' | null,
  kind: 'fill' | 'erase'
): SelectionPixelEditRefusal | null => {
  if (selected === null) {
    return 'no-target';
  }
  if (selected === 'group') {
    return 'unsupported';
  }
  const leaf = selected;
  if (!editsPixelsInPlace(leaf.layer)) {
    return leaf.layer.type === 'control' ? 'layer-unpaintable' : 'unsupported';
  }
  return (
    leafPaintRefusal(leaf) ?? (kind === 'erase' && isLayerTransparencyLocked(leaf.layer) ? 'layer-alpha-locked' : null)
  );
};

export const decidePixelEdit = ({
  contributionEnabled = true,
  effectiveLocked = false,
  hasSourceContent,
  isCacheReady,
  layer,
}: DecidePixelEditInput): PixelEditDecision => {
  if (layer.isLocked || effectiveLocked) {
    return { reason: 'locked', status: 'rejected' };
  }
  if (!layer.isEnabled || !contributionEnabled) {
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

/**
 * The layer after its transform is baked into paint pixels at `rect`. A control layer's kept original moves with the
 * bake, or detaches when the bake would shear its frame.
 */
export const buildMaterializedPixelLayer = (layer: PixelEditableLayer, rect: Rect): PixelEditableLayer => {
  const materialized: PixelEditableLayer = {
    ...structuredClone(layer),
    source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
    transform: { ...IDENTITY_TRANSFORM },
  };
  if (materialized.type === 'control') {
    return rebaseControlOriginal(materialized, layer.transform);
  }
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
