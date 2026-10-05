/**
 * Shared layer/source geometry. `getSourceBounds` returns document-space bounds; `getSourcePixelSize` returns
 * unscaled cache dimensions because compositing applies transforms.
 */

import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  ParametricShapeKind,
} from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { transformBounds } from '@workbench/canvas-engine/math/rect';
import { estimateTextExtent } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';

import { isLayerContributing } from './layerEligibility';

/** A layer type that carries a rasterizable `source` (raster or control). */
type SourceLayer = Extract<CanvasLayerContract, { source: CanvasLayerSourceContract }>;

/** A mask-bearing layer (inpaint mask / regional guidance); its `mask` holds an alpha bitmap. */
type MaskLayer = Extract<CanvasLayerContract, { mask: unknown }>;

/** True when a layer carries a `source` field (raster / control layers). */
const hasSource = (layer: CanvasLayerContract): layer is SourceLayer =>
  layer.type === 'raster' || layer.type === 'control';

/** True when a layer carries a `mask` (inpaint mask / regional guidance). */
export const isMaskLayer = (layer: CanvasLayerContract): layer is MaskLayer =>
  layer.type === 'inpaint_mask' || layer.type === 'regional_guidance';

/**
 * Views mask alpha as paint for shared rasterization, growth, stroke and persistence paths. Compositing alone
 * colorizes with mask fill. Returns null for non-mask layers.
 */
export const maskAsPaintSource = (
  layer: CanvasLayerContract
): Extract<CanvasLayerSourceContract, { type: 'paint' }> | null =>
  isMaskLayer(layer) ? { bitmap: layer.mask.bitmap, offset: layer.mask.offset, type: 'paint' } : null;

/**
 * Shared rasterizable-source accessor: raster/control source or a synthetic paint view of mask alpha; otherwise
 * null.
 */
export const renderableSourceOf = (layer: CanvasLayerContract): CanvasLayerSourceContract | null => {
  if (hasSource(layer)) {
    return layer.source;
  }
  return maskAsPaintSource(layer);
};

/** A polygon shape needs three points to fill; anything less has no raster. */
export const isEmptyPolygonShape = (source: { kind: ParametricShapeKind | 'polygon'; points?: unknown[] }): boolean =>
  source.kind === 'polygon' && (source.points?.length ?? 0) < 3;

/** Rasterizable sources are image, paint, gradient, text and shapes; polygons require three points. */
export const isRenderableLayer = (layer: CanvasLayerContract): boolean => {
  if (!isLayerContributing(layer)) {
    return false;
  }
  // Enabled empty masks rasterize to zero-rect surfaces, skipped downstream like empty paint.
  if (isMaskLayer(layer)) {
    return true;
  }
  if (!hasSource(layer)) {
    return false;
  }
  const { source } = layer;
  switch (source.type) {
    case 'image':
    case 'paint':
    case 'gradient':
    case 'text':
      return true;
    case 'shape':
      return !isEmptyPolygonShape(source);
    default:
      return false;
  }
};

/**
 * Untransformed local cache extent: native image size, shape/text extent, explicit gradient extent (legacy
 * document dimensions), or paint bitmap size at its offset (legacy zero). Empty paint has an empty rect;
 * unsupported sources throw.
 */
export const getSourceContentRect = (layer: CanvasLayerContract, doc: CanvasDocumentContractV3): Rect => {
  const source = renderableSourceOf(layer);
  if (!source) {
    throw new Error(`getSourceContentRect: layer type '${layer.type}' has no rasterizable source`);
  }
  switch (source.type) {
    case 'image':
      return { height: source.image.height, width: source.image.width, x: 0, y: 0 };
    case 'shape':
      return {
        height: Math.max(1, Math.round(source.height)),
        width: Math.max(1, Math.round(source.width)),
        x: 0,
        y: 0,
      };
    case 'text': {
      const extent = estimateTextExtent(source);
      return { height: extent.height, width: extent.width, x: 0, y: 0 };
    }
    case 'gradient': {
      // Explicit extent when present; legacy gradients (no extent) were
      // document-sized by construction, so default to the document dims.
      const width = source.width ?? doc.width;
      const height = source.height ?? doc.height;
      return { height, width, x: 0, y: 0 };
    }
    case 'paint': {
      if (!source.bitmap) {
        // A brand-new / cleared paint layer holds no pixels: an empty rect.
        return { height: 0, width: 0, x: 0, y: 0 };
      }
      const offset = source.offset ?? { x: 0, y: 0 };
      return { height: source.bitmap.height, width: source.bitmap.width, x: offset.x, y: offset.y };
    }
  }
};

/**
 * Rotation-aware document bounds from transformed {@link getSourceContentRect}, for culling and fitting.
 * Unsupported sources throw.
 */
export const getSourceBounds = (layer: CanvasLayerContract, doc: CanvasDocumentContractV3): Rect => {
  const contentRect = getSourceContentRect(layer, doc);
  const { transform } = layer;
  const matrix = fromTRS({ x: transform.x, y: transform.y }, transform.rotation, transform.scaleX, transform.scaleY);
  return transformBounds(matrix, contentRect);
};

/** The native (unscaled) pixel size of a layer's raster cache surface. */
export const getSourcePixelSize = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV3
): { width: number; height: number } => {
  const rect = getSourceContentRect(layer, doc);
  return { height: rect.height, width: rect.width };
};
