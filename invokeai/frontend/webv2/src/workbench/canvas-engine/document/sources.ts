/**
 * Small pure helpers over canvas layers and their sources, shared by the engine
 * (cache sizing, culling, fit-to-content) and the rasterizers.
 *
 * `getSourceBounds` returns a layer's axis-aligned bounds in **document space**
 * (used for culling / fit-to-content), while `getSourcePixelSize` returns the
 * **native** (unscaled) pixel dimensions of the layer's raster cache surface —
 * the compositor applies the layer transform when drawing, so caches hold
 * unscaled pixels. `isRenderableLayer` reports whether a layer can be
 * rasterized today (enabled, with an image/paint source).
 *
 * Zero React, zero import-time side effects.
 */

import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerContract, CanvasLayerSourceContract } from '@workbench/types';

import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { transformBounds } from '@workbench/canvas-engine/math/rect';
import { estimateTextExtent } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';

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

/** The number of strict composite groups (see {@link layerGroupRank}). */
export const LAYER_GROUP_COUNT = 4;

/**
 * The strict composite-group rank of a layer, ordered bottom→top: raster (0) <
 * control (1) < regional guidance (2) < inpaint mask (3). Mirrors legacy
 * `arrangeEntities` / `CanvasEntityRendererModule` and the layers panel's grouped
 * sections, so a layer's global insertion index never lets it draw — or be
 * hit-tested — out of group order (a raster created above a control layer still
 * composites, and is grabbed, below it). The compositor draws groups in this
 * order; the move/context-menu hit-test iterates them top→bottom, keeping the two
 * in lockstep so the layer the user clicks is the layer they see on top.
 */
export const layerGroupRank = (layer: CanvasLayerContract): number => {
  switch (layer.type) {
    case 'control':
      return 1;
    case 'regional_guidance':
      return 2;
    case 'inpaint_mask':
      return 3;
    default:
      // raster (and any future non-grouped renderable) composites at the bottom.
      return 0;
  }
};

/**
 * A mask layer's alpha bitmap viewed as a `paint` source, so the whole paint
 * pipeline (rasterize, content-sized cache, growth, stroke, persistence) can
 * operate on masks unchanged: the mask stores an alpha stencil exactly like a
 * paint bitmap; only the compositor differs (it colorizes the alpha with the
 * layer's `fill` instead of blitting the RGBA directly). Returns `null` for a
 * non-mask layer.
 */
export const maskAsPaintSource = (
  layer: CanvasLayerContract
): Extract<CanvasLayerSourceContract, { type: 'paint' }> | null =>
  isMaskLayer(layer) ? { bitmap: layer.mask.bitmap, offset: layer.mask.offset, type: 'paint' } : null;

/**
 * A layer's rasterizable source: its own `source` for raster/control layers, or
 * a synthetic `paint` view of a mask layer's alpha bitmap. `null` for layers
 * with neither (there are none today). This is the single "what pixels does this
 * layer hold" accessor the engine's rasterize / cache / persistence paths share.
 */
export const renderableSourceOf = (layer: CanvasLayerContract): CanvasLayerSourceContract | null => {
  if (hasSource(layer)) {
    return layer.source;
  }
  return maskAsPaintSource(layer);
};

/**
 * True when a layer's source is one the engine can rasterize today: image,
 * paint, gradient, text, or a rect/ellipse shape. A `polygon` shape has no
 * rasterizer yet (deferred), so it is not renderable.
 */
export const isRenderableLayer = (layer: CanvasLayerContract): boolean => {
  if (!layer.isEnabled) {
    return false;
  }
  // Mask layers are renderable whenever enabled: an empty (bitmap-less) mask
  // rasterizes to a zero-rect surface (skipped downstream, like empty paint).
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
      return source.kind !== 'polygon';
    default:
      return false;
  }
};

/**
 * True when a layer carries pixels the engine can rasterize AND merge: an
 * enabled, unlocked paint/image raster layer. Narrower than
 * {@link isRenderableLayer} — masks, control layers, shapes, text, and gradient
 * layers are all renderable but NOT mergeable. Locked matches the paint tool's
 * rule (`isLocked || !isEnabled` refuses a paint target): merge writes pixels
 * into the below layer and deletes the upper one, so it must refuse a locked
 * layer on either side just as painting refuses a locked target. The layers
 * panel's merge-down enablement (`canMergeLayerDown`) and the engine's
 * `mergeLayerDown` guard both defer to this single predicate so the hotkey,
 * context menu, and engine can never disagree about what is mergeable.
 */
export const isMergeableRasterLayer = (layer: CanvasLayerContract): boolean =>
  layer.isEnabled &&
  !layer.isLocked &&
  layer.type === 'raster' &&
  (layer.source.type === 'paint' || layer.source.type === 'image');

/**
 * A layer's content rectangle in its LOCAL (untransformed) coordinate space —
 * the extent its raster cache surface covers, before the layer transform is
 * applied. Every layer type is content-sized:
 *
 * - `image`: `[0, 0, w, h]` at the image's native pixels.
 * - `shape` / `text`: `[0, 0, w, h]` at the source's own / estimated extent.
 * - `gradient`: `[0, 0, width, height]` from the explicit extent, defaulting to
 *   the document dims for legacy gradients that predate the extent field.
 * - `paint`: the persisted bitmap's dims at its `offset` (legacy default `0, 0`),
 *   or an EMPTY rect when the layer has no bitmap yet (a brand-new paint layer).
 *
 * Throws for layers without a rasterizable source, so callers fail loudly.
 */
export const getSourceContentRect = (layer: CanvasLayerContract, doc: CanvasDocumentContractV2): Rect => {
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
 * A layer's axis-aligned bounds in DOCUMENT space: its {@link getSourceContentRect}
 * projected through the layer transform (rotation-aware). Used for culling and
 * fit-to-content. Throws for layers without a rasterizable source.
 */
export const getSourceBounds = (layer: CanvasLayerContract, doc: CanvasDocumentContractV2): Rect => {
  const contentRect = getSourceContentRect(layer, doc);
  const { transform } = layer;
  const matrix = fromTRS({ x: transform.x, y: transform.y }, transform.rotation, transform.scaleX, transform.scaleY);
  return transformBounds(matrix, contentRect);
};

/** The native (unscaled) pixel size of a layer's raster cache surface. */
export const getSourcePixelSize = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV2
): { width: number; height: number } => {
  const rect = getSourceContentRect(layer, doc);
  return { height: rect.height, width: rect.width };
};
