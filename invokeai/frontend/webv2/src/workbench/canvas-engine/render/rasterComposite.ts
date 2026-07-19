import type {
  CanvasAdjustmentsContract,
  CanvasBlendMode,
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { adjustmentsKey, applyAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';

type Ctx = RasterSurface['ctx'];

/** The engine-owned structural subset needed to plan and render a raster layer contribution. */
export interface CompositeLayerRef {
  id: string;
  sourceRef: string;
  contentSize: { width: number; height: number };
  contentOffset: { x: number; y: number };
  transform: { x: number; y: number; scaleX: number; scaleY: number; rotation: number };
  opacity: number;
  blendMode: CanvasBlendMode;
  adjustments?: CanvasAdjustmentsContract;
}

/** The engine-owned structural subset consumed by the raster compositor. */
export interface CompositeEntry {
  bbox: Rect;
  layers: readonly CompositeLayerRef[];
}

export interface BaseRasterCompositeEntry extends CompositeEntry {
  key: string;
  kind: 'base-raster';
  layers: CompositeLayerRef[];
}

export interface RenderRasterCompositeDeps {
  backend: {
    createSurface(width: number, height: number): RasterSurface;
  };
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect }>;
  readImageData?(surface: RasterSurface, rect: Rect): ImageData;
  writeImageData?(surface: RasterSurface, imageData: ImageData, x: number, y: number): void;
}

const defaultReadImageData = (surface: RasterSurface, rect: Rect): ImageData =>
  surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height);

const defaultWriteImageData = (surface: RasterSurface, imageData: ImageData, x: number, y: number): void =>
  surface.ctx.putImageData(imageData, x, y);

/** True when a layer is an enabled raster layer with rasterizable, non-empty pixels. */
const isBaseRasterLayer = (layer: CanvasLayerContract): layer is CanvasRasterLayerContractV2 => {
  if (!layer.isEnabled || layer.type !== 'raster') {
    return false;
  }
  if (layer.source.type === 'image') {
    return true;
  }
  return layer.source.type === 'paint' && layer.source.bitmap !== null;
};

/** A stable string identifying a source's pixels (its asset name, or an empty sentinel). */
const sourceRefOf = (source: CanvasLayerSourceContract): string => {
  switch (source.type) {
    case 'image':
      return `image:${source.image.imageName}`;
    case 'paint':
      return source.bitmap ? `paint:${source.bitmap.imageName}` : 'paint:empty';
    default:
      return `${source.type}:unsupported`;
  }
};

/** The native (unscaled) content rect of a base-raster layer's source (layer-local). */
const contentRectOf = (layer: CanvasRasterLayerContractV2, doc: CanvasDocumentContractV2): Rect => {
  const { source } = layer;
  if (source.type === 'image') {
    return { height: source.image.height, width: source.image.width, x: 0, y: 0 };
  }
  if (source.type === 'paint' && source.bitmap) {
    const offset = source.offset ?? { x: 0, y: 0 };
    return { height: source.bitmap.height, width: source.bitmap.width, x: offset.x, y: offset.y };
  }
  return { height: doc.height, width: doc.width, x: 0, y: 0 };
};

/** Projects a document layer into its frozen composite contribution. */
const toLayerRef = (layer: CanvasRasterLayerContractV2, doc: CanvasDocumentContractV2): CompositeLayerRef => {
  const rect = contentRectOf(layer, doc);
  const hasAdjustments = !isIdentityAdjustments(layer.adjustments);
  return {
    blendMode: layer.blendMode,
    contentOffset: { x: rect.x, y: rect.y },
    contentSize: { height: rect.height, width: rect.width },
    id: layer.id,
    opacity: layer.opacity,
    sourceRef: sourceRefOf(layer.source),
    ...(hasAdjustments && layer.adjustments ? { adjustments: layer.adjustments } : {}),
    transform: {
      rotation: layer.transform.rotation,
      scaleX: layer.transform.scaleX,
      scaleY: layer.transform.scaleY,
      x: layer.transform.x,
      y: layer.transform.y,
    },
  };
};

const rectKey = (rect: Rect): string => `${rect.x},${rect.y},${rect.width},${rect.height}`;

const layerKey = (ref: CompositeLayerRef): string => {
  const t = ref.transform;
  const o = ref.contentOffset;
  return [
    ref.id,
    ref.sourceRef,
    o.x,
    o.y,
    t.x,
    t.y,
    t.scaleX,
    t.scaleY,
    t.rotation,
    ref.opacity,
    ref.blendMode,
    ref.adjustments ? adjustmentsKey(ref.adjustments) : '-',
  ].join(':');
};

/** Union of the provided composite-layer bounds in document space, or `null` when empty. */
export const getCompositeLayerBounds = (layers: readonly CompositeLayerRef[]): Rect | null => {
  let bounds: Rect | null = null;
  for (const ref of layers) {
    const nativeRect: Rect = {
      height: ref.contentSize.height,
      width: ref.contentSize.width,
      x: ref.contentOffset.x,
      y: ref.contentOffset.y,
    };
    const layerBounds = transformBounds(layerMatrix(ref), nativeRect);
    bounds = bounds === null ? layerBounds : union(bounds, layerBounds);
  }
  return bounds;
};

/** Plans the enabled base-raster layers over an exact document-space rectangle. */
export const planBaseRasterComposite = (document: CanvasDocumentContractV2, rect: Rect): BaseRasterCompositeEntry => {
  const layers = document.layers.filter(isBaseRasterLayer).map((layer) => toLayerRef(layer, document));
  return {
    bbox: rect,
    key: `base-raster|${rectKey(rect)}|${layers.map(layerKey).join('|')}`,
    kind: 'base-raster',
    layers,
  };
};

/** Tight outward-rounded bounds of all enabled raster content in the document. */
export const getBaseRasterContentBounds = (document: CanvasDocumentContractV2): Rect | null => {
  const bounds = getCompositeLayerBounds(planBaseRasterComposite(document, document.bbox).layers);
  return bounds === null ? null : roundOut(bounds);
};

/** Document→bbox translate matrix (the "view" the entry is composited under). */
const bboxView = (bbox: Rect): Mat2d => ({ a: 1, b: 0, c: 0, d: 1, e: -bbox.x, f: -bbox.y });

/** Applies a matrix to a 2D context's transform. */
const setTransform = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

/** The layer's local→document transform matrix. */
const layerMatrix = (ref: CompositeLayerRef): Mat2d =>
  fromTRS(
    { x: ref.transform.x, y: ref.transform.y },
    ref.transform.rotation,
    ref.transform.scaleX,
    ref.transform.scaleY
  );

/** Composites an entry's layers, in z-order, onto a fresh bbox-sized surface. */
export const renderRasterComposite = async (
  entry: CompositeEntry,
  deps: RenderRasterCompositeDeps
): Promise<RasterSurface> => {
  const { bbox } = entry;
  const width = Math.max(0, bbox.width);
  const height = Math.max(0, bbox.height);
  const surface = deps.backend.createSurface(width, height);
  const ctx = surface.ctx;
  const readImageData = deps.readImageData ?? defaultReadImageData;
  const writeImageData = deps.writeImageData ?? defaultWriteImageData;

  setTransform(ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  ctx.clearRect(0, 0, surface.width, surface.height);

  const view = bboxView(bbox);
  // Layers are stored top-first (index 0 = top-most); draw bottom→top.
  for (let i = entry.layers.length - 1; i >= 0; i--) {
    const ref = entry.layers[i];
    if (!ref) {
      continue;
    }
    const layerSurface = await deps.getLayerSurface(ref.id);
    if (ref.adjustments) {
      // Bake non-destructive adjustments so the generated image matches what the
      // user sees: render this layer alone into a bbox temp, apply the LUTs, then
      // composite the adjusted temp with the layer's opacity/blend.
      const temp = deps.backend.createSurface(width, height);
      const tempCtx = temp.ctx;
      setTransform(tempCtx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
      tempCtx.clearRect(0, 0, width, height);
      setTransform(tempCtx, multiply(view, layerMatrix(ref)));
      tempCtx.drawImage(layerSurface.surface.canvas, layerSurface.rect.x, layerSurface.rect.y);
      const fullRect: Rect = { height, width, x: 0, y: 0 };
      const pixels = readImageData(temp, fullRect);
      applyAdjustments(pixels, ref.adjustments);
      writeImageData(temp, pixels, 0, 0);
      ctx.save();
      setTransform(ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
      ctx.globalAlpha = ref.opacity;
      ctx.globalCompositeOperation = blendToComposite(ref.blendMode);
      ctx.drawImage(temp.canvas, 0, 0);
      ctx.restore();
      continue;
    }
    ctx.save();
    ctx.globalAlpha = ref.opacity;
    ctx.globalCompositeOperation = blendToComposite(ref.blendMode);
    setTransform(ctx, multiply(view, layerMatrix(ref)));
    // Draw the cache at its layer-local content origin (content-sized paint
    // layers place their pixels off-zero).
    ctx.drawImage(layerSurface.surface.canvas, layerSurface.rect.x, layerSurface.rect.y);
    ctx.restore();
  }

  return surface;
};
