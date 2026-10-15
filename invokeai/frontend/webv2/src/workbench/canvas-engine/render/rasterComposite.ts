import type {
  CanvasAdjustmentsContract,
  CanvasBlendMode,
  CanvasDocumentContractV3,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { GroupCompositeScope } from '@workbench/canvas-engine/render/groupCompositeScopes';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import { fromTRS, invert, multiply } from '@workbench/canvas-engine/math/mat2d';
import { expand, intersect, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { adjustmentsKey, applyAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';
import {
  collectCompositedGroups,
  planGroupCompositeScopes,
} from '@workbench/canvas-engine/render/groupCompositeScopes';

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
  /** Adjusted-group isolation scopes over `layers` (top-first, nested). Absent ⇒ fully pass-through. */
  groupScopes?: readonly GroupCompositeScope[];
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
  /** Owned access to a layer's pixels, released once they are drawn. */
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect; release(): void }>;
  /** The display's layer-local adjusted copy of `surface` (built and memoized on a miss), or null when `surface` is not a live cache. */
  adjustedSurface?(
    layerId: string,
    surface: RasterSurface,
    adjustments: CanvasAdjustmentsContract
  ): RasterSurface | null;
  readImageData?(surface: RasterSurface, rect: Rect): ImageData;
  writeImageData?(surface: RasterSurface, imageData: ImageData, x: number, y: number): void;
}

const defaultReadImageData = (surface: RasterSurface, rect: Rect): ImageData =>
  surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height);

const defaultWriteImageData = (surface: RasterSurface, imageData: ImageData, x: number, y: number): void =>
  surface.ctx.putImageData(imageData, x, y);

/** True when a leaf is a contributing raster layer with rasterizable, non-empty pixels. */
const isBaseRasterLeaf = (leaf: SemanticLeaf): leaf is SemanticLeaf & { layer: CanvasRasterLayerContractV2 } => {
  const { layer } = leaf;
  if (!leaf.contributionEnabled || layer.type !== 'raster') {
    return false;
  }
  if (layer.source.type === 'image') {
    return true;
  }
  if (layer.source.type === 'paint') {
    return layer.source.bitmap !== null;
  }
  return layer.source.type === 'text' && layer.source.content.length > 0;
};

/** JSON with recursively sorted object keys, so text edits produce deterministic identities. */
const stableSourceKey = (value: unknown): string =>
  JSON.stringify(value, (_key, entry: unknown) =>
    entry && typeof entry === 'object' && !Array.isArray(entry)
      ? Object.fromEntries(Object.entries(entry).sort(([a], [b]) => (a < b ? -1 : 1)))
      : entry
  );

/** A stable string identifying a source's pixels (its asset name, or an empty sentinel). */
const sourceRefOf = (source: CanvasLayerSourceContract): string => {
  switch (source.type) {
    case 'image':
      return `image:${source.image.imageName}`;
    case 'paint':
      return source.bitmap ? `paint:${source.bitmap.imageName}` : 'paint:empty';
    case 'text':
      return `text:${stableSourceKey(source)}`;
    default:
      return `${source.type}:unsupported`;
  }
};

/** The native (unscaled) content rect of a base-raster layer's source (layer-local). */
const contentRectOf = (layer: CanvasRasterLayerContractV2, doc: CanvasDocumentContractV3): Rect => {
  const { source } = layer;
  if (source.type === 'image') {
    return { height: source.image.height, width: source.image.width, x: 0, y: 0 };
  }
  if (source.type === 'paint' && source.bitmap) {
    const offset = source.offset ?? { x: 0, y: 0 };
    return { height: source.bitmap.height, width: source.bitmap.width, x: offset.x, y: offset.y };
  }
  if (source.type === 'text') {
    return getSourceContentRect(layer, doc);
  }
  return { height: doc.height, width: doc.width, x: 0, y: 0 };
};

/** Projects a document layer into its frozen composite contribution. */
const toLayerRef = (layer: CanvasRasterLayerContractV2, doc: CanvasDocumentContractV3): CompositeLayerRef => {
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

const scopeKey = (scope: GroupCompositeScope): string =>
  `${scope.id}@${scope.start}-${scope.end}:${adjustmentsKey(scope.adjustments)}:${scope.opacity}:${scope.blendMode}(${scope.children.map(scopeKey).join(',')})`;

/** Plans the enabled base-raster layers over an exact document-space rectangle. */
export const planBaseRasterComposite = (document: CanvasDocumentContractV3, rect: Rect): BaseRasterCompositeEntry => {
  const drawn = compileDocumentLeaves(document).filter(isBaseRasterLeaf);
  const layers = drawn.map((leaf) => toLayerRef(leaf.layer, document));
  const groupScopes = planGroupCompositeScopes(drawn, collectCompositedGroups(document));
  return {
    bbox: rect,
    key: `base-raster|${rectKey(rect)}|${layers.map(layerKey).join('|')}|${groupScopes.map(scopeKey).join('|')}`,
    kind: 'base-raster',
    layers,
    ...(groupScopes.length > 0 ? { groupScopes } : {}),
  };
};

/** Tight outward-rounded bounds of all enabled raster content in the document. */
export const getBaseRasterContentBounds = (document: CanvasDocumentContractV3): Rect | null => {
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
  const readImageData = deps.readImageData ?? defaultReadImageData;
  const writeImageData = deps.writeImageData ?? defaultWriteImageData;
  const view = bboxView(bbox);
  const fullRect: Rect = { height, width, x: 0, y: 0 };

  const drawRef = async (ctx: Ctx, ref: CompositeLayerRef): Promise<void> => {
    const layerSurface = await deps.getLayerSurface(ref.id);
    try {
      drawLayer(ctx, ref, layerSurface);
    } finally {
      layerSurface.release();
    }
  };

  /** Adjusts `[source at origin]` after resampling into an output-sized buffer; bounded by the output, not the layer. */
  const drawAdjustedAtOutput = (
    ctx: Ctx,
    ref: CompositeLayerRef,
    source: RasterSurface,
    origin: Rect,
    placed: Mat2d,
    adjustments: CanvasAdjustmentsContract
  ): void => {
    const temp = deps.backend.createSurface(width, height);
    setTransform(temp.ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
    temp.ctx.clearRect(0, 0, width, height);
    setTransform(temp.ctx, placed);
    temp.ctx.drawImage(source.canvas, origin.x, origin.y);
    const pixels = readImageData(temp, fullRect);
    applyAdjustments(pixels, adjustments);
    writeImageData(temp, pixels, 0, 0);
    ctx.save();
    setTransform(ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
    ctx.globalAlpha = ref.opacity;
    ctx.globalCompositeOperation = blendToComposite(ref.blendMode);
    ctx.drawImage(temp.canvas, 0, 0);
    ctx.restore();
  };

  const drawLayer = (ctx: Ctx, ref: CompositeLayerRef, layerSurface: { surface: RasterSurface; rect: Rect }): void => {
    const placed = multiply(view, layerMatrix(ref));
    let source = layerSurface.surface;
    let origin: Rect = layerSurface.rect;
    if (ref.adjustments) {
      // Adjust layer-local pixels before resampling, as the display does: reuse the display's adjusted copy, else
      // adjust only the region the output samples. A downscaled layer samples more pixels than the output holds,
      // so it adjusts after resampling instead, keeping the work bounded by the output.
      const shared = deps.adjustedSurface?.(ref.id, source, ref.adjustments) ?? null;
      if (shared) {
        source = shared;
      } else {
        const inverse = invert(placed);
        const sampled = inverse ? intersect(roundOut(expand(transformBounds(inverse, fullRect), 1)), origin) : origin;
        if (!sampled) {
          return;
        }
        if (sampled.width * sampled.height > width * height) {
          drawAdjustedAtOutput(ctx, ref, source, origin, placed, ref.adjustments);
          return;
        }
        const local = deps.backend.createSurface(sampled.width, sampled.height);
        setTransform(local.ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
        local.ctx.drawImage(source.canvas, origin.x - sampled.x, origin.y - sampled.y);
        const pixels = readImageData(local, { height: sampled.height, width: sampled.width, x: 0, y: 0 });
        applyAdjustments(pixels, ref.adjustments);
        writeImageData(local, pixels, 0, 0);
        source = local;
        origin = sampled;
      }
    }
    ctx.save();
    ctx.globalAlpha = ref.opacity;
    ctx.globalCompositeOperation = blendToComposite(ref.blendMode);
    setTransform(ctx, placed);
    // Draw at the layer-local content origin (content-sized paint layers place their pixels off-zero).
    ctx.drawImage(source.canvas, origin.x, origin.y);
    ctx.restore();
  };

  /** Draws `[start, end)` bottom→top; a scope isolates into a buffer, applies its stack, lands source-over. */
  const renderRange = async (
    ctx: Ctx,
    start: number,
    end: number,
    scopes: readonly GroupCompositeScope[]
  ): Promise<void> => {
    let scopeIndex = scopes.length - 1;
    for (let i = end - 1; i >= start;) {
      const scope = scopeIndex >= 0 ? scopes[scopeIndex]! : null;
      if (scope && i >= scope.start && i < scope.end) {
        const buffer = deps.backend.createSurface(width, height);
        setTransform(buffer.ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
        buffer.ctx.clearRect(0, 0, width, height);
        await renderRange(buffer.ctx, scope.start, scope.end, scope.children);
        if (!isIdentityAdjustments(scope.adjustments)) {
          const pixels = readImageData(buffer, fullRect);
          applyAdjustments(pixels, scope.adjustments);
          writeImageData(buffer, pixels, 0, 0);
        }
        ctx.save();
        setTransform(ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
        ctx.globalAlpha = scope.opacity;
        ctx.globalCompositeOperation = blendToComposite(scope.blendMode);
        ctx.drawImage(buffer.canvas, 0, 0);
        ctx.restore();
        i = scope.start - 1;
        scopeIndex -= 1;
        continue;
      }
      const ref = entry.layers[i];
      if (ref) {
        await drawRef(ctx, ref);
      }
      i -= 1;
    }
  };

  const surface = deps.backend.createSurface(width, height);
  setTransform(surface.ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  surface.ctx.clearRect(0, 0, surface.width, surface.height);
  await renderRange(surface.ctx, 0, entry.layers.length, entry.groupScopes ?? []);
  return surface;
};
