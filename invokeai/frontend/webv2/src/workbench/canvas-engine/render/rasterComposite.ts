import type {
  CanvasAdjustmentsContract,
  CanvasBlendMode,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { LayerUnreadable } from '@workbench/canvas-engine/layerExportGuards';
import type { GroupCompositeScope } from '@workbench/canvas-engine/render/groupCompositeScopes';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentIndex } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import { fromTRS, invert, multiply } from '@workbench/canvas-engine/math/mat2d';
import { expand, intersect, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import {
  adjustmentsKey,
  applyAdjustments,
  hasSpatialAdjustments,
  isIdentityAdjustments,
  isPositionDependentAdjustments,
} from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';
import {
  adjustExactly,
  AdjustmentRefusedError,
  imageDataOf,
  type SpatialRunner,
} from '@workbench/canvas-engine/render/exactAdjustments';
import {
  collectCompositedGroups,
  planGroupCompositeScopes,
} from '@workbench/canvas-engine/render/groupCompositeScopes';

type Ctx = RasterSurface['ctx'];

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

export interface CompositeEntry {
  bbox: Rect;
  layers: readonly CompositeLayerRef[];
  /** Top-first, nested; absent means fully pass-through. */
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
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect; release(): void }>;
  /** Exact copies only, never built here: {@link compositeScratchPixels} covers what a miss adjusts. */
  adjustedSurface?(
    layerId: string,
    surface: RasterSurface,
    adjustments: CanvasAdjustmentsContract
  ): RasterSurface | null;
  readImageData?(surface: RasterSurface, rect: Rect): ImageData;
  writeImageData?(surface: RasterSurface, imageData: ImageData, x: number, y: number): void;
  /** Without it, a contributing spatial entry throws {@link AdjustmentRefusedError} rather than render without it. */
  runSpatial?: SpatialRunner;
  signal?: AbortSignal;
  /** Rendering reserves here only what actual pixels need beyond {@link compositeScratchPixels}. */
  reserve?(
    bytes: number
  ):
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
}

/** Nothing beyond the reservation was allocated; export maps the class to `over-budget`. */
export class RasterCompositeOverBudgetError extends Error {
  constructor() {
    super('The canvas composite exceeds the available raster memory budget.');
    this.name = 'RasterCompositeOverBudgetError';
  }
}

/** Export reports it as an `unreadable` refusal naming the layer. */
export class RasterCompositeUnreadableError extends Error {
  constructor(readonly refusal: LayerUnreadable) {
    super(`The pixels of layer ${refusal.layerId} could not be read.`);
    this.name = 'RasterCompositeUnreadableError';
  }
}

/** Export maps it to `not-ready`. */
export class RasterCompositeNotReadyError extends Error {
  constructor(readonly layerId: string) {
    super(`The pixels of layer ${layerId} are not ready.`);
    this.name = 'RasterCompositeNotReadyError';
  }
}

const defaultReadImageData = (surface: RasterSurface, rect: Rect): ImageData =>
  surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height);

const defaultWriteImageData = (surface: RasterSurface, imageData: ImageData, x: number, y: number): void =>
  surface.ctx.putImageData(imageData, x, y);

export const isBaseRasterContentLayer = (layer: CanvasLayerContract): layer is CanvasRasterLayerContractV2 => {
  if (layer.type !== 'raster') {
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

const isBaseRasterLeaf = (leaf: SemanticLeaf): leaf is SemanticLeaf & { layer: CanvasRasterLayerContractV2 } =>
  leaf.contributionEnabled && isBaseRasterContentLayer(leaf.layer);

/** Sorted keys, so text edits produce deterministic identities. */
const stableSourceKey = (value: unknown): string =>
  JSON.stringify(value, (_key, entry: unknown) =>
    entry && typeof entry === 'object' && !Array.isArray(entry)
      ? Object.fromEntries(Object.entries(entry).sort(([a], [b]) => (a < b ? -1 : 1)))
      : entry
  );

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

/** `contentRects` replaces a leaf's planned content rect with the extent of the pixels it will draw. */
export const planRasterComposite = (
  document: CanvasDocumentContractV3,
  leaves: readonly (SemanticLeaf & { layer: CanvasRasterLayerContractV2 })[],
  rect: Rect,
  contentRects?: ReadonlyMap<string, Rect>
): CompositeEntry => {
  const layers = leaves.map((leaf) => {
    const ref = toLayerRef(leaf.layer, document);
    const content = contentRects?.get(leaf.id);
    return content
      ? {
          ...ref,
          contentOffset: { x: content.x, y: content.y },
          contentSize: { height: content.height, width: content.width },
        }
      : ref;
  });
  const groupScopes = planGroupCompositeScopes(leaves, collectCompositedGroups(document));
  return { bbox: rect, layers, ...(groupScopes.length > 0 ? { groupScopes } : {}) };
};

export const planBaseRasterComposite = (document: CanvasDocumentContractV3, rect: Rect): BaseRasterCompositeEntry => {
  const { layers, groupScopes } = planRasterComposite(
    document,
    compileDocumentLeaves(document).filter(isBaseRasterLeaf),
    rect
  );
  return {
    bbox: rect,
    key: `base-raster|${rectKey(rect)}|${layers.map(layerKey).join('|')}|${(groupScopes ?? []).map(scopeKey).join('|')}`,
    kind: 'base-raster',
    layers: [...layers],
    ...(groupScopes ? { groupScopes } : {}),
  };
};

export const getBaseRasterContentBounds = (document: CanvasDocumentContractV3): Rect | null => {
  const bounds = getCompositeLayerBounds(planBaseRasterComposite(document, document.bbox).layers);
  return bounds === null ? null : roundOut(bounds);
};

const bboxView = (bbox: Rect): Mat2d => ({ a: 1, b: 0, c: 0, d: 1, e: -bbox.x, f: -bbox.y });

const setTransform = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

const layerMatrix = (ref: Pick<CompositeLayerRef, 'transform'>): Mat2d =>
  fromTRS(
    { x: ref.transform.x, y: ref.transform.y },
    ref.transform.rotation,
    ref.transform.scaleX,
    ref.transform.scaleY
  );

const requiresWholeSource = (adjustments: CanvasAdjustmentsContract | undefined): boolean =>
  hasSpatialAdjustments(adjustments);

/** Whole source for context-dependent stacks; otherwise the sampled region plus resampling margin. */
const layerSourceRegion = (
  ref: Pick<CompositeLayerRef, 'transform' | 'adjustments'>,
  frame: Rect,
  content: Rect
): Rect | null => {
  if (requiresWholeSource(ref.adjustments)) {
    return content;
  }
  const inverse = invert(multiply(bboxView(frame), layerMatrix(ref)));
  const output: Rect = { height: Math.max(0, frame.height), width: Math.max(0, frame.width), x: 0, y: 0 };
  return inverse ? intersect(roundOut(expand(transformBounds(inverse, output), 1)), content) : content;
};

/** Aligned to the source pixel grid; rotation or rescaling keeps the full source, since its origin affects sampling. */
export const getRasterSnapshotCaptureRect = (
  document: CanvasDocumentContractV3,
  layer: CanvasLayerContract,
  frame: Rect | undefined,
  content: Rect
): Rect => {
  const t = layer.transform;
  if (!frame || t.rotation !== 0 || Math.abs(t.scaleX) !== 1 || Math.abs(t.scaleY) !== 1) {
    return { ...content };
  }
  if (layer.type === 'raster') {
    const index = getDocumentIndex(document);
    if (
      index.byId.get(layer.id)!.path.some((id) => {
        const ancestor = index.byId.get(id)!.node;
        return ancestor.type === 'group' && requiresWholeSource(ancestor.adjustments);
      })
    ) {
      return { ...content };
    }
  }
  const sampled = layerSourceRegion(layer, frame, content);
  if (!sampled) {
    return { x: content.x, y: content.y, width: 0, height: 0 };
  }
  const pixels = intersect(roundOut({ ...sampled, x: sampled.x - content.x, y: sampled.y - content.y }), {
    x: 0,
    y: 0,
    width: content.width,
    height: content.height,
  });
  return pixels
    ? { ...pixels, x: pixels.x + content.x, y: pixels.y + content.y }
    : { x: content.x, y: content.y, width: 0, height: 0 };
};

const areaOf = (rect: Rect | null): number => (rect ? Math.max(0, rect.width) * Math.max(0, rect.height) : 0);

/** Position-dependent stacks adjust the whole sampled layer-local region; spatial stacks all of `content`. */
const layerScratchPixels = (ref: CompositeLayerRef, frame: Rect, content: Rect): number => {
  if (requiresWholeSource(ref.adjustments)) {
    return areaOf(layerSourceRegion(ref, frame, content));
  }
  const sampled = isPositionDependentAdjustments(ref.adjustments) ? layerSourceRegion(ref, frame, content) : null;
  return Math.max(areaOf(frame), areaOf(sampled));
};

const plannedContent = (ref: CompositeLayerRef): Rect => ({
  ...ref.contentSize,
  x: ref.contentOffset.x,
  y: ref.contentOffset.y,
});

const placedExtent = (placed: readonly { readonly ref: CompositeLayerRef; readonly rect: Rect }[]): Rect | null => {
  let extent: Rect | null = null;
  for (const { rect, ref } of placed) {
    if (rect.width > 0 && rect.height > 0) {
      const bounds = transformBounds(layerMatrix(ref), rect);
      extent = extent ? union(extent, bounds) : bounds;
    }
  }
  return extent ? roundOut(extent) : null;
};

const plannedScopeExtent = (entry: CompositeEntry, scope: GroupCompositeScope): Rect | null =>
  placedExtent(entry.layers.slice(scope.start, scope.end).map((ref) => ({ rect: plannedContent(ref), ref })));

/** One buffer per group scope and adjusted layer, each read back once; measured excess reserves via `deps.reserve`. */
export const compositeScratchPixels = (entry: CompositeEntry): number => {
  const scratchOf = (start: number, end: number, scopes: readonly GroupCompositeScope[], frame: Rect): number => {
    let pixels = 0;
    let index = start;
    for (const scope of scopes) {
      for (; index < scope.start; index += 1) {
        pixels += leafScratch(index, frame);
      }
      const inner = requiresWholeSource(scope.adjustments) ? plannedScopeExtent(entry, scope) : frame;
      if (inner) {
        pixels += areaOf(inner) + scratchOf(scope.start, scope.end, scope.children, inner);
      }
      index = scope.end;
    }
    for (; index < end; index += 1) {
      pixels += leafScratch(index, frame);
    }
    return pixels;
  };
  const leafScratch = (index: number, frame: Rect): number => {
    const ref = entry.layers[index]!;
    return ref.adjustments ? layerScratchPixels(ref, frame, plannedContent(ref)) : 0;
  };
  return scratchOf(0, entry.layers.length, entry.groupScopes ?? [], entry.bbox);
};

type LeasedLayerSurface = Awaited<ReturnType<RenderRasterCompositeDeps['getLayerSurface']>>;

const IDENTITY: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };

/** Spatial stacks evaluate on their owner's whole source, so any bbox over the same content draws the same pixels. */
export const renderRasterComposite = async (
  entry: CompositeEntry,
  deps: RenderRasterCompositeDeps
): Promise<RasterSurface> => {
  const { bbox } = entry;
  const readImageData = deps.readImageData ?? defaultReadImageData;
  const writeImageData = deps.writeImageData ?? defaultWriteImageData;
  const fullRectOf = (rect: Rect): Rect => ({
    height: Math.max(0, rect.height),
    width: Math.max(0, rect.width),
    x: 0,
    y: 0,
  });

  const evaluate = async (
    pixels: ImageData,
    adjustments: CanvasAdjustmentsContract,
    origin: Rect,
    ownerId: string
  ): Promise<ImageData> => {
    const result = await adjustExactly(pixels, adjustments, origin, { run: deps.runSpatial, signal: deps.signal });
    if (result.status === 'aborted') {
      throw new DOMException('The raster composite was aborted.', 'AbortError');
    }
    if (result.status !== 'ok') {
      throw new AdjustmentRefusedError(ownerId, result.status);
    }
    return result.data === pixels.data ? pixels : imageDataOf(result.data, pixels.width, pixels.height);
  };

  /** Rasterized pixels can exceed the planned bounds (measured text), so the excess is reserved first. */
  const reserveExcess = (actualPixels: number, plannedPixels: number): { release(): void } | null => {
    const excess = actualPixels - plannedPixels;
    const reservation = excess > 0 ? deps.reserve?.(excess * 2 * 4) : undefined;
    if (reservation?.status === 'over-budget') {
      throw new RasterCompositeOverBudgetError();
    }
    return reservation?.lease ?? null;
  };

  /** Position-independent stacks only: per-pixel color maps ignore the origin. */
  const drawAdjustedAtOutput = (
    ctx: Ctx,
    frame: Rect,
    ref: CompositeLayerRef,
    source: RasterSurface,
    origin: Rect,
    placed: Mat2d,
    adjustments: CanvasAdjustmentsContract
  ): void => {
    const output = fullRectOf(frame);
    const temp = deps.backend.createSurface(output.width, output.height);
    setTransform(temp.ctx, IDENTITY);
    temp.ctx.clearRect(0, 0, output.width, output.height);
    setTransform(temp.ctx, placed);
    temp.ctx.drawImage(source.canvas, origin.x, origin.y);
    const pixels = readImageData(temp, output);
    applyAdjustments(pixels, adjustments, frame);
    writeImageData(temp, pixels, 0, 0);
    ctx.save();
    setTransform(ctx, IDENTITY);
    ctx.globalAlpha = ref.opacity;
    ctx.globalCompositeOperation = blendToComposite(ref.blendMode);
    ctx.drawImage(temp.canvas, 0, 0);
    ctx.restore();
  };

  const drawLayer = async (
    ctx: Ctx,
    frame: Rect,
    ref: CompositeLayerRef,
    layerSurface: { surface: RasterSurface; rect: Rect }
  ): Promise<void> => {
    let extra: { release(): void } | null = null;
    try {
      const placed = multiply(bboxView(frame), layerMatrix(ref));
      let source = layerSurface.surface;
      let origin: Rect = layerSurface.rect;
      if (ref.adjustments) {
        // Adjust layer-local pixels before resampling, as the display does, so resampling averages noise alike.
        const shared = deps.adjustedSurface?.(ref.id, source, ref.adjustments) ?? null;
        const sampled = shared ? null : layerSourceRegion(ref, frame, origin);
        if (shared) {
          source = shared;
        } else if (!sampled || areaOf(sampled) === 0) {
          return;
        } else if (requiresWholeSource(ref.adjustments)) {
          extra = reserveExcess(areaOf(sampled), layerScratchPixels(ref, frame, plannedContent(ref)));
          const local = deps.backend.createSurface(sampled.width, sampled.height);
          setTransform(local.ctx, IDENTITY);
          local.ctx.drawImage(source.canvas, 0, 0);
          const pixels = readImageData(local, fullRectOf(sampled));
          writeImageData(local, await evaluate(pixels, ref.adjustments, sampled, ref.id), 0, 0);
          source = local;
        } else {
          if (areaOf(sampled) > areaOf(frame) && !isPositionDependentAdjustments(ref.adjustments)) {
            drawAdjustedAtOutput(ctx, frame, ref, source, origin, placed, ref.adjustments);
            return;
          }
          extra = reserveExcess(areaOf(sampled), layerScratchPixels(ref, frame, plannedContent(ref)));
          const local = deps.backend.createSurface(sampled.width, sampled.height);
          setTransform(local.ctx, IDENTITY);
          local.ctx.drawImage(source.canvas, origin.x - sampled.x, origin.y - sampled.y);
          const pixels = readImageData(local, fullRectOf(sampled));
          applyAdjustments(pixels, ref.adjustments, sampled);
          writeImageData(local, pixels, 0, 0);
          source = local;
          origin = sampled;
        }
      }
      ctx.save();
      ctx.globalAlpha = ref.opacity;
      ctx.globalCompositeOperation = blendToComposite(ref.blendMode);
      setTransform(ctx, placed);
      // Content-sized paint layers place their pixels off-zero.
      ctx.drawImage(source.canvas, origin.x, origin.y);
      ctx.restore();
    } finally {
      extra?.release();
    }
  };

  const drawRef = async (
    ctx: Ctx,
    frame: Rect,
    index: number,
    leased: ReadonlyMap<number, LeasedLayerSurface>
  ): Promise<void> => {
    const ref = entry.layers[index];
    if (!ref) {
      return;
    }
    const held = leased.get(index);
    const layerSurface = held ?? (await deps.getLayerSurface(ref.id));
    try {
      await drawLayer(ctx, frame, ref, layerSurface);
    } finally {
      if (!held) {
        layerSurface.release();
      }
    }
  };

  const landScope = (ctx: Ctx, frame: Rect, scope: GroupCompositeScope, buffer: RasterSurface, at: Rect): void => {
    ctx.save();
    setTransform(ctx, IDENTITY);
    ctx.globalAlpha = scope.opacity;
    ctx.globalCompositeOperation = blendToComposite(scope.blendMode);
    ctx.drawImage(buffer.canvas, at.x - frame.x, at.y - frame.y);
    ctx.restore();
  };

  /** Composites over the members' union, never clipped to the output; members are leased first to measure it. */
  const drawSpatialScope = async (
    ctx: Ctx,
    frame: Rect,
    scope: GroupCompositeScope,
    leased: ReadonlyMap<number, LeasedLayerSurface>
  ): Promise<void> => {
    const held = new Map(leased);
    const acquired: LeasedLayerSurface[] = [];
    try {
      for (let index = scope.start; index < scope.end; index += 1) {
        if (!held.has(index)) {
          const layerSurface = await deps.getLayerSurface(entry.layers[index]!.id);
          acquired.push(layerSurface);
          held.set(index, layerSurface);
        }
      }
      const members: { ref: CompositeLayerRef; rect: Rect }[] = [];
      for (let index = scope.start; index < scope.end; index += 1) {
        members.push({ rect: held.get(index)!.rect, ref: entry.layers[index]! });
      }
      const extent = placedExtent(members);
      if (!extent) {
        return;
      }
      const extra = reserveExcess(areaOf(extent), areaOf(plannedScopeExtent(entry, scope)));
      try {
        const buffer = deps.backend.createSurface(extent.width, extent.height);
        setTransform(buffer.ctx, IDENTITY);
        buffer.ctx.clearRect(0, 0, extent.width, extent.height);
        await renderRange(buffer.ctx, extent, scope.start, scope.end, scope.children, held);
        const pixels = readImageData(buffer, fullRectOf(extent));
        writeImageData(buffer, await evaluate(pixels, scope.adjustments, extent, scope.id), 0, 0);
        landScope(ctx, frame, scope, buffer, extent);
      } finally {
        extra?.release();
      }
    } finally {
      for (const layerSurface of acquired) {
        layerSurface.release();
      }
    }
  };

  const renderRange = async (
    ctx: Ctx,
    frame: Rect,
    start: number,
    end: number,
    scopes: readonly GroupCompositeScope[],
    leased: ReadonlyMap<number, LeasedLayerSurface>
  ): Promise<void> => {
    let scopeIndex = scopes.length - 1;
    for (let i = end - 1; i >= start;) {
      const scope = scopeIndex >= 0 ? scopes[scopeIndex]! : null;
      if (scope && i >= scope.start && i < scope.end) {
        if (requiresWholeSource(scope.adjustments)) {
          await drawSpatialScope(ctx, frame, scope, leased);
        } else {
          const output = fullRectOf(frame);
          const buffer = deps.backend.createSurface(output.width, output.height);
          setTransform(buffer.ctx, IDENTITY);
          buffer.ctx.clearRect(0, 0, output.width, output.height);
          await renderRange(buffer.ctx, frame, scope.start, scope.end, scope.children, leased);
          if (!isIdentityAdjustments(scope.adjustments)) {
            const pixels = readImageData(buffer, output);
            // Group stacks work in document pixels; the buffer is the frame at scale 1.
            applyAdjustments(pixels, scope.adjustments, frame);
            writeImageData(buffer, pixels, 0, 0);
          }
          landScope(ctx, frame, scope, buffer, frame);
        }
        i = scope.start - 1;
        scopeIndex -= 1;
        continue;
      }
      await drawRef(ctx, frame, i, leased);
      i -= 1;
    }
  };

  const output = fullRectOf(bbox);
  const surface = deps.backend.createSurface(output.width, output.height);
  setTransform(surface.ctx, IDENTITY);
  surface.ctx.clearRect(0, 0, surface.width, surface.height);
  await renderRange(surface.ctx, bbox, 0, entry.layers.length, entry.groupScopes ?? [], new Map());
  return surface;
};
