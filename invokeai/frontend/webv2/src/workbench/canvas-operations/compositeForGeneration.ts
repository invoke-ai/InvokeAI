import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { SpatialRunner } from '@workbench/canvas-engine/render/exactAdjustments';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';
import type {
  CompositeEntry,
  CompositeMaskLayerRef,
  CompositePlan,
} from '@workbench/canvas-operations/generationContracts';

import { sha256Hex } from '@platform/browser/sha256';
import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { alphaBounds } from '@workbench/canvas-engine/render/alphaBounds';
import {
  compositeScratchPixels,
  RasterCompositeOverBudgetError,
  renderRasterComposite,
} from '@workbench/canvas-engine/render/rasterComposite';
import { getCompositeLayerBounds } from '@workbench/canvas-operations/generationCompositePlan';

type Ctx = RasterSurface['ctx'];

/** SHA-256 hex of a blob's bytes, via `@platform/browser/sha256` (matches `bitmapStore`). */
const defaultHashBlob = async (blob: Blob): Promise<string> => sha256Hex(await blob.arrayBuffer());

const defaultReadImageData = (surface: RasterSurface, rect: Rect): ImageData =>
  surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height);

const defaultWriteImageData = (surface: RasterSurface, imageData: ImageData, x: number, y: number): void =>
  surface.ctx.putImageData(imageData, x, y);

/** Caller-owned: `byKey` skips unchanged plans, `byHash` reuses uploads of identical pixels. */
export interface CompositeDedupeCache {
  byKey: Map<string, CompositeCacheEntry>;
  byHash: Map<string, CanvasImageUploadResult>;
}

export interface CompositeCacheEntry {
  imageName: string;
  width: number;
  height: number;
  pixelHash: string;
  bboxFullyCovered: boolean;
  /** Whether any pixel is not fully transparent; masks store their coverage here too. */
  bboxHasContent: boolean;
  /** A regional mask's alpha bounds in bbox pixels; only regional entries have them. */
  bounds?: Rect;
}

export const createCompositeDedupeCache = (): CompositeDedupeCache => ({
  byHash: new Map(),
  byKey: new Map(),
});

class BoundedMap<K, V> extends Map<K, V> {
  constructor(private readonly capacity: number) {
    super();
  }

  override set(key: K, value: V): this {
    if (!this.has(key) && this.size >= this.capacity) {
      const oldest = this.keys().next().value;
      if (oldest !== undefined) {
        this.delete(oldest);
      }
    }
    return super.set(key, value);
  }
}

export const createBoundedCompositeDedupeCache = (capacity = 16): CompositeDedupeCache => ({
  byHash: new BoundedMap(capacity),
  byKey: new BoundedMap(capacity),
});

export interface ExecuteCompositePlanDeps {
  backend: {
    createSurface(width: number, height: number): RasterSurface;
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  /** Drawing applies rect.origin before the layer transform. */
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect; release(): void }>;
  uploadImage(blob: Blob): Promise<CanvasImageUploadResult>;
  dedupe: CompositeDedupeCache;
  hashBlob?(blob: Blob): Promise<string>;
  readImageData?(surface: RasterSurface, rect: Rect): ImageData;
  /** Reserves transient raster bytes; over budget refuses with `RasterCompositeOverBudgetError`. */
  reserve?(
    bytes: number
  ):
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
  writeImageData?(surface: RasterSurface, imageData: ImageData, x: number, y: number): void;
  /** Without it a spatial stack refuses rather than compositing without its spatial entries. */
  runSpatial?: SpatialRunner;
  signal?: AbortSignal;
}

const reserveComposite = (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps,
  scratchPixels: number
): { release(): void } => {
  const pixels = Math.max(0, entry.bbox.width) * Math.max(0, entry.bbox.height);
  // Control layers skip the final scan buffer: they are opaque by construction.
  const finalBuffers = entry.kind === 'control-layer' ? 1 : 2;
  const reservation = deps.reserve?.((pixels * finalBuffers + scratchPixels * 2) * 4);
  if (reservation?.status === 'over-budget') {
    throw new RasterCompositeOverBudgetError();
  }
  return reservation?.status === 'ok' ? reservation.lease : { release: () => undefined };
};

export interface CompositeEntryResult {
  key: string;
  imageName: string;
  width: number;
  height: number;
  pixelHash: string;
  reusedUpload: boolean;
}

export interface CompositeResult {
  base: CompositeEntryResult;
  contentBounds: Rect | null;
  bboxFullyCovered: boolean;
  bboxHasContent: boolean;
}

const bboxView = (bbox: Rect): Mat2d => ({ a: 1, b: 0, c: 0, d: 1, e: -bbox.x, f: -bbox.y });

const setTransform = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

/** Geometry-only pre-pass so txt2img needs no upload; actualLayerRects replaces estimates with measurements. */
export const computeCompositeContentBounds = (
  plan: CompositePlan,
  actualLayerRects?: ReadonlyMap<string, Rect>
): Rect | null => {
  const entry = plan.entries.find((e) => e.kind === 'base-raster');
  if (!entry) {
    return null;
  }
  if (!actualLayerRects || actualLayerRects.size === 0) {
    return getCompositeLayerBounds(entry.layers);
  }
  const layers = entry.layers.map((layer) => {
    const rect = actualLayerRects.get(layer.id);
    return rect
      ? {
          ...layer,
          contentOffset: { x: rect.x, y: rect.y },
          contentSize: { height: rect.height, width: rect.width },
        }
      : layer;
  });
  return getCompositeLayerBounds(layers);
};

const scanCoverage = (imageData: ImageData): { fullyOpaque: boolean; hasContent: boolean } => {
  const { data, height, width } = imageData;
  if (width <= 0 || height <= 0) {
    return { fullyOpaque: false, hasContent: false };
  }
  let fullyOpaque = true;
  let hasContent = false;
  for (let i = 3; i < data.length && (fullyOpaque || !hasContent); i += 4) {
    const alpha = data[i] ?? 0;
    fullyOpaque &&= alpha === 255;
    hasContent ||= alpha > 0;
  }
  return { fullyOpaque, hasContent };
};

/** Over black: backend RGB normalization would matte erased pixels over white, creating control signal. */
const flattenControlSurfaceOverBlack = (surface: RasterSurface): void => {
  const { ctx } = surface;
  ctx.save();
  setTransform(ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'destination-over';
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, surface.width, surface.height);
  ctx.restore();
};

type RasterEntryResult = CompositeEntryResult & {
  bboxFullyCovered: boolean;
  bboxHasContent: boolean;
  bounds: Rect | null;
};

const executeRasterEntry = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<RasterEntryResult> => {
  const hashBlob = deps.hashBlob ?? defaultHashBlob;
  const readImageData = deps.readImageData ?? defaultReadImageData;

  const cached = deps.dedupe.byKey.get(entry.key);
  if (cached) {
    return {
      bboxFullyCovered: cached.bboxFullyCovered,
      bboxHasContent: cached.bboxHasContent,
      bounds: cached.bounds ?? null,
      height: cached.height,
      imageName: cached.imageName,
      key: entry.key,
      pixelHash: cached.pixelHash,
      reusedUpload: true,
      width: cached.width,
    };
  }

  const reservation = reserveComposite(entry, deps, compositeScratchPixels(entry));
  try {
    const surface = await renderRasterComposite(entry, deps);
    let bboxFullyCovered: boolean;
    let bboxHasContent: boolean;
    let bounds: Rect | null = null;
    if (entry.kind === 'control-layer') {
      flattenControlSurfaceOverBlack(surface);
      bboxFullyCovered = surface.width > 0 && surface.height > 0;
      bboxHasContent = bboxFullyCovered;
    } else {
      const fullRect: Rect = { height: surface.height, width: surface.width, x: 0, y: 0 };
      // One readback serves both scans; the surface is bbox-sized, so the bounds are bbox pixels.
      const pixels = readImageData(surface, fullRect);
      const coverage = scanCoverage(pixels);
      bboxFullyCovered = coverage.fullyOpaque;
      bboxHasContent = coverage.hasContent;
      bounds = entry.kind === 'regional-mask' ? alphaBounds(pixels) : null;
    }

    const blob = await deps.backend.encodeSurface(surface);
    const pixelHash = await hashBlob(blob);

    let upload = deps.dedupe.byHash.get(pixelHash);
    let reusedUpload = true;
    if (!upload) {
      upload = await deps.uploadImage(blob);
      deps.dedupe.byHash.set(pixelHash, upload);
      reusedUpload = false;
    }

    deps.dedupe.byKey.set(entry.key, {
      bboxFullyCovered,
      bboxHasContent,
      ...(bounds ? { bounds } : {}),
      height: upload.height,
      imageName: upload.imageName,
      pixelHash,
      width: upload.width,
    });

    return {
      bboxFullyCovered,
      bboxHasContent,
      bounds,
      height: upload.height,
      imageName: upload.imageName,
      key: entry.key,
      pixelHash,
      reusedUpload,
      width: upload.width,
    };
  } finally {
    reservation.release();
  }
};

export const executeCompositePlan = async (
  plan: CompositePlan,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeResult> => {
  const entry = plan.entries.find((e) => e.kind === 'base-raster');
  if (!entry) {
    throw new Error('executeCompositePlan: plan has no base-raster entry');
  }

  const contentBounds = getCompositeLayerBounds(entry.layers);
  const { bboxFullyCovered, bboxHasContent, bounds: _bounds, ...base } = await executeRasterEntry(entry, deps);

  return { base, bboxFullyCovered, bboxHasContent, contentBounds };
};

export const executeControlComposite = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeEntryResult> => {
  const {
    bboxFullyCovered: _bboxFullyCovered,
    bboxHasContent: _bboxHasContent,
    bounds: _bounds,
    ...result
  } = await executeRasterEntry(entry, deps);
  return result;
};

export interface RegionalMaskCompositeResult extends CompositeEntryResult {
  bounds: Rect;
}

/** Alpha preserved for alpha_mask_to_tensor. */
export const executeRegionalMaskComposite = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<RegionalMaskCompositeResult> => {
  const {
    bboxFullyCovered: _bboxFullyCovered,
    bboxHasContent: _bboxHasContent,
    bounds,
    ...result
  } = await executeRasterEntry(entry, deps);
  if (!bounds) {
    throw new Error(`executeRegionalMaskComposite: a ${entry.kind} entry is not a regional mask.`);
  }
  return { ...result, bounds };
};

/** Darkening these masks preserves legacy semantics: dark inpaints, white keeps. */
export const toGrayscaleMaskPixels = (imageData: ImageData, attributeValue: number): void => {
  const { data } = imageData;
  const masked = Math.max(0, Math.min(255, 255 - Math.round(255 * attributeValue)));
  for (let i = 0; i + 3 < data.length; i += 4) {
    const gray = (data[i + 3] ?? 0) > 127 ? masked : 255;
    data[i] = gray;
    data[i + 1] = gray;
    data[i + 2] = gray;
    data[i + 3] = 255;
  }
};

const maskLayerMatrix = (ref: CompositeMaskLayerRef): Mat2d =>
  fromTRS(
    { x: ref.transform.x, y: ref.transform.y },
    ref.transform.rotation,
    ref.transform.scaleX,
    ref.transform.scaleY
  );

const hasNonWhitePixel = (imageData: ImageData): boolean => {
  const { data, height, width } = imageData;
  if (width <= 0 || height <= 0) {
    return false;
  }
  for (let i = 0; i + 3 < data.length; i += 4) {
    if ((data[i] ?? 255) < 255) {
      return true;
    }
  }
  return false;
};

export interface MaskCompositeResult {
  key: string;
  imageName: string;
  width: number;
  height: number;
  pixelHash: string;
  reusedUpload: boolean;
  hasContent: boolean;
}

const compositeMaskEntry = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps,
  writeImageData: (surface: RasterSurface, imageData: ImageData, x: number, y: number) => void,
  readImageData: (surface: RasterSurface, rect: Rect) => ImageData
): Promise<RasterSurface> => {
  const { bbox } = entry;
  const maskLayers = entry.maskLayers ?? [];
  const width = Math.max(0, bbox.width);
  const height = Math.max(0, bbox.height);
  const accumulator = deps.backend.createSurface(width, height);
  const accCtx = accumulator.ctx;

  setTransform(accCtx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  accCtx.fillStyle = 'white';
  accCtx.fillRect(0, 0, width, height);

  const view = bboxView(bbox);
  const fullRect: Rect = { height, width, x: 0, y: 0 };

  for (const ref of maskLayers) {
    const temp = deps.backend.createSurface(width, height);
    const tempCtx = temp.ctx;
    setTransform(tempCtx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
    tempCtx.clearRect(0, 0, width, height);
    const layerSurface = await deps.getLayerSurface(ref.id);
    try {
      setTransform(tempCtx, multiply(view, maskLayerMatrix(ref)));
      tempCtx.drawImage(layerSurface.surface.canvas, layerSurface.rect.x, layerSurface.rect.y);
    } finally {
      layerSurface.release();
    }

    const pixels = readImageData(temp, fullRect);
    toGrayscaleMaskPixels(pixels, ref.attributeValue);
    writeImageData(temp, pixels, 0, 0);

    // Darken-combine onto the accumulator (min per channel), matching legacy.
    setTransform(accCtx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
    accCtx.globalAlpha = 1;
    accCtx.globalCompositeOperation = 'darken';
    accCtx.drawImage(temp.canvas, 0, 0);
  }

  accCtx.globalCompositeOperation = 'source-over';
  return accumulator;
};

export const executeMaskComposite = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<MaskCompositeResult> => {
  const hashBlob = deps.hashBlob ?? defaultHashBlob;
  const readImageData = deps.readImageData ?? defaultReadImageData;
  const writeImageData = deps.writeImageData ?? defaultWriteImageData;

  const cached = deps.dedupe.byKey.get(entry.key);
  if (cached) {
    return {
      hasContent: cached.bboxFullyCovered,
      height: cached.height,
      imageName: cached.imageName,
      key: entry.key,
      pixelHash: cached.pixelHash,
      reusedUpload: true,
      width: cached.width,
    };
  }

  const reservation = reserveComposite(
    entry,
    deps,
    (entry.maskLayers?.length ?? 0) * Math.max(0, entry.bbox.width) * Math.max(0, entry.bbox.height)
  );
  try {
    const surface = await compositeMaskEntry(entry, deps, writeImageData, readImageData);
    const hasContent = hasNonWhitePixel(
      readImageData(surface, { height: surface.height, width: surface.width, x: 0, y: 0 })
    );

    const blob = await deps.backend.encodeSurface(surface);
    const pixelHash = await hashBlob(blob);

    let upload = deps.dedupe.byHash.get(pixelHash);
    let reusedUpload = true;
    if (!upload) {
      upload = await deps.uploadImage(blob);
      deps.dedupe.byHash.set(pixelHash, upload);
      reusedUpload = false;
    }

    // Reuse the `bboxFullyCovered` slot to persist `hasContent` for this key.
    deps.dedupe.byKey.set(entry.key, {
      bboxFullyCovered: hasContent,
      bboxHasContent: hasContent,
      height: upload.height,
      imageName: upload.imageName,
      pixelHash,
      width: upload.width,
    });

    return {
      hasContent,
      height: upload.height,
      imageName: upload.imageName,
      key: entry.key,
      pixelHash,
      reusedUpload,
      width: upload.width,
    };
  } finally {
    reservation.release();
  }
};
