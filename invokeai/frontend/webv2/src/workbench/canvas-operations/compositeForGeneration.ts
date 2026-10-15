/**
 * Execute a {@link CompositePlan} with injected raster, encoding, hashing, and upload dependencies. Composite
 * enabled layers in z-order through their transforms, opacity, and blend modes into the bbox, then compute
 * document-space bounds and alpha coverage. The caller-owned {@link CompositeDedupeCache} skips unchanged plans
 * and reuses uploads with identical pixel hashes.
 */

import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';
import type {
  CompositeEntry,
  CompositeMaskLayerRef,
  CompositePlan,
} from '@workbench/canvas-operations/generationContracts';

import { sha256Hex } from '@platform/browser/sha256';
import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { renderRasterComposite } from '@workbench/canvas-engine/render/rasterComposite';
import { getCompositeLayerBounds } from '@workbench/canvas-operations/generationCompositePlan';

type Ctx = RasterSurface['ctx'];

/** SHA-256 hex of a blob's bytes, via `@platform/browser/sha256` (matches `bitmapStore`). */
const defaultHashBlob = async (blob: Blob): Promise<string> => sha256Hex(await blob.arrayBuffer());

/** Reads a surface's pixels via its 2D context (real DOM path; injectable for tests). */
const defaultReadImageData = (surface: RasterSurface, rect: Rect): ImageData =>
  surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height);

/** Writes pixels back to a surface's 2D context (real DOM path; injectable for tests). */
const defaultWriteImageData = (surface: RasterSurface, imageData: ImageData, x: number, y: number): void =>
  surface.ctx.putImageData(imageData, x, y);

/**
 * A caller-owned dedupe cache, persisted across executor calls:
 * - `byKey`: plan key → its last result, so an unchanged plan skips all work.
 * - `byHash`: pixel hash → uploaded image, so a changed plan with identical
 *   pixels reuses the upload.
 */
export interface CompositeDedupeCache {
  byKey: Map<string, CompositeCacheEntry>;
  byHash: Map<string, CanvasImageUploadResult>;
}

/** A cached executor result for one plan key. */
export interface CompositeCacheEntry {
  imageName: string;
  width: number;
  height: number;
  pixelHash: string;
  bboxFullyCovered: boolean;
}

/** Creates an empty {@link CompositeDedupeCache}. */
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

/** Canvas-owned bounded cache used across generation transactions. */
export const createBoundedCompositeDedupeCache = (capacity = 16): CompositeDedupeCache => ({
  byHash: new BoundedMap(capacity),
  byKey: new BoundedMap(capacity),
});

/** Injected dependencies for {@link executeCompositePlan}. */
export interface ExecuteCompositePlanDeps {
  /** Surface factory + encoder seam (usually the engine's `RasterBackend`). */
  backend: {
    createSurface(width: number, height: number): RasterSurface;
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  /**
   * Return the rasterized surface and its layer-local content rect; drawing applies rect.origin before the layer
   * transform, then releases it.
   */
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect; release(): void }>;
  /**
   * Upload the composite and return its server name and dimensions; the engine marks generation composites
   * intermediate.
   */
  uploadImage(blob: Blob): Promise<CanvasImageUploadResult>;
  /** Persistent dedupe state (see {@link CompositeDedupeCache}). */
  dedupe: CompositeDedupeCache;
  /** Content-hashes a blob (default SHA-256 hex via `@platform/browser/sha256`). */
  hashBlob?(blob: Blob): Promise<string>;
  /** Reads a surface region's pixels for the coverage scan (default `getImageData`). */
  readImageData?(surface: RasterSurface, rect: Rect): ImageData;
  /** Reserves transient raster bytes for the complete composite operation. */
  reserve?(
    bytes: number
  ):
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
  /** Writes pixels back to a surface (default `putImageData`; injectable for tests). */
  writeImageData?(surface: RasterSurface, imageData: ImageData, x: number, y: number): void;
}

export class CompositeOverBudgetError extends Error {
  constructor() {
    super('The canvas composite exceeds the available raster memory budget.');
    this.name = 'CompositeOverBudgetError';
  }
}

const reserveComposite = (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps,
  layerCount: number
): { release(): void } => {
  const pixels = Math.max(0, entry.bbox.width) * Math.max(0, entry.bbox.height);
  // Final/accumulator surface plus a final scan ImageData (except for control
  // layers, which are made opaque by construction), plus one temporary surface
  // and one ImageData buffer for every adjusted/mask layer.
  const finalBuffers = entry.kind === 'control-layer' ? 1 : 2;
  const reservation = deps.reserve?.(pixels * 4 * (finalBuffers + layerCount * 2));
  if (reservation?.status === 'over-budget') {
    throw new CompositeOverBudgetError();
  }
  return reservation?.status === 'ok' ? reservation.lease : { release: () => undefined };
};

/** The base composite's upload identity + hash. */
export interface CompositeEntryResult {
  /** The entry's stable plan key. */
  key: string;
  /** The uploaded (or reused) image name. */
  imageName: string;
  width: number;
  height: number;
  /** SHA-256 of the composited PNG bytes. */
  pixelHash: string;
  /** True when this result came from cache/dedupe (no upload happened this call). */
  reusedUpload: boolean;
}

/** The full result of executing a plan: the base image + mode-detection geometry. */
export interface CompositeResult {
  base: CompositeEntryResult;
  /** Union of enabled raster content bounds in document space, or `null`. */
  contentBounds: Rect | null;
  /** Whether the composited bbox surface is fully opaque (no transparent holes). */
  bboxFullyCovered: boolean;
}

/** Document→bbox translate matrix (the "view" the entry is composited under). */
const bboxView = (bbox: Rect): Mat2d => ({ a: 1, b: 0, c: 0, d: 1, e: -bbox.x, f: -bbox.y });

/** Applies a matrix to a 2D context's transform. */
const setTransform = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

/**
 * Union base-raster bounds in document space, or null without enabled content. This geometry-only pre-pass avoids
 * uploads for txt2img; actualLayerRects replaces estimates with captured browser measurements.
 */
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

/** True when every pixel of `imageData` is fully opaque (alpha === 255). Empty → false. */
const isFullyOpaque = (imageData: ImageData): boolean => {
  const { data, height, width } = imageData;
  if (width <= 0 || height <= 0) {
    return false;
  }
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 255) {
      return false;
    }
  }
  return true;
};

/**
 * Flatten control composites over black: backend RGB normalization would otherwise matte erased pixels over white,
 * creating control signal. Editable surfaces retain alpha.
 */
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

/**
 * Share raster composition, coverage scanning, encoding, and plan-key/content-hash upload deduplication between
 * {@link executeCompositePlan} and {@link executeControlComposite}.
 */
const executeRasterEntry = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeEntryResult & { bboxFullyCovered: boolean }> => {
  const hashBlob = deps.hashBlob ?? defaultHashBlob;
  const readImageData = deps.readImageData ?? defaultReadImageData;

  const cached = deps.dedupe.byKey.get(entry.key);
  if (cached) {
    return {
      bboxFullyCovered: cached.bboxFullyCovered,
      height: cached.height,
      imageName: cached.imageName,
      key: entry.key,
      pixelHash: cached.pixelHash,
      reusedUpload: true,
      width: cached.width,
    };
  }

  const countGroupScopes = (scopes: typeof entry.groupScopes): number =>
    (scopes ?? []).reduce((total, scope) => total + 1 + countGroupScopes(scope.children), 0);
  const reservation = reserveComposite(
    entry,
    deps,
    entry.layers.filter((layer) => layer.adjustments !== undefined).length + countGroupScopes(entry.groupScopes)
  );
  try {
    const surface = await renderRasterComposite(entry, deps);
    let bboxFullyCovered: boolean;
    if (entry.kind === 'control-layer') {
      flattenControlSurfaceOverBlack(surface);
      bboxFullyCovered = surface.width > 0 && surface.height > 0;
    } else {
      const fullRect: Rect = { height: surface.height, width: surface.width, x: 0, y: 0 };
      bboxFullyCovered = isFullyOpaque(readImageData(surface, fullRect));
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
      height: upload.height,
      imageName: upload.imageName,
      pixelHash,
      width: upload.width,
    });

    return {
      bboxFullyCovered,
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
  const { bboxFullyCovered, ...base } = await executeRasterEntry(entry, deps);

  return { base, bboxFullyCovered, contentBounds };
};

/** Composite one control layer over the bbox; the shared cache avoids re-uploading unchanged controls. */
export const executeControlComposite = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeEntryResult> => {
  const { bboxFullyCovered: _bboxFullyCovered, ...result } = await executeRasterEntry(entry, deps);
  return result;
};

/** Composite one regional mask with alpha preserved for alpha_mask_to_tensor; share the base/control dedupe cache. */
export const executeRegionalMaskComposite = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeEntryResult> => {
  const { bboxFullyCovered: _bboxFullyCovered, ...result } = await executeRasterEntry(entry, deps);
  return result;
};

// ---- Grayscale mask composite (inpaint/outpaint) ---------------------------

/**
 * Convert alpha > 127 to 255 - round(255 * attributeValue), otherwise white, with opaque alpha. Darken-compositing
 * these masks preserves legacy semantics: dark inpaints, white keeps.
 */
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

/** The local→document transform matrix for a mask layer ref. */
const maskLayerMatrix = (ref: CompositeMaskLayerRef): Mat2d =>
  fromTRS(
    { x: ref.transform.x, y: ref.transform.y },
    ref.transform.rotation,
    ref.transform.scaleX,
    ref.transform.scaleY
  );

/** True when any pixel is non-white (a masked region exists). Empty → false. */
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

/** The result of a grayscale mask composite: its upload identity + whether it has any masked pixels. */
export interface MaskCompositeResult {
  key: string;
  imageName: string;
  width: number;
  height: number;
  pixelHash: string;
  reusedUpload: boolean;
  /** True when the composite contains a masked (non-white) region within the bbox. */
  hasContent: boolean;
}

/** Composites one mask entry's layers into a grayscale bbox surface (white bg, darken combine). */
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

  // White background: unmasked area stays white ("keep").
  setTransform(accCtx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  accCtx.fillStyle = 'white';
  accCtx.fillRect(0, 0, width, height);

  const view = bboxView(bbox);
  const fullRect: Rect = { height, width, x: 0, y: 0 };

  for (const ref of maskLayers) {
    // Render the mask alpha into a temp bbox surface through its transform.
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

    // Convert its alpha to grayscale by the layer's attribute value.
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

/**
 * Composite inpaint/noise masks over white, scan mask coverage, and reuse the caller-owned plan-key/content-hash
 * upload cache.
 */
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

  const reservation = reserveComposite(entry, deps, entry.maskLayers?.length ?? 0);
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
