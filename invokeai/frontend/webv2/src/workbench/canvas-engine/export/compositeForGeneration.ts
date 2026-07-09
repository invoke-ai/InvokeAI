/**
 * The composite-plan executor: turns a {@link CompositePlan} into an uploaded,
 * bbox-sized base image plus the geometry the mode detector needs.
 *
 * This is the impure half of the canvas → generation pipeline (the planner in
 * `generation/canvas/compositePlan.ts` is pure). It lives under `canvas-engine`
 * and has zero React: every side-effecting dependency (surface allocation,
 * layer rasterization, encode, hash, upload) is injected, so it runs in node
 * tests against `render/raster.testStub.ts` + a mock uploader — no fetch, no DOM.
 *
 * For each `base-raster` entry it:
 * 1. Composites the entry's enabled raster layers, in z-order, through each
 *    layer's transform / opacity / blend mode, cropped to the bbox onto a
 *    bbox-sized surface (following `render/compositor.ts`'s draw model, where
 *    the "view" is a bbox translate). Layers are rasterized on demand via the
 *    injected {@link ExecuteCompositePlanDeps.getLayerSurface}.
 * 2. Computes `contentBounds` (union of the entry's layer bounds in document
 *    space) and `bboxFullyCovered` (an alpha scan of the composited surface).
 * 3. Encodes → PNG blob → SHA-256, then dedupes: an unchanged plan key reuses
 *    the previous upload with zero new work, and a changed plan whose pixels
 *    hash identically reuses the previous upload via the content hash.
 *
 * Dedupe state lives in a caller-owned {@link CompositeDedupeCache} passed
 * through `deps`, so it persists across invokes while the function stays a plain
 * `(plan, deps) → result`.
 */

import type { CanvasImageUploadResult } from '@workbench/canvas-engine/backend/canvasImages';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';
import type {
  CompositeEntry,
  CompositeLayerRef,
  CompositeMaskLayerRef,
  CompositePlan,
} from '@workbench/generation/canvas/types';

import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { applyAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';

type Ctx = RasterSurface['ctx'];

/** SHA-256 hex of a blob's bytes, via the Web Crypto API (matches `bitmapStore`). */
const defaultHashBlob = async (blob: Blob): Promise<string> => {
  const buffer = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  const bytes = new Uint8Array(digest);
  let hex = '';
  for (const byte of bytes) {
    hex += byte.toString(16).padStart(2, '0');
  }
  return hex;
};

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

/** Injected dependencies for {@link executeCompositePlan}. */
export interface ExecuteCompositePlanDeps {
  /** Surface factory + encoder seam (usually the engine's `RasterBackend`). */
  backend: {
    createSurface(width: number, height: number): RasterSurface;
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  /**
   * Ensures a layer's cache is rasterized and returns its surface plus the
   * content `rect` (layer-local origin/size) those pixels occupy. The executor
   * draws the surface at `rect.origin` (then through the layer transform). The
   * engine wires this to its rasterize path; tests return a stub.
   */
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect }>;
  /**
   * Uploads a composited blob and resolves to its server image name + dims. The
   * engine wires this to `uploadCanvasImage(blob, { isIntermediate: true })`.
   */
  uploadImage(blob: Blob): Promise<CanvasImageUploadResult>;
  /** Persistent dedupe state (see {@link CompositeDedupeCache}). */
  dedupe: CompositeDedupeCache;
  /** Content-hashes a blob (default SHA-256 hex via `crypto.subtle`). */
  hashBlob?(blob: Blob): Promise<string>;
  /** Reads a surface region's pixels for the coverage scan (default `getImageData`). */
  readImageData?(surface: RasterSurface, rect: Rect): ImageData;
  /** Writes pixels back to a surface (default `putImageData`; injectable for tests). */
  writeImageData?(surface: RasterSurface, imageData: ImageData, x: number, y: number): void;
}

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

/** The layer's local→document transform matrix. */
const layerMatrix = (ref: CompositeLayerRef): Mat2d =>
  fromTRS(
    { x: ref.transform.x, y: ref.transform.y },
    ref.transform.rotation,
    ref.transform.scaleX,
    ref.transform.scaleY
  );

/**
 * Union of a plan's base-raster content bounds in document space, or `null`
 * when the plan has no enabled raster content. Pure geometry (no pixels, no
 * upload), so the invoke orchestrator can run it as a bounds-only pre-pass to
 * decide txt2img (no bbox overlap) *before* paying for a composite/encode/upload.
 */
export const computeCompositeContentBounds = (plan: CompositePlan): Rect | null => {
  const entry = plan.entries.find((e) => e.kind === 'base-raster');
  return entry ? computeContentBounds(entry.layers) : null;
};

/** Union of the entry's layer bounds in document space, or `null` when empty. */
const computeContentBounds = (layers: CompositeLayerRef[]): Rect | null => {
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

/** Composites an entry's layers, in z-order, onto a fresh bbox-sized surface. */
const compositeEntry = async (entry: CompositeEntry, deps: ExecuteCompositePlanDeps): Promise<RasterSurface> => {
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

/**
 * Composites, scans coverage, encodes, hashes, dedupes, and (when needed)
 * uploads a single raster-style entry (`base-raster` or `control-layer`).
 * Shared by {@link executeCompositePlan} and {@link executeControlComposite} so
 * both go through the identical plan-key + content-hash dedupe path.
 */
const executeRasterEntry = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeEntryResult & { bboxFullyCovered: boolean }> => {
  const hashBlob = deps.hashBlob ?? defaultHashBlob;
  const readImageData = deps.readImageData ?? defaultReadImageData;

  // Plan-key hit: nothing that affects these pixels changed — reuse everything,
  // no composite, no encode, no upload.
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

  const surface = await compositeEntry(entry, deps);
  const bboxFullyCovered = isFullyOpaque(
    readImageData(surface, { height: surface.height, width: surface.width, x: 0, y: 0 })
  );

  const blob = await deps.backend.encodeSurface(surface);
  const pixelHash = await hashBlob(blob);

  // Content-hash dedupe: identical pixels (even under a different key) reuse the
  // already-uploaded image — no second upload.
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
};

/**
 * Executes `plan`'s base-raster composite: composites, scans coverage, encodes,
 * dedupes, and (when needed) uploads. Returns the base image identity plus the
 * `contentBounds` / `bboxFullyCovered` facts the mode detector consumes.
 */
export const executeCompositePlan = async (
  plan: CompositePlan,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeResult> => {
  const entry = plan.entries.find((e) => e.kind === 'base-raster');
  if (!entry) {
    throw new Error('executeCompositePlan: plan has no base-raster entry');
  }

  const contentBounds = computeContentBounds(entry.layers);
  const { bboxFullyCovered, ...base } = await executeRasterEntry(entry, deps);

  return { base, bboxFullyCovered, contentBounds };
};

/**
 * Executes a single `control-layer` composite entry (one enabled control layer,
 * composited alone over the bbox). Reuses the same dedupe cache as the base
 * composite, so an unchanged control layer skips re-upload across invokes.
 */
export const executeControlComposite = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeEntryResult> => {
  const { bboxFullyCovered: _bboxFullyCovered, ...result } = await executeRasterEntry(entry, deps);
  return result;
};

/**
 * Executes a single `regional-mask` composite entry (one enabled regional-guidance
 * layer's mask, composited alone over the bbox with its alpha preserved). The
 * uploaded image's alpha channel is the region coverage, consumed by
 * `alpha_mask_to_tensor`. Reuses the same dedupe cache + raster path as the base
 * / control composites, so an unchanged region mask skips re-upload.
 */
export const executeRegionalMaskComposite = async (
  entry: CompositeEntry,
  deps: ExecuteCompositePlanDeps
): Promise<CompositeEntryResult> => {
  const { bboxFullyCovered: _bboxFullyCovered, ...result } = await executeRasterEntry(entry, deps);
  return result;
};

// ---- Grayscale mask composite (inpaint/outpaint) ---------------------------

/**
 * Converts a mask layer's alpha into legacy grayscale, in place: a masked pixel
 * (alpha > 127) becomes `255 - round(255 * attributeValue)` (black at full
 * strength), an unmasked pixel becomes white (255); alpha is forced opaque. This
 * mirrors `getGrayscaleMaskCompositeImageDTO`'s per-pixel step so multiple masks
 * can be darken-composited over a white background (dark = inpaint, white = keep).
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
    setTransform(tempCtx, multiply(view, maskLayerMatrix(ref)));
    tempCtx.drawImage(layerSurface.surface.canvas, layerSurface.rect.x, layerSurface.rect.y);

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
 * Executes a grayscale mask composite entry (`inpaint-mask` / `noise-mask`):
 * composites the mask layers into a white-backed grayscale image, scans whether
 * any masked region exists, then encodes / dedupes / uploads exactly like the
 * base composite. Content-hash + plan-key dedupe reuse the same caller-owned
 * cache so an unchanged mask skips re-upload.
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
};
