/**
 * Export raster layers to a Photoshop (.psd) document.
 *
 * Split into a PURE planner and an IMPURE executor, mirroring the
 * planner/executor split of `compositeForGeneration.ts`:
 *
 * - {@link planPsdExport} is pure geometry (no DOM, no `ag-psd`, no engine): it
 *   turns each raster layer's transform + content rect into a PSD layer entry
 *   (position, opacity, blend, hidden, order) and the document bounds. Unit
 *   tested in node.
 * - {@link executePsdExport} is the side-effecting half: it bakes each layer's
 *   pixels through the {@link RasterBackend} seam, lazily imports `ag-psd`
 *   (`writePsd`) at call time so the library never enters the main bundle, and
 *   triggers a browser download. Verified by types + manual QA (opening the PSD).
 *
 * ### Conventions
 * - **Order.** The canvas document stores layers top-first (index 0 = top-most).
 *   ag-psd's `children` array is BOTTOM-to-top (`children[0]` is the bottom-most
 *   layer, written first to the PSD layer records, which the format stores
 *   bottom-up). So the plan reverses the top-first input into bottom-to-top.
 * - **Bounds.** The PSD canvas is the union of every EXPORTED layer's
 *   world-space (document-space) content AABB — document/bbox-independent. An
 *   empty union means nothing to export.
 * - **Opacity.** ag-psd's `Layer.opacity` is 0..1 (the writer multiplies by 255
 *   internally), NOT 0..255. Our `layer.opacity` is already 0..1, so it passes
 *   through unchanged (clamped).
 * - **Hidden.** Every raster layer with content is exported; hidden (disabled)
 *   layers are written with `hidden: true` rather than dropped.
 * - **Adjustments.** Non-destructive raster adjustments are BAKED into the
 *   layer's pixels (PSD has no matching non-destructive representation we emit),
 *   exactly as `compositeForGeneration` bakes them, so the PSD matches what the
 *   user sees. Opacity/blend stay as PSD layer properties (not baked).
 */

import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';
import type { CanvasAdjustmentsContract, CanvasBlendMode } from '@workbench/types';
import type { BlendMode, Layer as AgPsdLayer, Psd } from 'ag-psd';

import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { applyAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';

/**
 * Maximum PSD side length. ag-psd/Photoshop tolerate up to 300000px, but a
 * multi-gigabyte export from an unbounded-canvas union helps nobody — refuse
 * past a sane cap and tell the user. Legacy Photoshop's own PSD limit is 30000.
 */
export const PSD_MAX_DIMENSION = 30000;

/** A canvas layer transform (TRS), duplicated to keep this module contract-light. */
export interface PsdLayerTransform {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
}

/**
 * Maps a document blend mode to ag-psd's blend key. Every blend mode the canvas
 * supports has a direct PSD equivalent (Photoshop is the origin of these modes),
 * so this is total; an unknown value falls back to 'normal' and is reported via
 * {@link PsdExportOk.unmappedBlends}.
 */
const BLEND_MODE_TO_PSD: Record<CanvasBlendMode, BlendMode> = {
  color: 'color',
  'color-burn': 'color burn',
  'color-dodge': 'color dodge',
  darken: 'darken',
  difference: 'difference',
  exclusion: 'exclusion',
  'hard-light': 'hard light',
  hue: 'hue',
  lighten: 'lighten',
  luminosity: 'luminosity',
  multiply: 'multiply',
  normal: 'normal',
  overlay: 'overlay',
  saturation: 'saturation',
  screen: 'screen',
  'soft-light': 'soft light',
};

/** The ag-psd blend key for a document blend mode ('normal' for anything unmapped). */
export const blendModeToPsd = (mode: CanvasBlendMode): BlendMode => BLEND_MODE_TO_PSD[mode] ?? 'normal';

/** One raster layer's export-relevant facts, in the document's top-first order. */
export interface PsdExportLayerInput {
  id: string;
  name: string;
  transform: PsdLayerTransform;
  /** The layer's content rect in LOCAL space (origin may be negative). */
  contentRect: Rect;
  /** 0..1. */
  opacity: number;
  blendMode: CanvasBlendMode;
  /** Hidden (disabled) layers are exported with `hidden: true`, not dropped. */
  isEnabled: boolean;
  /** Non-destructive adjustments to bake into the layer's pixels, if any. */
  adjustments?: CanvasAdjustmentsContract;
}

/** A single planned PSD layer (already in ag-psd bottom-to-top order). */
export interface PsdPlanLayer {
  id: string;
  name: string;
  /** Position within the PSD canvas (relative to the union origin). */
  left: number;
  top: number;
  right: number;
  bottom: number;
  /** The layer's world-space AABB (document space) the executor bakes into. */
  worldRect: Rect;
  transform: PsdLayerTransform;
  /** Layer-local content rect (executor draws the cache at its origin). */
  contentRect: Rect;
  /** 0..1 (ag-psd's range). */
  opacity: number;
  blendMode: BlendMode;
  /** Canvas `globalCompositeOperation` for the flattened composite preview. */
  compositeBlend: GlobalCompositeOperation;
  hidden: boolean;
  adjustments?: CanvasAdjustmentsContract;
}

/** A successful export plan. */
export interface PsdExportOk {
  status: 'ok';
  /** PSD canvas dimensions (= union bounds). */
  width: number;
  height: number;
  /** The union bounds in document space (origin is the PSD's (0,0)). */
  canvasRect: Rect;
  /** Layers in ag-psd order (bottom-to-top). */
  layers: PsdPlanLayer[];
  /** Distinct blend modes that had no PSD equivalent (fell back to 'normal'). */
  unmappedBlends: string[];
}

/** The plan, or a refusal (`empty` / `too-large`). */
export type PsdExportPlan = PsdExportOk | { status: 'empty' } | { status: 'too-large'; width: number; height: number };

/** Options for {@link planPsdExport}. */
export interface PlanPsdExportOptions {
  /** Override the per-side dimension cap (default {@link PSD_MAX_DIMENSION}). */
  maxDimension?: number;
}

const layerMatrix = (t: PsdLayerTransform): Mat2d => fromTRS({ x: t.x, y: t.y }, t.rotation, t.scaleX, t.scaleY);

/** Clamps to [0, 1] (defensive against out-of-range opacities). */
const clamp01 = (value: number): number => (value < 0 ? 0 : value > 1 ? 1 : value);

/**
 * Plans a PSD export from raster layers (top-first). Computes each layer's
 * world-space AABB, unions them for the PSD canvas, and produces per-layer PSD
 * entries in bottom-to-top order. Layers with no content (empty rect, or a
 * degenerate zero-area transform) contribute nothing and are omitted. Returns
 * `empty` when nothing has content and `too-large` when the union exceeds the
 * dimension cap.
 */
export const planPsdExport = (
  inputs: readonly PsdExportLayerInput[],
  options: PlanPsdExportOptions = {}
): PsdExportPlan => {
  const maxDimension = options.maxDimension ?? PSD_MAX_DIMENSION;

  // World-space AABB per layer (null = no content: empty local rect or a
  // zero-area transform that collapses the bounds).
  const withBounds = inputs.map((input) => {
    if (isEmpty(input.contentRect)) {
      return { input, worldRect: null as Rect | null };
    }
    const worldRect = roundOut(transformBounds(layerMatrix(input.transform), input.contentRect));
    return { input, worldRect: isEmpty(worldRect) ? null : worldRect };
  });

  let bounds: Rect | null = null;
  for (const { worldRect } of withBounds) {
    if (worldRect) {
      bounds = bounds === null ? worldRect : union(bounds, worldRect);
    }
  }
  if (bounds === null || isEmpty(bounds)) {
    return { status: 'empty' };
  }
  const canvasRect = roundOut(bounds);
  if (canvasRect.width > maxDimension || canvasRect.height > maxDimension) {
    return { height: canvasRect.height, status: 'too-large', width: canvasRect.width };
  }

  const unmappedBlends = new Set<string>();
  // ag-psd order is bottom-to-top; inputs are top-first, so reverse. Layers
  // without content are dropped.
  const layers: PsdPlanLayer[] = [];
  for (let i = withBounds.length - 1; i >= 0; i -= 1) {
    const { input, worldRect } = withBounds[i]!;
    if (!worldRect) {
      continue;
    }
    const mapped = BLEND_MODE_TO_PSD[input.blendMode];
    if (!mapped) {
      unmappedBlends.add(input.blendMode);
    }
    const left = worldRect.x - canvasRect.x;
    const top = worldRect.y - canvasRect.y;
    layers.push({
      adjustments: input.adjustments,
      blendMode: mapped ?? 'normal',
      bottom: top + worldRect.height,
      compositeBlend: blendToComposite(input.blendMode),
      contentRect: input.contentRect,
      hidden: !input.isEnabled,
      id: input.id,
      left,
      name: input.name,
      opacity: clamp01(input.opacity),
      right: left + worldRect.width,
      top,
      transform: input.transform,
      worldRect,
    });
  }

  return {
    canvasRect,
    height: canvasRect.height,
    layers,
    status: 'ok',
    unmappedBlends: [...unmappedBlends],
    width: canvasRect.width,
  };
};

// ---- Executor (impure) -----------------------------------------------------

type Ctx = RasterSurface['ctx'];

/** Reads a surface region's pixels (real DOM path; injectable for tests). */
const defaultReadImageData = (surface: RasterSurface, rect: Rect): ImageData =>
  surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height);

/** Writes pixels back to a surface (real DOM path; injectable for tests). */
const defaultWriteImageData = (surface: RasterSurface, imageData: ImageData, x: number, y: number): void =>
  surface.ctx.putImageData(imageData, x, y);

/**
 * Serializes a {@link Psd} to bytes via a LAZILY-imported `ag-psd`, so the
 * library is never pulled into the main bundle (Vite code-splits the dynamic
 * import into its own chunk, loaded only when an export runs).
 */
const defaultWritePsd = async (psd: Psd): Promise<ArrayBuffer> => {
  const { writePsd } = await import('ag-psd');
  return writePsd(psd);
};

/** Triggers a browser download of the PSD bytes (Blob + anchor click). */
const defaultDownload = (data: ArrayBuffer, fileName: string): void => {
  const blob = new Blob([data], { type: 'image/vnd.adobe.photoshop' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
};

/** Injected dependencies for {@link executePsdExport}. */
export interface ExecutePsdExportDeps {
  /** Surface factory (usually the engine's `RasterBackend`). */
  backend: { createSurface(width: number, height: number): RasterSurface };
  /** Cancels before the next background allocation or side effect. */
  signal?: AbortSignal;
  /**
   * Ensures a layer's cache is rasterized and returns its surface plus the
   * content `rect` (layer-local origin/size) those pixels occupy. The engine
   * wires this to its rasterize path (reading live paint caches when present).
   */
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect }>;
  /** Reads a surface region's pixels (default `getImageData`). */
  readImageData?(surface: RasterSurface, rect: Rect): ImageData;
  /** Writes pixels back to a surface (default `putImageData`). */
  writeImageData?(surface: RasterSurface, imageData: ImageData, x: number, y: number): void;
  /** Serializes a PSD to bytes (default: lazy `ag-psd` `writePsd`). */
  writePsd?(psd: Psd): Promise<ArrayBuffer>;
  /** Triggers the download (default: Blob + anchor click). */
  download?(data: ArrayBuffer, fileName: string): void;
}

/** Sets a 2D context transform from a matrix. */
const setTransform = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

const throwIfAborted = (signal?: AbortSignal): void => {
  if (signal?.aborted) {
    throw new DOMException('PSD export was aborted.', 'AbortError');
  }
};

/**
 * Bakes one planned layer's pixels into a world-AABB-sized surface: draws the
 * layer's cache through its transform (offset into the AABB), then bakes any
 * non-destructive adjustments in place. Opacity/blend are NOT baked — they ride
 * on the PSD layer. Returns the surface + its straight-alpha `ImageData`.
 */
const bakeLayer = async (
  planLayer: PsdPlanLayer,
  deps: ExecutePsdExportDeps,
  read: (surface: RasterSurface, rect: Rect) => ImageData,
  write: (surface: RasterSurface, imageData: ImageData, x: number, y: number) => void
): Promise<{ surface: RasterSurface; imageData: ImageData }> => {
  throwIfAborted(deps.signal);
  const { worldRect } = planLayer;
  const width = worldRect.width;
  const height = worldRect.height;
  const surface = deps.backend.createSurface(width, height);
  const ctx = surface.ctx;
  setTransform(ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  ctx.clearRect(0, 0, width, height);

  const { rect, surface: cache } = await deps.getLayerSurface(planLayer.id);
  throwIfAborted(deps.signal);
  // local→world then shift into AABB-local (translation only affects e/f).
  const local = layerMatrix(planLayer.transform);
  setTransform(ctx, { ...local, e: local.e - worldRect.x, f: local.f - worldRect.y });
  // The cache holds pixels for `rect` in layer-local space; draw at that origin.
  if (rect.width > 0 && rect.height > 0) {
    ctx.drawImage(cache.canvas, rect.x, rect.y);
  }

  const fullRect: Rect = { height, width, x: 0, y: 0 };
  const imageData = read(surface, fullRect);
  if (planLayer.adjustments) {
    applyAdjustments(imageData, planLayer.adjustments);
    // Write the adjusted pixels back so the flattened composite (below) reuses
    // this surface directly.
    write(surface, imageData, 0, 0);
  }
  return { imageData, surface };
};

/**
 * Executes a PSD export plan: bakes each layer, flattens the enabled layers into
 * a merged composite (so Photoshop/Bridge show a correct preview — ag-psd does
 * NOT regenerate the composite), assembles the {@link Psd}, serializes via the
 * lazily-imported `ag-psd`, and triggers a download. No-op for a non-`ok` plan.
 */
export const executePsdExport = async (
  plan: PsdExportPlan,
  fileName: string,
  deps: ExecutePsdExportDeps
): Promise<void> => {
  if (plan.status !== 'ok') {
    return;
  }
  const read = deps.readImageData ?? defaultReadImageData;
  const write = deps.writeImageData ?? defaultWriteImageData;
  const writePsdFn = deps.writePsd ?? defaultWritePsd;
  const download = deps.download ?? defaultDownload;

  const children: AgPsdLayer[] = [];
  const baked: { planLayer: PsdPlanLayer; surface: RasterSurface }[] = [];

  for (const planLayer of plan.layers) {
    throwIfAborted(deps.signal);
    const { imageData, surface } = await bakeLayer(planLayer, deps, read, write);
    baked.push({ planLayer, surface });
    children.push({
      blendMode: planLayer.blendMode,
      bottom: planLayer.bottom,
      hidden: planLayer.hidden,
      imageData,
      left: planLayer.left,
      name: planLayer.name,
      opacity: planLayer.opacity,
      right: planLayer.right,
      top: planLayer.top,
    });
  }

  // Flatten the enabled layers (bottom-to-top = plan order) into the merged
  // composite the PSD carries as its full-document preview.
  throwIfAborted(deps.signal);
  const composite = deps.backend.createSurface(plan.width, plan.height);
  const cctx = composite.ctx;
  setTransform(cctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  cctx.clearRect(0, 0, plan.width, plan.height);
  for (const { planLayer, surface } of baked) {
    if (planLayer.hidden) {
      continue;
    }
    cctx.globalAlpha = planLayer.opacity;
    cctx.globalCompositeOperation = planLayer.compositeBlend;
    cctx.drawImage(surface.canvas, planLayer.left, planLayer.top);
  }
  cctx.globalAlpha = 1;
  cctx.globalCompositeOperation = 'source-over';

  const psd: Psd = {
    children,
    height: plan.height,
    imageData: read(composite, { height: plan.height, width: plan.width, x: 0, y: 0 }),
    width: plan.width,
  };

  throwIfAborted(deps.signal);
  const bytes = await writePsdFn(psd);
  throwIfAborted(deps.signal);
  download(bytes, fileName);
};
