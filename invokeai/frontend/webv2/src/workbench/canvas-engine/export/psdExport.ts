/**
 * PSD export separates pure geometry planning from raster baking, lazy `ag-psd` serialization and download.
 * Reverse every top-first tree level for ag-psd's bottom-first children; omit empty leaves/folders and size the
 * document to exported world bounds.
 *
 * Export disabled content as hidden layers/folders; flatten only effectively enabled content for the merged
 * preview. Opacity stays in [0,1]. Bake raster adjustments into pixels, but retain opacity/blend as PSD
 * properties.
 */

import type { CanvasAdjustmentsContract, CanvasBlendMode, CanvasColorLabel } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';
import type { BlendMode, Layer as AgPsdLayer, Psd } from 'ag-psd';

import { downloadBlob } from '@platform/browser/downloadBlob';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { applyAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';

/** Cap exports at the legacy PSD 30000px side limit to bound unbounded-canvas allocations. */
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
 * Maps supported blends to PSD keys; unknown values fall back to normal and populate {@link
 * PsdExportOk.unmappedBlends}.
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

/** One raster layer's export-relevant facts, at its place in the top-first tree. */
export interface PsdExportLayerInput {
  id: string;
  name: string;
  transform: PsdLayerTransform;
  /** The layer's content rect in LOCAL space (origin may be negative). */
  contentRect: Rect;
  /** 0..1. */
  opacity: number;
  blendMode: CanvasBlendMode;
  /** The layer's own flag; a disabled layer is exported with `hidden: true`, not dropped. */
  isEnabled: boolean;
  /** Organizational color label; PSD carries it natively as `layerColor`. */
  colorLabel?: CanvasColorLabel;
  /** Non-destructive adjustments to bake into the layer's pixels, if any. */
  adjustments?: CanvasAdjustmentsContract;
}

/** A raster group: a folder in the PSD, children top-first. */
export interface PsdExportGroupInput {
  type: 'group';
  id: string;
  name: string;
  /** The group's own flag; a disabled group is a hidden folder. */
  isEnabled: boolean;
  /** 0..1; PSD folders carry opacity natively. Absent means 1. */
  opacity?: number;
  /** PSD folders carry a blend mode natively. Absent means 'normal'. */
  blendMode?: CanvasBlendMode;
  /** Organizational color label; PSD carries it natively as `layerColor`. */
  colorLabel?: CanvasColorLabel;
  children: readonly PsdExportNodeInput[];
}

export type PsdExportNodeInput = PsdExportLayerInput | PsdExportGroupInput;

const isGroupInput = (input: PsdExportNodeInput): input is PsdExportGroupInput =>
  'type' in input && input.type === 'group';

/** A single planned PSD layer (already in ag-psd bottom-to-top order). */
export interface PsdPlanLayer {
  kind: 'layer';
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
  /** The layer's own visibility flag, as the PSD stores it. */
  hidden: boolean;
  /** Enabled with every folder above it enabled: the leaves the merged preview flattens. */
  contributes: boolean;
  colorLabel?: CanvasColorLabel;
  adjustments?: CanvasAdjustmentsContract;
}

/** A planned PSD folder; `children` are in ag-psd bottom-to-top order. */
export interface PsdPlanFolder {
  kind: 'folder';
  id: string;
  name: string;
  hidden: boolean;
  /** 0..1; written onto the folder, and applied when the preview isolates it. */
  opacity: number;
  blendMode: BlendMode;
  /** Canvas `globalCompositeOperation` for the flattened composite preview. */
  compositeBlend: GlobalCompositeOperation;
  colorLabel?: CanvasColorLabel;
  children: PsdPlanNode[];
}

export type PsdPlanNode = PsdPlanLayer | PsdPlanFolder;

/** A successful export plan. */
export interface PsdExportOk {
  status: 'ok';
  /** PSD canvas dimensions (= union bounds). */
  width: number;
  height: number;
  /** The union bounds in document space (origin is the PSD's (0,0)). */
  canvasRect: Rect;
  /** Every exported layer, flattened, in ag-psd order (bottom-to-top). */
  layers: PsdPlanLayer[];
  /** The folder tree the PSD carries, bottom-to-top at every level; `layers` are its leaves. */
  tree: PsdPlanNode[];
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
 * Plans world-AABB union bounds and a bottom-first PSD tree. Omit empty/degenerate leaves and empty folders;
 * return empty or too-large when applicable.
 */
export const planPsdExport = (
  inputs: readonly PsdExportNodeInput[],
  options: PlanPsdExportOptions = {}
): PsdExportPlan => {
  const maxDimension = options.maxDimension ?? PSD_MAX_DIMENSION;

  // Top-first depth-first leaves with effective enablement and world bounds; null bounds indicate empty or
  // degenerate content.
  const withBounds: { input: PsdExportLayerInput; contributes: boolean; worldRect: Rect | null }[] = [];
  const walk = (nodes: readonly PsdExportNodeInput[], enabled: boolean): void => {
    for (const node of nodes) {
      if (isGroupInput(node)) {
        walk(node.children, enabled && node.isEnabled);
        continue;
      }
      const contributes = enabled && node.isEnabled;
      if (isEmpty(node.contentRect)) {
        withBounds.push({ contributes, input: node, worldRect: null });
        continue;
      }
      const worldRect = roundOut(transformBounds(layerMatrix(node.transform), node.contentRect));
      withBounds.push({ contributes, input: node, worldRect: isEmpty(worldRect) ? null : worldRect });
    }
  };
  walk(inputs, true);

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
  // ag-psd order is bottom-to-top; leaves are top-first, so reverse. Layers
  // without content are dropped.
  const layers: PsdPlanLayer[] = [];
  const byId = new Map<string, PsdPlanLayer>();
  for (let i = withBounds.length - 1; i >= 0; i -= 1) {
    const { contributes, input, worldRect } = withBounds[i]!;
    if (!worldRect) {
      continue;
    }
    const mapped = BLEND_MODE_TO_PSD[input.blendMode];
    if (!mapped) {
      unmappedBlends.add(input.blendMode);
    }
    const left = worldRect.x - canvasRect.x;
    const top = worldRect.y - canvasRect.y;
    const layer: PsdPlanLayer = {
      adjustments: input.adjustments,
      blendMode: mapped ?? 'normal',
      bottom: top + worldRect.height,
      colorLabel: input.colorLabel,
      compositeBlend: blendToComposite(input.blendMode),
      contentRect: input.contentRect,
      contributes,
      hidden: !input.isEnabled,
      id: input.id,
      kind: 'layer',
      left,
      name: input.name,
      opacity: clamp01(input.opacity),
      right: left + worldRect.width,
      top,
      transform: input.transform,
      worldRect,
    };
    layers.push(layer);
    byId.set(input.id, layer);
  }

  // The same reversal at every level of the tree; folders keep only what survived.
  const buildTree = (nodes: readonly PsdExportNodeInput[]): PsdPlanNode[] => {
    const out: PsdPlanNode[] = [];
    for (let i = nodes.length - 1; i >= 0; i -= 1) {
      const node = nodes[i]!;
      if (isGroupInput(node)) {
        const children = buildTree(node.children);
        if (children.length > 0) {
          const blendMode = node.blendMode ?? 'normal';
          const opacity = clamp01(node.opacity ?? 1);
          // A default group is PASS-THROUGH: its members blend with what is
          // below the group. Photoshop's folder default is the same, so only a
          // group that actually composites in isolation (opacity or blend)
          // gets an isolating folder blend key.
          const isolated = opacity !== 1 || blendMode !== 'normal';
          const mapped = BLEND_MODE_TO_PSD[blendMode];
          if (!mapped) {
            unmappedBlends.add(blendMode);
          }
          out.push({
            blendMode: isolated ? (mapped ?? 'normal') : 'pass through',
            children,
            colorLabel: node.colorLabel,
            compositeBlend: blendToComposite(blendMode),
            hidden: !node.isEnabled,
            id: node.id,
            kind: 'folder',
            name: node.name,
            opacity,
          });
        }
        continue;
      }
      const layer = byId.get(node.id);
      if (layer) {
        out.push(layer);
      }
    }
    return out;
  };

  return {
    canvasRect,
    height: canvasRect.height,
    layers,
    status: 'ok',
    tree: buildTree(inputs),
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

/** Lazy-loads `ag-psd` only for serialization, keeping it out of the main bundle. */
const defaultWritePsd = async (psd: Psd): Promise<ArrayBuffer> => {
  const { writePsd } = await import('ag-psd');
  return writePsd(psd);
};

/** Triggers a browser download of the PSD bytes (Blob + anchor click). */
const defaultDownload = (data: ArrayBuffer, fileName: string): void => {
  const blob = new Blob([data], { type: 'image/vnd.adobe.photoshop' });
  downloadBlob(blob, fileName);
};

/** Injected dependencies for {@link executePsdExport}. */
export interface ExecutePsdExportDeps {
  /** Surface factory (usually the engine's `RasterBackend`). */
  backend: { createSurface(width: number, height: number): RasterSurface };
  /** Cancels before the next background allocation or side effect. */
  signal?: AbortSignal;
  /** Returns rasterized pixels and local content bounds, using live paint caches when available. */
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
 * Bakes transformed pixels and adjustments into world-AABB bounds, returning surface and straight-alpha ImageData.
 * Opacity/blend remain PSD properties.
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
 * Bakes layers, builds the enabled merged preview (ag-psd does not generate it), serializes and downloads. Non-ok
 * plans do nothing.
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

  const bakedById = new Map<string, AgPsdLayer>();
  const baked: { planLayer: PsdPlanLayer; surface: RasterSurface }[] = [];

  for (const planLayer of plan.layers) {
    throwIfAborted(deps.signal);
    const { imageData, surface } = await bakeLayer(planLayer, deps, read, write);
    baked.push({ planLayer, surface });
    bakedById.set(planLayer.id, {
      blendMode: planLayer.blendMode,
      bottom: planLayer.bottom,
      hidden: planLayer.hidden,
      imageData,
      left: planLayer.left,
      ...(planLayer.colorLabel ? { layerColor: planLayer.colorLabel } : {}),
      name: planLayer.name,
      opacity: planLayer.opacity,
      right: planLayer.right,
      top: planLayer.top,
    });
  }
  const toChildren = (nodes: readonly PsdPlanNode[]): AgPsdLayer[] =>
    nodes.map((node) =>
      node.kind === 'folder'
        ? {
            blendMode: node.blendMode,
            children: toChildren(node.children),
            hidden: node.hidden,
            ...(node.colorLabel ? { layerColor: node.colorLabel } : {}),
            name: node.name,
            opacity: node.opacity,
            opened: true,
          }
        : bakedById.get(node.id)!
    );
  const children = toChildren(plan.tree);

  // Flatten bottom-first. Isolate folders with opacity/blend so the merged preview matches readers' folder
  // compositing.
  throwIfAborted(deps.signal);
  const surfaceById = new Map(baked.map(({ planLayer, surface }) => [planLayer.id, surface]));
  const drawPreview = (ctx: Ctx, nodes: readonly PsdPlanNode[]): void => {
    for (const node of nodes) {
      if (node.hidden) {
        continue;
      }
      if (node.kind === 'folder') {
        if (node.opacity === 1 && node.compositeBlend === 'source-over') {
          drawPreview(ctx, node.children);
          continue;
        }
        const buffer = deps.backend.createSurface(plan.width, plan.height);
        setTransform(buffer.ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
        buffer.ctx.clearRect(0, 0, plan.width, plan.height);
        drawPreview(buffer.ctx, node.children);
        ctx.globalAlpha = node.opacity;
        ctx.globalCompositeOperation = node.compositeBlend;
        ctx.drawImage(buffer.canvas, 0, 0);
        continue;
      }
      const surface = surfaceById.get(node.id);
      if (!surface) {
        continue;
      }
      ctx.globalAlpha = node.opacity;
      ctx.globalCompositeOperation = node.compositeBlend;
      ctx.drawImage(surface.canvas, node.left, node.top);
    }
  };
  const composite = deps.backend.createSurface(plan.width, plan.height);
  const cctx = composite.ctx;
  setTransform(cctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  cctx.clearRect(0, 0, plan.width, plan.height);
  drawPreview(cctx, plan.tree);
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
