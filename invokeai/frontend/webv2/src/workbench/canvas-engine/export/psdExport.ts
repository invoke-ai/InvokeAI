/** Trees are top-first, ag-psd children bottom-first, so every level reverses; disabled content exports hidden.
 Adjustments bake into pixels; opacity and blend stay PSD properties. */

import type { CanvasAdjustmentsContract, CanvasBlendMode, CanvasColorLabel } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';
import type { BlendMode, Layer as AgPsdLayer, Psd } from 'ag-psd';

import { downloadBlob } from '@platform/browser/downloadBlob';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';
import {
  adjustExactly,
  AdjustmentRefusedError,
  imageDataOf,
  type SpatialRunner,
} from '@workbench/canvas-engine/render/exactAdjustments';

/** The legacy PSD side limit, which also bounds unbounded-canvas allocations. */
export const PSD_MAX_DIMENSION = 30000;

/** Duplicated to keep this module contract-light. */
export interface PsdLayerTransform {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
}

/** Unknown values fall back to normal and populate {@link PsdExportOk.unmappedBlends}. */
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

export const blendModeToPsd = (mode: CanvasBlendMode): BlendMode => BLEND_MODE_TO_PSD[mode] ?? 'normal';

export interface PsdExportLayerInput {
  id: string;
  name: string;
  transform: PsdLayerTransform;
  /** Layer-local; origin may be negative. */
  contentRect: Rect;
  opacity: number;
  blendMode: CanvasBlendMode;
  /** A disabled layer is exported with `hidden: true`, not dropped. */
  isEnabled: boolean;
  /** PSD carries it natively as `layerColor`. */
  colorLabel?: CanvasColorLabel;
  /** Layer-local reference space. */
  adjustments?: CanvasAdjustmentsContract;
  /** Innermost first; PSD cannot represent them, so they bake onto this layer alone, approximating the composite. */
  groupStacks?: readonly PsdGroupStack[];
}

export interface PsdGroupStack {
  readonly id: string;
  readonly adjustments: CanvasAdjustmentsContract;
}

export interface PsdExportGroupInput {
  type: 'group';
  id: string;
  name: string;
  isEnabled: boolean;
  /** Absent means 1. */
  opacity?: number;
  /** Absent means 'normal'. */
  blendMode?: CanvasBlendMode;
  /** PSD carries it natively as `layerColor`. */
  colorLabel?: CanvasColorLabel;
  children: readonly PsdExportNodeInput[];
}

export type PsdExportNodeInput = PsdExportLayerInput | PsdExportGroupInput;

const isGroupInput = (input: PsdExportNodeInput): input is PsdExportGroupInput =>
  'type' in input && input.type === 'group';

/** Already in ag-psd bottom-to-top order. */
export interface PsdPlanLayer {
  kind: 'layer';
  id: string;
  name: string;
  /** Relative to the union origin. */
  left: number;
  top: number;
  right: number;
  bottom: number;
  worldRect: Rect;
  transform: PsdLayerTransform;
  contentRect: Rect;
  opacity: number;
  blendMode: BlendMode;
  /** For the flattened composite preview. */
  compositeBlend: GlobalCompositeOperation;
  hidden: boolean;
  /** Enabled with every folder above it enabled: the leaves the merged preview flattens. */
  contributes: boolean;
  colorLabel?: CanvasColorLabel;
  adjustments?: CanvasAdjustmentsContract;
  groupStacks?: readonly PsdGroupStack[];
}

/** `children` are bottom-to-top. */
export interface PsdPlanFolder {
  kind: 'folder';
  id: string;
  name: string;
  hidden: boolean;
  opacity: number;
  blendMode: BlendMode;
  compositeBlend: GlobalCompositeOperation;
  colorLabel?: CanvasColorLabel;
  children: PsdPlanNode[];
}

export type PsdPlanNode = PsdPlanLayer | PsdPlanFolder;

export interface PsdExportOk {
  status: 'ok';
  width: number;
  height: number;
  /** Its origin is the PSD's (0,0). */
  canvasRect: Rect;
  /** Bottom-to-top. */
  layers: PsdPlanLayer[];
  /** Bottom-to-top at every level; `layers` are its leaves. */
  tree: PsdPlanNode[];
  unmappedBlends: string[];
}

export type PsdExportPlan = PsdExportOk | { status: 'empty' } | { status: 'too-large'; width: number; height: number };

export interface PlanPsdExportOptions {
  maxDimension?: number;
}

const layerMatrix = (t: PsdLayerTransform): Mat2d => fromTRS({ x: t.x, y: t.y }, t.rotation, t.scaleX, t.scaleY);

const clamp01 = (value: number): number => (value < 0 ? 0 : value > 1 ? 1 : value);

/** Omits empty or degenerate leaves and empty folders. */
export const planPsdExport = (
  inputs: readonly PsdExportNodeInput[],
  options: PlanPsdExportOptions = {}
): PsdExportPlan => {
  const maxDimension = options.maxDimension ?? PSD_MAX_DIMENSION;

  // Null bounds mark empty or degenerate content.
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
      groupStacks: input.groupStacks,
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

  const buildTree = (nodes: readonly PsdExportNodeInput[]): PsdPlanNode[] => {
    const out: PsdPlanNode[] = [];
    for (let i = nodes.length - 1; i >= 0; i -= 1) {
      const node = nodes[i]!;
      if (isGroupInput(node)) {
        const children = buildTree(node.children);
        if (children.length > 0) {
          const blendMode = node.blendMode ?? 'normal';
          const opacity = clamp01(node.opacity ?? 1);
          // A default group is pass-through, like Photoshop's folder default; only an isolating group gets a blend key.
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

  const tree = buildTree(inputs);
  const countRecords = (nodes: readonly PsdPlanNode[]): number =>
    nodes.reduce((total, node) => total + (node.kind === 'folder' ? 2 + countRecords(node.children) : 1), 0);
  if (countRecords(tree) > 32767) {
    return { status: 'too-large', width: canvasRect.width, height: canvasRect.height };
  }
  return {
    canvasRect,
    height: canvasRect.height,
    layers,
    status: 'ok',
    tree,
    unmappedBlends: [...unmappedBlends],
    width: canvasRect.width,
  };
};

/** Only executor-owned pixels and PSD metadata, never engine resources. */
export interface PsdSerializationRequest {
  id: number;
  psd: Psd;
}
export type PsdSerializationResponse = { id: number; bytes: ArrayBuffer } | { id: number; error: string };

/** One worker per operation: termination interrupts the synchronous writer and discards its heap. */
export const serializePsd = (psd: Psd, signal?: AbortSignal): Promise<ArrayBuffer> => {
  if (signal?.aborted) {
    return Promise.reject(new DOMException('PSD export was aborted.', 'AbortError'));
  }
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./psdSerialization.worker.ts', import.meta.url), { type: 'module' });
    const cleanup = (): void => {
      signal?.removeEventListener('abort', abort);
      worker.onmessage = null;
      worker.onerror = null;
      worker.onmessageerror = null;
      worker.terminate();
    };
    const fail = (error: Error): void => {
      cleanup();
      reject(error);
    };
    const abort = (): void => fail(new DOMException('PSD export was aborted.', 'AbortError'));
    // The worker is private to this call, so id 1 cannot collide.
    worker.onmessage = ({ data }: MessageEvent<PsdSerializationResponse>) => {
      if (data.id !== 1) {
        return;
      }
      cleanup();
      if ('error' in data) {
        reject(new Error(data.error));
      } else {
        resolve(data.bytes);
      }
    };
    worker.onerror = (event) => {
      event.preventDefault();
      fail(new Error(event.message || 'PSD serialization worker failed.'));
    };
    worker.onmessageerror = () => fail(new Error('PSD serialization response could not be read.'));
    signal?.addEventListener('abort', abort, { once: true });
    try {
      const buffers: Transferable[] = [];
      const toWire = (node: AgPsdLayer): AgPsdLayer => {
        const { imageData, children, ...metadata } = node;
        // ImageData is cloned even with its buffer transferred; transfer plain PixelData instead.
        if (imageData) {
          buffers.push(imageData.data.buffer);
        }
        return {
          ...metadata,
          ...(imageData
            ? { imageData: { width: imageData.width, height: imageData.height, data: imageData.data } }
            : {}),
          ...(children ? { children: children.map(toWire) } : {}),
        };
      };
      const wire = toWire(psd) as Psd;
      worker.postMessage({ id: 1, psd: wire } satisfies PsdSerializationRequest, buffers);
    } catch (error) {
      fail(error instanceof Error ? error : new Error(String(error)));
    }
  });
};

type Ctx = RasterSurface['ctx'];

const defaultReadImageData = (surface: RasterSurface, rect: Rect): ImageData =>
  surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height);

const defaultWriteImageData = (surface: RasterSurface, imageData: ImageData, x: number, y: number): void =>
  surface.ctx.putImageData(imageData, x, y);

const defaultDownload = async (data: ArrayBuffer, fileName: string): Promise<void> => {
  await downloadBlob(new Blob([data], { type: 'image/vnd.adobe.photoshop' }), fileName);
};

export interface ExecutePsdExportDeps {
  backend: { createSurface(width: number, height: number): RasterSurface };
  signal?: AbortSignal;
  onStage?(stage: 'serializing' | 'downloading'): void;
  /** Without it a spatial stack throws {@link AdjustmentRefusedError} rather than bake without its spatial entries. */
  runSpatial?: SpatialRunner;
  /** Uses live paint caches when available. */
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect }>;
  readImageData?(surface: RasterSurface, rect: Rect): ImageData;
  writeImageData?(surface: RasterSurface, imageData: ImageData, x: number, y: number): void;
  /** Default: an operation-owned worker. */
  writePsd?(psd: Psd): Promise<ArrayBuffer>;
  download?(data: ArrayBuffer, fileName: string): void | Promise<void>;
}

const setTransform = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

const throwIfAborted = (signal?: AbortSignal): void => {
  if (signal?.aborted) {
    throw new DOMException('PSD export was aborted.', 'AbortError');
  }
};

const evaluate = async (
  pixels: ImageData,
  adjustments: CanvasAdjustmentsContract,
  origin: Rect,
  ownerId: string,
  deps: ExecutePsdExportDeps
): Promise<ImageData> => {
  const result = await adjustExactly(pixels, adjustments, origin, { run: deps.runSpatial, signal: deps.signal });
  if (result.status === 'aborted') {
    throw new DOMException('PSD export was aborted.', 'AbortError');
  }
  if (result.status !== 'ok') {
    throw new AdjustmentRefusedError(ownerId, result.status);
  }
  return result.data === pixels.data ? pixels : imageDataOf(result.data, pixels.width, pixels.height);
};

/** The layer's own stack adjusts layer-local pixels before resampling, as the display does; group stacks after. */
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
  let source = cache;
  if (planLayer.adjustments && rect.width > 0 && rect.height > 0) {
    source = deps.backend.createSurface(rect.width, rect.height);
    setTransform(source.ctx, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
    source.ctx.drawImage(cache.canvas, 0, 0);
    const localPixels = read(source, { height: rect.height, width: rect.width, x: 0, y: 0 });
    write(source, await evaluate(localPixels, planLayer.adjustments, rect, planLayer.id, deps), 0, 0);
  }
  const local = layerMatrix(planLayer.transform);
  setTransform(ctx, { ...local, e: local.e - worldRect.x, f: local.f - worldRect.y });
  if (rect.width > 0 && rect.height > 0) {
    ctx.drawImage(source.canvas, rect.x, rect.y);
  }

  if (source !== cache) {
    source.resize(0, 0);
  }

  const fullRect: Rect = { height, width, x: 0, y: 0 };
  let imageData = read(surface, fullRect);
  if (planLayer.groupStacks && planLayer.groupStacks.length > 0) {
    for (const stack of planLayer.groupStacks) {
      imageData = await evaluate(imageData, stack.adjustments, worldRect, stack.id, deps);
    }
    // Write back so the flattened composite reuses this surface.
    write(surface, imageData, 0, 0);
  }
  return { imageData, surface };
};

/** ag-psd does not generate the merged preview. */
export const executePsdExport = async (
  plan: PsdExportPlan,
  fileName: string,
  deps: ExecutePsdExportDeps
): Promise<void> => {
  if (plan.status !== 'ok') {
    return;
  }
  const surfaces: RasterSurface[] = [];
  const backend = deps.backend;
  deps = {
    ...deps,
    backend: {
      createSurface: (width, height) => {
        const surface = backend.createSurface(width, height);
        surfaces.push(surface);
        return surface;
      },
    },
  };
  try {
    const read = deps.readImageData ?? defaultReadImageData;
    const write = deps.writeImageData ?? defaultWriteImageData;
    const writePsdFn = deps.writePsd ?? ((psd: Psd) => serializePsd(psd, deps.signal));
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

    // Isolate folders with opacity or blend so the preview matches readers' folder compositing.
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

    for (const surface of surfaces) {
      surface.resize(0, 0);
    }
    surfaces.length = 0;
    deps.onStage?.('serializing');
    throwIfAborted(deps.signal);
    const bytes = await writePsdFn(psd);
    throwIfAborted(deps.signal);
    // Once the browser accepts the click, cancellation cannot retract the download.
    deps.onStage?.('downloading');
    await download(bytes, fileName);
  } finally {
    for (const surface of surfaces) {
      surface.resize(0, 0);
    }
  }
};
