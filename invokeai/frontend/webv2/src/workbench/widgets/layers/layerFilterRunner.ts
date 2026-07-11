import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { BackendGraphContract } from '@workbench/types';

import { buildFilterGraph } from '@workbench/generation/canvas/filterGraphs';

export interface LayerFilterResult {
  imageName: string;
  width: number;
  height: number;
  origin: { x: number; y: number };
}

export interface RunLayerFilterOptions {
  input: { surface: RasterSurface; rect: Rect };
  filterType: string;
  settings?: Record<string, unknown>;
  signal?: AbortSignal;
  deps: {
    encodeSurface(surface: RasterSurface): Promise<Blob>;
    uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
    runFilterGraph(options: {
      graph: BackendGraphContract;
      outputNodeId: string;
      signal?: AbortSignal;
    }): Promise<{ height: number; imageName: string; width: number }>;
  };
}

export const resolveFilterOutputRect = (options: {
  filterType: string;
  output: { width: number; height: number };
  settings?: Record<string, unknown>;
  source: Rect;
}): Rect => {
  if (options.filterType === 'img_blur') {
    const configuredRadius = options.settings?.radius;
    const radius =
      typeof configuredRadius === 'number' && Number.isFinite(configuredRadius)
        ? Math.min(4_096, Math.max(0, configuredRadius))
        : 8;
    const padding = Math.max(0, Math.ceil(radius * (options.settings?.blur_type === 'box' ? 1 : 3)));
    return {
      height: options.output.height,
      width: options.output.width,
      x: options.source.x - padding,
      y: options.source.y - padding,
    };
  }
  if (options.filterType === 'spandrel_filter') {
    return { ...options.output, x: options.source.x, y: options.source.y };
  }
  return { ...options.source };
};

const throwIfAborted = (signal: AbortSignal | undefined): void => {
  if (signal?.aborted) {
    throw new DOMException('The layer-filter request was aborted.', 'AbortError');
  }
};

export const runLayerFilter = async (options: RunLayerFilterOptions): Promise<LayerFilterResult> => {
  throwIfAborted(options.signal);
  const blob = await options.deps.encodeSurface(options.input.surface);
  throwIfAborted(options.signal);
  const uploaded = await options.deps.uploadIntermediate(blob, options.signal);
  throwIfAborted(options.signal);
  const built = buildFilterGraph(options.filterType, uploaded.imageName, options.settings);
  const output = await options.deps.runFilterGraph({
    graph: built.graph,
    outputNodeId: built.outputNodeId,
    signal: options.signal,
  });
  throwIfAborted(options.signal);
  const rect = resolveFilterOutputRect({
    filterType: options.filterType,
    output,
    settings: options.settings,
    source: options.input.rect,
  });
  return {
    height: rect.height,
    imageName: output.imageName,
    origin: { x: rect.x, y: rect.y },
    width: rect.width,
  };
};
