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
    uploadIntermediate(blob: Blob): Promise<{ imageName: string }>;
    runFilterGraph(options: {
      graph: BackendGraphContract;
      outputNodeId: string;
      signal?: AbortSignal;
    }): Promise<{ imageName: string }>;
  };
}

const throwIfAborted = (signal: AbortSignal | undefined): void => {
  if (signal?.aborted) {
    throw new DOMException('The layer-filter request was aborted.', 'AbortError');
  }
};

export const runLayerFilter = async (options: RunLayerFilterOptions): Promise<LayerFilterResult> => {
  throwIfAborted(options.signal);
  const blob = await options.deps.encodeSurface(options.input.surface);
  throwIfAborted(options.signal);
  const uploaded = await options.deps.uploadIntermediate(blob);
  throwIfAborted(options.signal);
  const built = buildFilterGraph(options.filterType, uploaded.imageName, options.settings);
  const output = await options.deps.runFilterGraph({
    graph: built.graph,
    outputNodeId: built.outputNodeId,
    signal: options.signal,
  });
  throwIfAborted(options.signal);
  return {
    height: options.input.rect.height,
    imageName: output.imageName,
    origin: { x: options.input.rect.x, y: options.input.rect.y },
    width: options.input.rect.width,
  };
};
