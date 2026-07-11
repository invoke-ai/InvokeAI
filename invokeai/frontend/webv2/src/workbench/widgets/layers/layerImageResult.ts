import type { RunUtilityGraphOptions, UtilityGraphResult } from '@workbench/canvas-engine/backend/utilityQueue';
import type { ExportBakedLayerBlobResult, LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { SamInput, SamModel } from '@workbench/generation/canvas/samGraph';
import type { CanvasImageRef } from '@workbench/types';

import { buildSamGraph, documentToExportLocalSamInput } from '@workbench/generation/canvas/samGraph';

export interface SelectObjectReadyResult {
  status: 'ready';
  image: CanvasImageRef;
  rect: Rect;
  guard: LayerExportGuard;
}

export type SelectObjectRunResult =
  | SelectObjectReadyResult
  | { status: 'aborted' | 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' }
  | { status: 'failed'; message: string };

export interface SelectObjectRunnerDeps {
  exportLayer(layerId: string): Promise<ExportBakedLayerBlobResult>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
  runGraph(
    options: Pick<RunUtilityGraphOptions, 'graph' | 'outputNodeId' | 'signal'>
  ): Promise<Pick<UtilityGraphResult, 'imageName' | 'origin'>>;
}

const isAbortError = (error: unknown): boolean => error instanceof Error && error.name === 'AbortError';
const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export interface SelectObjectPreparedSource {
  guard: LayerExportGuard;
  imageName: string;
  rect: Rect;
}

export type PrepareSelectObjectSourceResult =
  | { status: 'ready'; source: SelectObjectPreparedSource }
  | Exclude<SelectObjectRunResult, SelectObjectReadyResult>;

export const prepareSelectObjectSource = async (
  layerId: string,
  deps: Pick<SelectObjectRunnerDeps, 'exportLayer' | 'uploadIntermediate'>,
  signal?: AbortSignal
): Promise<PrepareSelectObjectSourceResult> => {
  if (signal?.aborted) {
    return { status: 'aborted' };
  }
  try {
    const exported = await deps.exportLayer(layerId);
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    if (exported.status !== 'ok') {
      return exported;
    }
    const uploaded = await deps.uploadIntermediate(exported.blob, signal);
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    return {
      source: { guard: exported.guard, imageName: uploaded.imageName, rect: exported.rect },
      status: 'ready',
    };
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) {
      return { status: 'aborted' };
    }
    return { message: errorMessage(error), status: 'failed' };
  }
};

export interface ProcessSelectObjectSourceOptions {
  source: SelectObjectPreparedSource;
  input: SamInput;
  model: SamModel;
  invert: boolean;
  applyPolygonRefinement: boolean;
  signal?: AbortSignal;
  runGraph: SelectObjectRunnerDeps['runGraph'];
}

export const processSelectObjectSource = async (
  options: ProcessSelectObjectSourceOptions
): Promise<SelectObjectRunResult> => {
  if (options.signal?.aborted) {
    return { status: 'aborted' };
  }
  try {
    const built = buildSamGraph({
      applyPolygonRefinement: options.applyPolygonRefinement,
      imageName: options.source.imageName,
      input: documentToExportLocalSamInput(options.input, options.source.rect),
      invert: options.invert,
      model: options.model,
    });
    const output = await options.runGraph({
      graph: built.graph,
      outputNodeId: built.outputNodeId,
      signal: options.signal,
    });
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    return {
      guard: options.source.guard,
      image: {
        height: options.source.rect.height,
        imageName: output.imageName,
        width: options.source.rect.width,
      },
      rect: options.source.rect,
      status: 'ready',
    };
  } catch (error) {
    if (options.signal?.aborted || isAbortError(error)) {
      return { status: 'aborted' };
    }
    return { message: errorMessage(error), status: 'failed' };
  }
};
