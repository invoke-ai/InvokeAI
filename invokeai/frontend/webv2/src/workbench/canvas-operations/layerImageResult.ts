import type { ExportBakedLayerBlobResult, LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { RunUtilityGraphOptions, UtilityGraphResult } from '@workbench/canvas-operations/backend/utilityQueue';
import type { SamSessionErrorCode } from '@workbench/canvas-operations/operationTypes';
import type { SamInput, SamModel } from '@workbench/generation/canvas/samGraph';
import type { CanvasImageRef } from '@workbench/types';

import { UtilityQueueError } from '@workbench/canvas-operations/backend/utilityQueue';
import { buildSamGraph, documentToExportLocalSamInput, isSamInputValid } from '@workbench/generation/canvas/samGraph';

export interface SelectObjectReadyResult {
  status: 'ready';
  image: CanvasImageRef;
  rect: Rect;
  guard: LayerExportGuard;
}

export type SelectObjectRunResult =
  | SelectObjectReadyResult
  | { status: 'aborted' | 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' }
  | { status: 'invalid-input' }
  | { status: 'dimension-mismatch'; message: string }
  | { status: 'failed'; message: string; code: SamSessionErrorCode };

export interface SelectObjectRunnerDeps {
  exportSource(): Promise<ExportBakedLayerBlobResult>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ height: number; imageName: string; width: number }>;
  runGraph(options: Pick<RunUtilityGraphOptions, 'graph' | 'outputNodeId' | 'signal'>): Promise<UtilityGraphResult>;
}

const isAbortError = (error: unknown): boolean => error instanceof Error && error.name === 'AbortError';
const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export interface SelectObjectPreparedSource {
  guard: LayerExportGuard;
  imageName: string;
  height: number;
  rect: Rect;
  width: number;
}

export type PrepareSelectObjectSourceResult =
  | { status: 'ready'; source: SelectObjectPreparedSource }
  | Exclude<SelectObjectRunResult, SelectObjectReadyResult>;

export const prepareSelectObjectSource = async (
  deps: Pick<SelectObjectRunnerDeps, 'exportSource' | 'uploadIntermediate'>,
  signal?: AbortSignal,
  onPhase?: (phase: 'uploading') => void
): Promise<PrepareSelectObjectSourceResult> => {
  if (signal?.aborted) {
    return { status: 'aborted' };
  }
  let exported: ExportBakedLayerBlobResult;
  try {
    exported = await deps.exportSource();
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    if (exported.status !== 'ok') {
      return exported;
    }
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) {
      return { status: 'aborted' };
    }
    return { code: 'unknown', message: errorMessage(error), status: 'failed' };
  }

  try {
    onPhase?.('uploading');
    const uploaded = await deps.uploadIntermediate(exported.blob, signal);
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    if (!hasExactDimensions(uploaded, exported.rect)) {
      return {
        message: `Uploaded Select Object source dimensions ${String(uploaded.width)}x${String(uploaded.height)} do not match ${exported.rect.width}x${exported.rect.height}.`,
        status: 'dimension-mismatch',
      };
    }
    return {
      source: {
        guard: exported.guard,
        height: uploaded.height,
        imageName: uploaded.imageName,
        rect: exported.rect,
        width: uploaded.width,
      },
      status: 'ready',
    };
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) {
      return { status: 'aborted' };
    }
    return { code: 'upload', message: errorMessage(error), status: 'failed' };
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
  onPhase?: (phase: 'processing-sam') => void;
}

const hasExactDimensions = (
  value: { height: number; width: number },
  expected: { height: number; width: number }
): boolean =>
  Number.isFinite(value.width) &&
  Number.isInteger(value.width) &&
  value.width > 0 &&
  Number.isFinite(value.height) &&
  Number.isInteger(value.height) &&
  value.height > 0 &&
  value.width === expected.width &&
  value.height === expected.height;

export const processSelectObjectSource = async (
  options: ProcessSelectObjectSourceOptions
): Promise<SelectObjectRunResult> => {
  if (options.signal?.aborted) {
    return { status: 'aborted' };
  }
  try {
    if (!hasExactDimensions(options.source, options.source.rect)) {
      return {
        message: `Uploaded Select Object source dimensions ${String(options.source.width)}x${String(options.source.height)} do not match ${options.source.rect.width}x${options.source.rect.height}.`,
        status: 'dimension-mismatch',
      };
    }
    const input = documentToExportLocalSamInput(options.input, options.source.rect);
    if (!isSamInputValid(input)) {
      return { status: 'invalid-input' };
    }
    const built = buildSamGraph({
      applyPolygonRefinement: options.applyPolygonRefinement,
      imageName: options.source.imageName,
      input,
      invert: options.invert,
      model: options.model,
    });
    options.onPhase?.('processing-sam');
    const output = await options.runGraph({
      graph: built.graph,
      outputNodeId: built.outputNodeId,
      signal: options.signal,
    });
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    if (!hasExactDimensions(output, options.source.rect)) {
      return {
        message: `SAM output dimensions ${String(output.width)}x${String(output.height)} do not match ${options.source.rect.width}x${options.source.rect.height}.`,
        status: 'dimension-mismatch',
      };
    }
    return {
      guard: options.source.guard,
      image: {
        height: output.height,
        imageName: output.imageName,
        width: output.width,
      },
      rect: options.source.rect,
      status: 'ready',
    };
  } catch (error) {
    if (options.signal?.aborted || isAbortError(error)) {
      return { status: 'aborted' };
    }
    const code: SamSessionErrorCode =
      error instanceof UtilityQueueError
        ? error.reason === 'no-output'
          ? 'no-output'
          : error.reason === 'reconcile'
            ? 'reconcile'
            : 'queue'
        : 'queue';
    return { code, message: errorMessage(error), status: 'failed' };
  }
};
