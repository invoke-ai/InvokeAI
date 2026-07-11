import type { RunUtilityGraphOptions, UtilityGraphResult } from '@workbench/canvas-engine/backend/utilityQueue';
import type {
  CommitMaskImageResult,
  CommitMaskImageResultOptions,
  ExportBakedLayerBlobResult,
  LayerExportGuard,
  ReplaceSelectionFromImageResult,
} from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { SamInput, SamModel } from '@workbench/generation/canvas/samGraph';
import type { CanvasImageRef } from '@workbench/types';

import { buildSamGraph, documentToExportLocalSamInput } from '@workbench/generation/canvas/samGraph';

export type SelectObjectTarget = 'selection' | 'inpaint_mask' | 'regional_guidance';

export interface SelectObjectOptions {
  applyPolygonRefinement: boolean;
  model: SamModel;
  prompt: string;
  target: SelectObjectTarget;
}

export const createDefaultSelectObjectOptions = (): SelectObjectOptions => ({
  applyPolygonRefinement: false,
  model: 'segment-anything-2-large',
  prompt: '',
  target: 'selection',
});

export const isSelectObjectPromptValid = (prompt: string): boolean => prompt.trim().length > 0;

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
  runGraph(options: Pick<RunUtilityGraphOptions, 'graph' | 'outputNodeId' | 'signal'>): Promise<UtilityGraphResult>;
}

export interface RunSelectObjectOptions {
  layerId: string;
  input?: SamInput;
  /** Prompt-only compatibility seam for the current dialog. */
  prompt?: string;
  model: SamModel;
  invert?: boolean;
  applyPolygonRefinement: boolean;
  signal?: AbortSignal;
  deps: SelectObjectRunnerDeps;
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

export const runSelectObject = async (options: RunSelectObjectOptions): Promise<SelectObjectRunResult> => {
  const prepared = await prepareSelectObjectSource(options.layerId, options.deps, options.signal);
  if (prepared.status !== 'ready') {
    return prepared;
  }
  return processSelectObjectSource({
    applyPolygonRefinement: options.applyPolygonRefinement,
    input: options.input ?? { prompt: options.prompt ?? '', type: 'prompt' },
    invert: options.invert ?? false,
    model: options.model,
    runGraph: options.deps.runGraph,
    signal: options.signal,
    source: prepared.source,
  });
};

export type SelectObjectRouteResult = ReplaceSelectionFromImageResult | CommitMaskImageResult;

export interface SelectObjectRouteDeps {
  replaceSelectionFromImage(
    guard: LayerExportGuard,
    image: CanvasImageRef,
    rect: Rect,
    signal?: AbortSignal
  ): Promise<ReplaceSelectionFromImageResult>;
  makeImageDurable(imageName: string): Promise<void>;
  commitMaskImageResult(options: CommitMaskImageResultOptions): Promise<CommitMaskImageResult>;
}

export interface RouteSelectObjectResultOptions {
  result: SelectObjectReadyResult;
  target: SelectObjectTarget;
  signal: AbortSignal;
  isCurrent(): boolean;
  deps: SelectObjectRouteDeps;
}

const canceledRoute = (signal: AbortSignal, isCurrent: () => boolean): { status: 'aborted' | 'stale' } | null => {
  if (signal.aborted) {
    return { status: 'aborted' };
  }
  if (!isCurrent()) {
    return { status: 'stale' };
  }
  return null;
};

export const routeSelectObjectResult = async (
  options: RouteSelectObjectResultOptions
): Promise<SelectObjectRouteResult> => {
  const canceled = canceledRoute(options.signal, options.isCurrent);
  if (canceled) {
    return canceled;
  }
  try {
    if (options.target === 'selection') {
      const result = await options.deps.replaceSelectionFromImage(
        options.result.guard,
        options.result.image,
        options.result.rect,
        options.signal
      );
      // Once the engine reports an authoritative mutation result, preserve it.
      // A late dialog-session invalidation may suppress UI updates, but must not
      // relabel an already-applied selection as canceled or stale.
      return result;
    }

    await options.deps.makeImageDurable(options.result.image.imageName);
    const afterDurability = canceledRoute(options.signal, options.isCurrent);
    if (afterDurability) {
      return afterDurability;
    }
    const beforeCommit = canceledRoute(options.signal, options.isCurrent);
    if (beforeCommit) {
      return beforeCommit;
    }
    const result = await options.deps.commitMaskImageResult({
      guard: options.result.guard,
      image: options.result.image,
      rect: options.result.rect,
      signal: options.signal,
      target: options.target,
    });
    // The engine owns the final abort/guard check immediately before mutation.
    // Do not overwrite a committed result if the UI session invalidates after
    // that authoritative mutation has completed.
    return result;
  } catch (error) {
    if (options.signal.aborted || isAbortError(error)) {
      return { status: 'aborted' };
    }
    return { message: errorMessage(error), status: 'failed' };
  }
};
