import type { RunUtilityGraphOptions, UtilityGraphResult } from '@workbench/canvas-engine/backend/utilityQueue';
import type {
  CommitMaskImageResult,
  CommitMaskImageResultOptions,
  ExportBakedLayerBlobResult,
  LayerExportGuard,
  ReplaceSelectionFromImageResult,
} from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { SamModel } from '@workbench/generation/canvas/samGraph';
import type { CanvasImageRef } from '@workbench/types';

import { buildSamGraph } from '@workbench/generation/canvas/samGraph';

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
  prompt: string;
  model: SamModel;
  applyPolygonRefinement: boolean;
  signal?: AbortSignal;
  deps: SelectObjectRunnerDeps;
}

const isAbortError = (error: unknown): boolean => error instanceof Error && error.name === 'AbortError';
const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const runSelectObject = async (options: RunSelectObjectOptions): Promise<SelectObjectRunResult> => {
  if (options.signal?.aborted) {
    return { status: 'aborted' };
  }
  try {
    const exported = await options.deps.exportLayer(options.layerId);
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    if (exported.status !== 'ok') {
      return exported;
    }

    const uploaded = await options.deps.uploadIntermediate(exported.blob, options.signal);
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    const built = buildSamGraph({
      applyPolygonRefinement: options.applyPolygonRefinement,
      imageName: uploaded.imageName,
      model: options.model,
      prompt: options.prompt,
    });
    const output = await options.deps.runGraph({
      graph: built.graph,
      outputNodeId: built.outputNodeId,
      signal: options.signal,
    });
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    return {
      guard: exported.guard,
      image: { height: exported.rect.height, imageName: output.imageName, width: exported.rect.width },
      rect: exported.rect,
      status: 'ready',
    };
  } catch (error) {
    if (options.signal?.aborted || isAbortError(error)) {
      return { status: 'aborted' };
    }
    return { message: errorMessage(error), status: 'failed' };
  }
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
