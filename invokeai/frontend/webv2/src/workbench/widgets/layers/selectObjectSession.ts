import type { RunUtilityGraphOptions, UtilityGraphResult } from '@workbench/canvas-engine/backend/utilityQueue';
import type {
  CanvasOperationController,
  CanvasOperationRunResult,
  CanvasOperationSession,
} from '@workbench/canvas-engine/canvasOperationController';
import type { ExportBakedLayerBlobResult, LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { SamInput, SamModel } from '@workbench/generation/canvas/samGraph';

import { isSamInputValid } from '@workbench/generation/canvas/samGraph';

import type { SelectObjectPreparedSource, SelectObjectReadyResult } from './layerImageResult';

import { prepareSelectObjectSource, processSelectObjectSource } from './layerImageResult';

export type SelectObjectSessionStatus = 'ready' | 'scheduled' | 'processing' | 'error';

export interface SelectObjectSessionPreview<T> {
  data: T;
  guard: LayerExportGuard;
  image: SelectObjectReadyResult['image'];
  inputHash: string;
  isolated: boolean;
  rect: Rect;
  sourceImageName: string;
}

export interface SelectObjectSessionState<T> {
  input: SamInput;
  model: SamModel;
  invert: boolean;
  applyPolygonRefinement: boolean;
  autoProcess: boolean;
  isolatedPreview: boolean;
  status: SelectObjectSessionStatus;
  error: string | null;
  preview: SelectObjectSessionPreview<T> | null;
  sourceGuard: LayerExportGuard | null;
}

export interface SelectObjectSessionDeps<T> {
  controller: CanvasOperationController;
  exportLayer(layerId: string): Promise<ExportBakedLayerBlobResult>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
  runGraph(options: Pick<RunUtilityGraphOptions, 'graph' | 'outputNodeId' | 'signal'>): Promise<UtilityGraphResult>;
  decodePreview(result: SelectObjectReadyResult, signal: AbortSignal): Promise<T>;
  publishPreview(preview: SelectObjectSessionPreview<T>): undefined;
  cleanupPreview(): void;
  isGuardCurrent(guard: LayerExportGuard): boolean;
}

export interface CreateSelectObjectSessionOptions<T> {
  projectId: string;
  layerId: string;
  deps: SelectObjectSessionDeps<T>;
}

export type SelectObjectSessionProcessResult = CanvasOperationRunResult | 'deduped' | 'invalid';

export interface SelectObjectSession<T> {
  getSnapshot(): SelectObjectSessionState<T>;
  subscribe(listener: () => void): () => void;
  update(
    changes: Partial<
      Pick<
        SelectObjectSessionState<T>,
        'applyPolygonRefinement' | 'autoProcess' | 'input' | 'invert' | 'isolatedPreview' | 'model'
      >
    >
  ): void;
  process(): Promise<SelectObjectSessionProcessResult>;
  reset(): void;
  cancel(): void;
}

const defaultState = <T>(): SelectObjectSessionState<T> => ({
  applyPolygonRefinement: false,
  autoProcess: false,
  error: null,
  input: { prompt: '', type: 'prompt' },
  invert: false,
  isolatedPreview: true,
  model: 'segment-anything-2-large',
  preview: null,
  sourceGuard: null,
  status: 'ready',
});

const stableInputHash = <T>(state: SelectObjectSessionState<T>): string => {
  const input =
    state.input.type === 'prompt'
      ? ['prompt', state.input.prompt.trim()]
      : [
          'visual',
          state.input.includePoints.map(({ x, y }) => [x, y]),
          state.input.excludePoints.map(({ x, y }) => [x, y]),
          state.input.bbox
            ? [state.input.bbox.x, state.input.bbox.y, state.input.bbox.width, state.input.bbox.height]
            : null,
        ];
  return JSON.stringify([input, state.model, state.invert, state.applyPolygonRefinement, state.isolatedPreview]);
};

const resultError = (result: Exclude<Awaited<ReturnType<typeof prepareSelectObjectSource>>, { status: 'ready' }>) =>
  result.status === 'failed' ? result.message : `Select Object source is ${result.status}.`;

export const createSelectObjectSession = <T>(options: CreateSelectObjectSessionOptions<T>): SelectObjectSession<T> => {
  const { deps, layerId, projectId } = options;
  let state = defaultState<T>();
  let source: SelectObjectPreparedSource | null = null;
  let operation: CanvasOperationSession | null = null;
  let operationGuard: LayerExportGuard | null = null;
  let sourceController: AbortController | null = null;
  let requestToken = 0;
  let lastPublishedHash: string | null = null;
  let pendingHash: string | null = null;
  let pendingProcess: Promise<SelectObjectSessionProcessResult> | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let startingOperation = false;
  const listeners = new Set<() => void>();

  const publishState = (next: SelectObjectSessionState<T>): void => {
    state = next;
    for (const listener of listeners) {
      listener();
    }
  };

  const clearTimer = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const clearSource = (): void => {
    source = null;
    operation = null;
    operationGuard = null;
    lastPublishedHash = null;
    publishState({ ...state, preview: null, sourceGuard: null, status: 'ready' });
  };

  deps.controller.subscribe(() => {
    if (!startingOperation && operation && deps.controller.getSnapshot().status === 'idle') {
      sourceController?.abort();
      sourceController = null;
      clearSource();
    }
  });

  const cleanupPreview = (): void => {
    deps.cleanupPreview();
    if (state.preview) {
      publishState({ ...state, preview: null });
    }
  };

  const schedule = (): void => {
    clearTimer();
    if (!state.autoProcess || !isSamInputValid(state.input)) {
      if (state.status === 'scheduled') {
        publishState({ ...state, status: 'ready' });
      }
      return;
    }
    publishState({ ...state, error: null, status: 'scheduled' });
    timer = setTimeout(() => {
      timer = null;
      void process();
    }, 1_000);
  };

  const ensureSource = async (signal: AbortSignal): Promise<SelectObjectPreparedSource> => {
    if (source && deps.isGuardCurrent(source.guard)) {
      return source;
    }
    source = null;
    const prepared = await prepareSelectObjectSource(layerId, deps, signal);
    if (prepared.status !== 'ready') {
      throw new Error(resultError(prepared));
    }
    if (signal.aborted || !deps.isGuardCurrent(prepared.source.guard)) {
      throw new DOMException('Select Object source became stale.', 'AbortError');
    }
    source = prepared.source;
    return source;
  };

  const runRequest = async (
    token: number,
    hash: string,
    requestState: SelectObjectSessionState<T>,
    controller: AbortController
  ): Promise<SelectObjectSessionProcessResult> => {
    try {
      const preparedSource = await ensureSource(controller.signal);
      if (controller.signal.aborted || token !== requestToken) {
        return 'stale';
      }

      if (operationGuard !== preparedSource.guard || !operation) {
        startingOperation = true;
        const nextOperation = deps.controller.start({
          cleanupPreview,
          guard: preparedSource.guard,
          identity: { kind: 'select-object', layerId, projectId },
        });
        startingOperation = false;
        if (!nextOperation) {
          clearSource();
          return 'stale';
        }
        operation = nextOperation;
        operationGuard = preparedSource.guard;
      }

      publishState({ ...state, sourceGuard: preparedSource.guard, status: 'processing' });
      const result = await operation.run(
        async (signal) => {
          const processed = await processSelectObjectSource({
            applyPolygonRefinement: requestState.applyPolygonRefinement,
            input: requestState.input,
            invert: requestState.invert,
            model: requestState.model,
            runGraph: deps.runGraph,
            signal,
            source: preparedSource,
          });
          if (processed.status !== 'ready') {
            if (processed.status === 'aborted') {
              throw new DOMException('Select Object processing was aborted.', 'AbortError');
            }
            throw new Error(
              processed.status === 'failed' ? processed.message : `Select Object is ${processed.status}.`
            );
          }
          const data = await deps.decodePreview(processed, signal);
          if (signal.aborted) {
            throw new DOMException('Select Object preview decode was aborted.', 'AbortError');
          }
          return {
            data,
            guard: processed.guard,
            image: processed.image,
            inputHash: hash,
            isolated: requestState.isolatedPreview,
            rect: processed.rect,
            sourceImageName: preparedSource.imageName,
          } satisfies SelectObjectSessionPreview<T>;
        },
        (preview) => {
          deps.publishPreview(preview);
          lastPublishedHash = hash;
          publishState({ ...state, error: null, preview, sourceGuard: preview.guard, status: 'ready' });
          return undefined;
        }
      );
      if (token !== requestToken) {
        return 'stale';
      }
      if (result === 'error') {
        const controllerState = deps.controller.getSnapshot();
        publishState({
          ...state,
          error: controllerState.status === 'active' ? controllerState.error : 'Select Object processing failed.',
          status: 'error',
        });
      } else if (result === 'stale' && !deps.isGuardCurrent(preparedSource.guard)) {
        clearSource();
      }
      return result;
    } catch (cause) {
      if (
        controller.signal.aborted ||
        token !== requestToken ||
        (cause instanceof Error && cause.name === 'AbortError')
      ) {
        return 'stale';
      }
      publishState({ ...state, error: cause instanceof Error ? cause.message : String(cause), status: 'error' });
      return 'error';
    }
  };

  const process = (): Promise<SelectObjectSessionProcessResult> => {
    clearTimer();
    if (!isSamInputValid(state.input)) {
      publishState({ ...state, error: 'A Segment Anything input is required.', status: 'error' });
      return Promise.resolve('invalid');
    }
    const hash = stableInputHash(state);
    if (hash === pendingHash && pendingProcess) {
      return pendingProcess;
    }
    if (hash === lastPublishedHash) {
      return Promise.resolve('deduped');
    }

    requestToken += 1;
    const token = requestToken;
    sourceController?.abort();
    const controller = new AbortController();
    sourceController = controller;
    pendingHash = hash;
    publishState({ ...state, error: null, status: 'processing' });
    const requestState = state;
    const promise = runRequest(token, hash, requestState, controller).finally(() => {
      if (token === requestToken) {
        sourceController = null;
        pendingHash = null;
        pendingProcess = null;
      }
    });
    pendingProcess = promise;
    return promise;
  };

  const cancelCurrent = (): void => {
    clearTimer();
    requestToken += 1;
    sourceController?.abort();
    sourceController = null;
    pendingHash = null;
    pendingProcess = null;
    operation?.cancel();
    operation = null;
    operationGuard = null;
    source = null;
    lastPublishedHash = null;
  };

  return {
    cancel: () => {
      cancelCurrent();
      publishState({ ...state, error: null, preview: null, sourceGuard: null, status: 'ready' });
    },
    getSnapshot: () => state,
    process,
    reset: () => {
      cancelCurrent();
      publishState(defaultState<T>());
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    update: (changes) => {
      publishState({ ...state, ...changes, error: null });
      schedule();
    },
  };
};
