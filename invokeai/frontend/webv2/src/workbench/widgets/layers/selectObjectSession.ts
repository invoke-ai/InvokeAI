import type { RunUtilityGraphOptions, UtilityGraphResult } from '@workbench/canvas-engine/backend/utilityQueue';
import type {
  CanvasOperationController,
  CanvasOperationRunResult,
  CanvasOperationSession,
} from '@workbench/canvas-engine/canvasOperationController';
import type { CanvasCompositeExportGuard, ExportCanvasCompositeBlobResult } from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { SamInput, SamModel } from '@workbench/generation/canvas/samGraph';

import { isSamDocumentInputValid } from '@workbench/generation/canvas/samGraph';

import type { SelectObjectPreparedSource, SelectObjectReadyResult } from './layerImageResult';

import { prepareSelectObjectSource, processSelectObjectSource } from './layerImageResult';

export type SelectObjectSessionStatus = 'ready' | 'scheduled' | 'processing' | 'error';

export interface SelectObjectSessionPreview<T> {
  data: T;
  guard: CanvasCompositeExportGuard;
  image: SelectObjectReadyResult['image'];
  inputHash: string;
  previewId: number;
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
  sourceGuard: CanvasCompositeExportGuard | null;
}

export interface SelectObjectSessionDeps<T> {
  captureGuard(): CanvasCompositeExportGuard | null;
  controller: CanvasOperationController;
  exportComposite(): Promise<ExportCanvasCompositeBlobResult>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
  runGraph(
    options: Pick<RunUtilityGraphOptions, 'graph' | 'outputNodeId' | 'signal'>
  ): Promise<Pick<UtilityGraphResult, 'imageName' | 'origin'>>;
  decodePreview(result: SelectObjectReadyResult, signal: AbortSignal): Promise<T>;
  publishPreview(preview: SelectObjectSessionPreview<T>): undefined;
  cleanupPreview(): void;
  isGuardCurrent(guard: CanvasCompositeExportGuard): boolean;
}

export interface CreateSelectObjectSessionOptions<T> {
  projectId: string;
  deps: SelectObjectSessionDeps<T>;
}

export type SelectObjectSessionProcessResult = CanvasOperationRunResult | 'blocked' | 'deduped' | 'invalid';

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
  interruptProcessing(): void;
  reportError(message: string): void;
  reset(): void;
  cancel(): void;
  dispose(): void;
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
  return JSON.stringify([input, state.model, state.invert, state.applyPolygonRefinement]);
};

const resultError = (result: Exclude<Awaited<ReturnType<typeof prepareSelectObjectSource>>, { status: 'ready' }>) =>
  result.status === 'failed' ? result.message : `Select Object source is ${result.status}.`;

export const createSelectObjectSession = <T>(options: CreateSelectObjectSessionOptions<T>): SelectObjectSession<T> => {
  const { deps, projectId } = options;
  let state = defaultState<T>();
  let source: SelectObjectPreparedSource | null = null;
  let operation: CanvasOperationSession | null = null;
  let operationGuard: CanvasCompositeExportGuard | null = null;
  let requestToken = 0;
  let lastPublishedHash: string | null = null;
  let pendingHash: string | null = null;
  let pendingProcess: Promise<SelectObjectSessionProcessResult> | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let startingOperation = false;
  let disposed = false;
  let unsubscribeController: (() => void) | null = null;
  const listeners = new Set<() => void>();

  const publishState = (next: SelectObjectSessionState<T>): void => {
    if (next.status === 'error') {
      state = next.error === null ? { ...next, status: 'ready' } : next;
    } else {
      state = next.error === null ? next : { ...next, error: null };
    }
    for (const listener of listeners) {
      try {
        listener();
      } catch {
        // One faulty subscriber must not interrupt session state transitions.
      }
    }
  };

  const clearTimer = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const clearSource = (): void => {
    requestToken += 1;
    pendingHash = null;
    pendingProcess = null;
    source = null;
    operation = null;
    operationGuard = null;
    lastPublishedHash = null;
    publishState({ ...state, preview: null, sourceGuard: null, status: 'ready' });
  };

  const isGuardCurrent = (guard: CanvasCompositeExportGuard): boolean => {
    try {
      return deps.isGuardCurrent(guard);
    } catch {
      return false;
    }
  };

  const isSameGuard = (left: CanvasCompositeExportGuard, right: CanvasCompositeExportGuard): boolean =>
    left.projectId === right.projectId &&
    left.documentFingerprint === right.documentFingerprint &&
    left.documentGeneration === right.documentGeneration;

  unsubscribeController = deps.controller.subscribe(() => {
    if (disposed) {
      return;
    }
    if (!startingOperation && operation && deps.controller.getSnapshot().status === 'idle') {
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
    if (disposed) {
      return;
    }
    if (!state.autoProcess || !isSamDocumentInputValid(state.input)) {
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

  const ensureSource = async (
    guard: CanvasCompositeExportGuard,
    signal: AbortSignal
  ): Promise<SelectObjectPreparedSource> => {
    if (source?.guard === guard && isGuardCurrent(guard)) {
      return source;
    }
    source = null;
    const prepared = await prepareSelectObjectSource(deps, signal);
    if (prepared.status !== 'ready') {
      throw new Error(resultError(prepared));
    }
    if (signal.aborted || !isGuardCurrent(guard) || !isSameGuard(prepared.source.guard, guard)) {
      throw new DOMException('Select Object source became stale.', 'AbortError');
    }
    source = { ...prepared.source, guard };
    return source;
  };

  const runRequest = async (
    token: number,
    hash: string,
    requestState: SelectObjectSessionState<T>,
    guard: CanvasCompositeExportGuard
  ): Promise<SelectObjectSessionProcessResult> => {
    try {
      if (operationGuard !== guard || !operation) {
        startingOperation = true;
        let nextOperation: CanvasOperationSession | null = null;
        try {
          nextOperation = deps.controller.start({
            cleanupPreview,
            guard,
            identity: { kind: 'select-object', projectId },
          });
        } finally {
          startingOperation = false;
        }
        if (!nextOperation) {
          clearSource();
          return 'stale';
        }
        operation = nextOperation;
        operationGuard = guard;
      }

      publishState({ ...state, sourceGuard: guard, status: 'processing' });
      const result = await operation.run(
        async (signal) => {
          const preparedSource = await ensureSource(guard, signal);
          if (signal.aborted || token !== requestToken) {
            throw new DOMException('Select Object source preparation was aborted.', 'AbortError');
          }
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
            previewId: token,
            isolated: requestState.isolatedPreview,
            rect: processed.rect,
            sourceImageName: preparedSource.imageName,
          } satisfies SelectObjectSessionPreview<T>;
        },
        (preview) => {
          const publishedPreview =
            preview.isolated === state.isolatedPreview ? preview : { ...preview, isolated: state.isolatedPreview };
          deps.publishPreview(publishedPreview);
          lastPublishedHash = hash;
          publishState({
            ...state,
            error: null,
            preview: publishedPreview,
            sourceGuard: preview.guard,
            status: 'ready',
          });
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
      } else if (result === 'stale' && !isGuardCurrent(guard)) {
        clearSource();
      }
      return result;
    } catch (cause) {
      if (token !== requestToken || (cause instanceof Error && cause.name === 'AbortError')) {
        return 'stale';
      }
      publishState({ ...state, error: cause instanceof Error ? cause.message : String(cause), status: 'error' });
      return 'error';
    }
  };

  const process = (): Promise<SelectObjectSessionProcessResult> => {
    clearTimer();
    if (disposed) {
      return Promise.resolve('stale');
    }
    if (!isSamDocumentInputValid(state.input)) {
      publishState({ ...state, error: 'A Segment Anything input is required.', status: 'error' });
      return Promise.resolve('invalid');
    }
    if (state.sourceGuard && !isGuardCurrent(state.sourceGuard)) {
      requestToken += 1;
      operation?.cancel();
      if (!operation && state.preview) {
        cleanupPreview();
      }
      source = null;
      operation = null;
      operationGuard = null;
      pendingHash = null;
      pendingProcess = null;
      lastPublishedHash = null;
      publishState({ ...state, preview: null, sourceGuard: null, status: 'ready' });
    }
    const hash = stableInputHash(state);
    if (hash === pendingHash && pendingProcess) {
      publishState({ ...state, error: null, status: 'processing' });
      return pendingProcess;
    }
    if (
      hash === lastPublishedHash &&
      state.preview !== null &&
      state.sourceGuard === state.preview.guard &&
      source?.guard === state.preview.guard &&
      operationGuard === state.preview.guard &&
      isGuardCurrent(state.preview.guard)
    ) {
      publishState({ ...state, error: null, status: 'ready' });
      return Promise.resolve('deduped');
    }

    requestToken += 1;
    const token = requestToken;
    const guard = operationGuard && isGuardCurrent(operationGuard) ? operationGuard : deps.captureGuard();
    if (!guard || !isGuardCurrent(guard)) {
      publishState({ ...state, error: 'Select Object source is not ready.', status: 'error' });
      return Promise.resolve('error');
    }
    pendingHash = hash;
    publishState({ ...state, error: null, status: 'processing' });
    const requestState = state;
    const promise = runRequest(token, hash, requestState, guard).finally(() => {
      if (token === requestToken) {
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
    pendingHash = null;
    pendingProcess = null;
    operation?.cancel();
    operation = null;
    operationGuard = null;
    source = null;
    lastPublishedHash = null;
  };

  const invalidateProcessingState = (): void => {
    requestToken += 1;
    pendingHash = null;
    pendingProcess = null;
    lastPublishedHash = null;
    operation?.reset();
    if (!operation && state.preview) {
      cleanupPreview();
    }
  };

  return {
    cancel: () => {
      if (disposed) {
        return;
      }
      cancelCurrent();
      publishState({ ...state, error: null, preview: null, sourceGuard: null, status: 'ready' });
    },
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      clearTimer();
      requestToken += 1;
      pendingHash = null;
      pendingProcess = null;
      lastPublishedHash = null;
      unsubscribeController?.();
      unsubscribeController = null;
      operation?.cancel();
      if (!operation && state.preview) {
        cleanupPreview();
      }
      operation = null;
      operationGuard = null;
      source = null;
      publishState({ ...state, error: null, preview: null, sourceGuard: null, status: 'ready' });
      listeners.clear();
    },
    getSnapshot: () => state,
    interruptProcessing: () => {
      clearTimer();
      if (disposed || (state.status !== 'processing' && state.status !== 'scheduled')) {
        return;
      }
      requestToken += 1;
      pendingHash = null;
      pendingProcess = null;
      lastPublishedHash = null;
      operation?.interruptProcessing();
      publishState({ ...state, error: null, preview: null, status: 'ready' });
    },
    process,
    reportError: (message) => {
      if (!disposed) {
        publishState({ ...state, error: message, status: 'error' });
      }
    },
    reset: () => {
      if (disposed) {
        return;
      }
      cancelCurrent();
      publishState(defaultState<T>());
    },
    subscribe: (listener) => {
      if (disposed) {
        return () => undefined;
      }
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    update: (changes) => {
      if (disposed) {
        return;
      }
      const processingChanged =
        ('input' in changes && changes.input !== state.input) ||
        ('model' in changes && changes.model !== state.model) ||
        ('invert' in changes && changes.invert !== state.invert) ||
        ('applyPolygonRefinement' in changes && changes.applyPolygonRefinement !== state.applyPolygonRefinement);
      const isolationChanged = 'isolatedPreview' in changes && changes.isolatedPreview !== state.isolatedPreview;
      if (processingChanged) {
        invalidateProcessingState();
      }
      const preview =
        isolationChanged && state.preview
          ? { ...state.preview, isolated: changes.isolatedPreview ?? state.isolatedPreview }
          : state.preview;
      if (preview && preview !== state.preview) {
        deps.publishPreview(preview);
      }
      publishState({
        ...state,
        ...changes,
        error: null,
        preview: processingChanged ? null : preview,
        status: processingChanged ? 'ready' : state.status,
      });
      if (processingChanged || 'autoProcess' in changes) {
        schedule();
      }
    },
  };
};
