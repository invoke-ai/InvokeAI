import type { RunUtilityGraphOptions, UtilityGraphResult } from '@features/queue/utility';
import type { ExportBakedLayerBlobResult, LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type {
  CanvasOperationController,
  CanvasOperationRunResult,
  CanvasOperationSession,
} from '@workbench/canvas-operations/operationController';
import type { SamSessionError, SamSessionErrorCode } from '@workbench/canvas-operations/operationTypes';
import type { SamInput, SamModel } from '@workbench/canvas-operations/samGraph';

import { isSamDocumentInputValid } from '@workbench/canvas-operations/samGraph';

import type { SelectObjectPreparedSource, SelectObjectReadyResult } from './layerImageResult';

import { prepareSelectObjectSource, processSelectObjectSource } from './layerImageResult';

export type SelectObjectSessionStatus =
  | 'ready'
  | 'scheduled'
  | 'preparing-source'
  | 'uploading'
  | 'processing-sam'
  | 'rendering-preview'
  | 'error';

export interface SelectObjectSessionPreview<T> {
  data: T;
  guard: LayerExportGuard;
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
  error: SamSessionError | null;
  /** A `process()` asked for while a run's error showed is still in flight (see `SamSessionSnapshot.retrying`). */
  retrying: boolean;
  preview: SelectObjectSessionPreview<T> | null;
  sourceGuard: LayerExportGuard | null;
}

const SAM_ERROR_FROM_RUN: Record<SamSessionErrorCode, boolean> = {
  'adjustment-error': true,
  'adjustment-over-budget': true,
  'adjustment-too-large': true,
  'commit-failed': false,
  decode: true,
  empty: true,
  invalid: true,
  locked: false,
  'no-output': true,
  'not-ready': true,
  'output-dimension': true,
  queue: true,
  reconcile: true,
  unknown: true,
  unreadable: true,
  upload: true,
};

/**
 * Whether a run (source preparation, upload, SAM, preview decode) reported the error, so running again is its retry;
 * applying or saving the preview reports `commit-failed` or `locked`, which Apply and Save as retry instead.
 */
export const isSamRunError = (code: SamSessionErrorCode): boolean => SAM_ERROR_FROM_RUN[code];

export interface SelectObjectSessionDeps<T> {
  captureGuard(): LayerExportGuard | null;
  controller: CanvasOperationController;
  exportSource(): Promise<ExportBakedLayerBlobResult>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ height: number; imageName: string; width: number }>;
  runGraph(options: Pick<RunUtilityGraphOptions, 'graph' | 'outputNodeId' | 'signal'>): Promise<UtilityGraphResult>;
  decodePreview(result: SelectObjectReadyResult, signal: AbortSignal): Promise<T>;
  publishPreview(preview: SelectObjectSessionPreview<T>): undefined;
  cleanupPreview(): void;
  isGuardCurrent(guard: LayerExportGuard): boolean;
}

export interface CreateSelectObjectSessionOptions<T> {
  deps: SelectObjectSessionDeps<T>;
  /** Auto-update is an account preference; the session starts from it and Reset keeps it. */
  initialAutoProcess: boolean;
  operation: CanvasOperationSession;
}

export type SelectObjectSessionProcessResult = CanvasOperationRunResult | 'deduped' | 'invalid';

export type SelectObjectSessionChanges<T> = Partial<
  Pick<
    SelectObjectSessionState<T>,
    'applyPolygonRefinement' | 'autoProcess' | 'input' | 'invert' | 'isolatedPreview' | 'model'
  >
> & {
  /**
   * Switches between visual and prompt input: the current input is kept aside and the requested type's last input
   * (or an empty one) comes back. An explicit `input` in the same update wins; one of the other type is a switch too.
   */
  inputType?: SamInput['type'];
};

export interface SelectObjectSession<T> {
  getSnapshot(): SelectObjectSessionState<T>;
  subscribe(listener: () => void): () => void;
  update(changes: SelectObjectSessionChanges<T>): void;
  process(): Promise<SelectObjectSessionProcessResult>;
  reportError(error: SamSessionError | string): void;
  reset(): void;
  cancel(): void;
  dispose(): void;
}

type StashedSamInputs = { [K in SamInput['type']]?: Extract<SamInput, { type: K }> };

const EMPTY_INPUTS: Required<StashedSamInputs> = {
  prompt: { prompt: '', type: 'prompt' },
  visual: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
};

const defaultState = <T>(autoProcess: boolean): SelectObjectSessionState<T> => ({
  applyPolygonRefinement: false,
  autoProcess,
  error: null,
  input: EMPTY_INPUTS.visual,
  invert: false,
  isolatedPreview: true,
  model: 'segment-anything-2-large',
  preview: null,
  retrying: false,
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

const errorDetail = (cause: unknown): string => (cause instanceof Error ? cause.message : String(cause));

const withDetail = (code: SamSessionErrorCode, detail?: string): SamSessionError =>
  detail ? { code, detail } : { code };

const getPreparedSourceError = (
  result: Exclude<Awaited<ReturnType<typeof prepareSelectObjectSource>>, { status: 'ready' }>
): SamSessionError => {
  if (result.status === 'failed') {
    return withDetail(result.code, result.message);
  }
  if (result.status === 'empty') {
    return { code: 'empty' };
  }
  if (result.status === 'dimension-mismatch') {
    return withDetail('output-dimension', result.message);
  }
  if (result.status === 'adjustment-refused') {
    return { code: `adjustment-${result.reason}` };
  }
  if (result.status === 'unreadable') {
    return { code: 'unreadable' };
  }
  return { code: 'not-ready' };
};

const getProcessedSourceError = (
  result: Exclude<Awaited<ReturnType<typeof processSelectObjectSource>>, { status: 'ready' | 'aborted' }>
): SamSessionError => {
  if (result.status === 'invalid-input') {
    return { code: 'invalid' };
  }
  if (result.status === 'dimension-mismatch') {
    return withDetail('output-dimension', result.message);
  }
  if (result.status === 'failed') {
    return withDetail(result.code, result.message);
  }
  return { code: 'unknown' };
};

class SamSessionFailure extends Error {
  readonly error: SamSessionError;

  constructor(error: SamSessionError) {
    super(error.detail ?? error.code);
    this.name = 'SamSessionFailure';
    this.error = error;
  }
}

export const createSelectObjectSession = <T>(options: CreateSelectObjectSessionOptions<T>): SelectObjectSession<T> => {
  const { deps, operation } = options;
  let state = defaultState<T>(options.initialAutoProcess);
  // The input of the type not in use, so switching back restores it; not part of the snapshot.
  let stashedInputs: StashedSamInputs = {};
  let source: SelectObjectPreparedSource | null = null;
  let requestToken = 0;
  let lastPublishedHash: string | null = null;
  let pendingHash: string | null = null;
  let pendingPhase: SelectObjectSessionStatus | null = null;
  let pendingProcess: Promise<SelectObjectSessionProcessResult> | null = null;
  /** The request a retry started; it stays `retrying` until that request settles or is superseded. */
  let retryToken: number | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let disposed = false;
  let unsubscribeController: (() => void) | null = null;
  const listeners = new Set<() => void>();

  const publishState = (next: SelectObjectSessionState<T>): void => {
    const retrying = retryToken === requestToken && pendingHash !== null;
    if (next.status === 'error') {
      state = next.error === null ? { ...next, retrying, status: 'ready' } : { ...next, retrying };
    } else {
      state = { ...next, error: null, retrying };
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

  const publishPhase = (token: number, status: SelectObjectSessionStatus): void => {
    if (!disposed && token === requestToken) {
      pendingPhase = status;
      publishState({ ...state, error: null, status });
    }
  };

  const clearSource = (): void => {
    requestToken += 1;
    pendingHash = null;
    pendingPhase = null;
    pendingProcess = null;
    source = null;
    lastPublishedHash = null;
    publishState({ ...state, preview: null, sourceGuard: null, status: 'ready' });
  };

  const isGuardCurrent = (guard: LayerExportGuard): boolean => {
    try {
      return deps.isGuardCurrent(guard);
    } catch {
      return false;
    }
  };

  const isSameGuard = (left: LayerExportGuard, right: LayerExportGuard): boolean =>
    left.projectId === right.projectId &&
    left.layerId === right.layerId &&
    left.layer === right.layer &&
    left.cacheVersion === right.cacheVersion &&
    left.documentGeneration === right.documentGeneration;

  unsubscribeController = deps.controller.subscribe(() => {
    if (!disposed && deps.controller.getSnapshot().status === 'idle') {
      clearSource();
    }
  });

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
    guard: LayerExportGuard,
    signal: AbortSignal,
    token: number
  ): Promise<SelectObjectPreparedSource> => {
    if (source && isSameGuard(source.guard, guard) && isGuardCurrent(source.guard) && isGuardCurrent(guard)) {
      source = { ...source, guard };
      return source;
    }
    source = null;
    const prepared = await prepareSelectObjectSource(deps, signal, (phase) => publishPhase(token, phase));
    if (prepared.status !== 'ready') {
      throw new SamSessionFailure(getPreparedSourceError(prepared));
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
    guard: LayerExportGuard
  ): Promise<SelectObjectSessionProcessResult> => {
    let requestError: SamSessionError | null = null;
    try {
      pendingPhase = 'preparing-source';
      publishState({ ...state, sourceGuard: guard, status: 'preparing-source' });
      const result = await operation.run(
        async (signal) => {
          try {
            const preparedSource = await ensureSource(guard, signal, token);
            if (signal.aborted || token !== requestToken) {
              throw new DOMException('Select Object source preparation was aborted.', 'AbortError');
            }
            const processed = await processSelectObjectSource({
              applyPolygonRefinement: requestState.applyPolygonRefinement,
              input: requestState.input,
              invert: requestState.invert,
              model: requestState.model,
              runGraph: deps.runGraph,
              onPhase: (phase) => publishPhase(token, phase),
              signal,
              source: preparedSource,
            });
            if (processed.status !== 'ready') {
              if (processed.status === 'aborted') {
                throw new DOMException('Select Object processing was aborted.', 'AbortError');
              }
              throw new SamSessionFailure(getProcessedSourceError(processed));
            }
            if (!isGuardCurrent(guard) || !isSameGuard(processed.guard, guard)) {
              throw new DOMException('Select Object layer source became stale.', 'AbortError');
            }
            publishPhase(token, 'rendering-preview');
            let data: T;
            try {
              data = await deps.decodePreview(processed, signal);
            } catch (cause) {
              if (signal.aborted || (cause instanceof Error && cause.name === 'AbortError')) {
                throw cause;
              }
              const code =
                cause &&
                typeof cause === 'object' &&
                'samErrorCode' in cause &&
                cause.samErrorCode === 'output-dimension'
                  ? 'output-dimension'
                  : 'decode';
              throw new SamSessionFailure(withDetail(code, errorDetail(cause)));
            }
            if (signal.aborted || !isGuardCurrent(guard)) {
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
          } catch (cause) {
            if (cause instanceof SamSessionFailure) {
              requestError = cause.error;
            }
            throw cause;
          }
        },
        (preview) => {
          if (!isGuardCurrent(preview.guard)) {
            throw new DOMException('Select Object preview became stale.', 'AbortError');
          }
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
      if (result === 'error' && requestError === null && !isGuardCurrent(guard)) {
        clearSource();
        return 'stale';
      }
      if (result === 'error') {
        publishState({
          ...state,
          error: requestError ?? withDetail('unknown', 'Select Object processing failed.'),
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
      publishState({
        ...state,
        error: cause instanceof SamSessionFailure ? cause.error : withDetail('unknown', errorDetail(cause)),
        status: 'error',
      });
      return 'error';
    }
  };

  const process = (): Promise<SelectObjectSessionProcessResult> => {
    clearTimer();
    if (disposed) {
      return Promise.resolve('stale');
    }
    if (!isSamDocumentInputValid(state.input)) {
      publishState({ ...state, error: { code: 'invalid' }, status: 'error' });
      return Promise.resolve('invalid');
    }
    if (state.sourceGuard && !isGuardCurrent(state.sourceGuard)) {
      requestToken += 1;
      operation.interruptProcessing();
      source = null;
      pendingHash = null;
      pendingProcess = null;
      lastPublishedHash = null;
      publishState({ ...state, preview: null, sourceGuard: null, status: 'ready' });
    }
    const retry = state.error !== null && isSamRunError(state.error.code);
    const hash = stableInputHash(state);
    if (hash === pendingHash && pendingProcess) {
      publishState({ ...state, error: null, status: pendingPhase ?? 'preparing-source' });
      return pendingProcess;
    }
    if (
      hash === lastPublishedHash &&
      state.preview !== null &&
      state.sourceGuard === state.preview.guard &&
      source?.guard === state.preview.guard &&
      isGuardCurrent(state.preview.guard)
    ) {
      publishState({ ...state, error: null, status: 'ready' });
      return Promise.resolve('deduped');
    }

    requestToken += 1;
    const token = requestToken;
    const guard = deps.captureGuard();
    if (!guard || !isGuardCurrent(guard)) {
      publishState({ ...state, error: { code: 'not-ready' }, status: 'error' });
      return Promise.resolve('error');
    }
    pendingHash = hash;
    retryToken = retry ? token : null;
    const requestState = state;
    const promise = runRequest(token, hash, requestState, guard).finally(() => {
      if (token === requestToken) {
        pendingHash = null;
        pendingPhase = null;
        pendingProcess = null;
        if (state.retrying) {
          publishState(state);
        }
      }
    });
    pendingProcess = promise;
    return promise;
  };

  const cancelCurrent = (): void => {
    clearTimer();
    requestToken += 1;
    pendingHash = null;
    pendingPhase = null;
    pendingProcess = null;
    operation.cancel();
    source = null;
    lastPublishedHash = null;
  };

  const invalidateProcessingState = (): void => {
    requestToken += 1;
    pendingHash = null;
    pendingPhase = null;
    pendingProcess = null;
    lastPublishedHash = null;
    operation.reset();
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
      pendingPhase = null;
      pendingProcess = null;
      lastPublishedHash = null;
      unsubscribeController?.();
      unsubscribeController = null;
      operation.cancel();
      source = null;
      publishState({ ...state, error: null, preview: null, sourceGuard: null, status: 'ready' });
      listeners.clear();
    },
    getSnapshot: () => state,
    process,
    reportError: (error) => {
      if (!disposed) {
        publishState({
          ...state,
          error: typeof error === 'string' ? withDetail('unknown', error) : error,
          status: 'error',
        });
      }
    },
    reset: () => {
      if (disposed) {
        return;
      }
      clearTimer();
      requestToken += 1;
      pendingHash = null;
      pendingPhase = null;
      pendingProcess = null;
      operation.reset();
      source = null;
      lastPublishedHash = null;
      stashedInputs = {};
      publishState(defaultState<T>(state.autoProcess));
    },
    subscribe: (listener) => {
      if (disposed) {
        return () => undefined;
      }
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    update: ({ inputType, ...requested }) => {
      if (disposed) {
        return;
      }
      const changes: Omit<SelectObjectSessionChanges<T>, 'inputType'> = inputType &&
      inputType !== state.input.type &&
      !('input' in requested)
        ? { ...requested, input: stashedInputs[inputType] ?? EMPTY_INPUTS[inputType] }
        : requested;
      // Any switch of input type keeps the outgoing input, an explicit cross-type input included.
      if (changes.input && changes.input.type !== state.input.type) {
        stashedInputs = { ...stashedInputs, [state.input.type]: state.input };
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
