import type { LayerExportGuard, SelectObjectLifecycleGuard } from './engine';

export type CanvasOperationIdentity =
  | { kind: 'select-object'; projectId: string }
  | { kind: 'filter'; projectId: string; layerId: string };

export type CanvasOperationGuard = LayerExportGuard | SelectObjectLifecycleGuard;

export type CanvasOperationState =
  | { status: 'idle' }
  | {
      status: 'active';
      identity: CanvasOperationIdentity;
      phase: 'ready' | 'running' | 'error';
      error: string | null;
    };

export type CanvasOperationRunResult = 'published' | 'stale' | 'error';

export interface CanvasOperationSession {
  /**
   * Prepares a preview asynchronously, then commits it synchronously only while
   * this request and its export guard are still current. `commitPreview` must
   * explicitly return `undefined`; do not cast or erase an async callback to
   * `() => void`, which TypeScript cannot distinguish from a synchronous one.
   */
  run<T>(
    work: (signal: AbortSignal) => Promise<T>,
    commitPreview: (result: T) => undefined
  ): Promise<CanvasOperationRunResult>;
  reset(): void;
  /** Aborts only the active request and keeps this operation ready for retry or Cancel. */
  interruptProcessing(): void;
  cancel(): void;
}

export type StartCanvasOperationOptions =
  | {
      identity: Extract<CanvasOperationIdentity, { kind: 'select-object' }>;
      guard: SelectObjectLifecycleGuard;
      cleanupPreview(): void;
    }
  | {
      identity: Extract<CanvasOperationIdentity, { kind: 'filter' }>;
      guard: LayerExportGuard;
      cleanupPreview(): void;
    };

export interface CanvasOperationController {
  getSnapshot(): CanvasOperationState;
  subscribe(listener: () => void): () => void;
  start(options: StartCanvasOperationOptions): CanvasOperationSession | null;
  reset(): void;
  cancel(): void;
  invalidateSource(projectId: string, layerId: string): void;
  invalidateLayer(projectId: string, layerId: string): void;
  invalidateComposite(projectId: string): void;
  invalidateProject(projectId: string): void;
  invalidateDocument(projectId: string): void;
  dispose(): void;
}

export interface CanvasOperationControllerDeps {
  isGuardCurrent(guard: CanvasOperationGuard): boolean;
}

type ActiveOperation = StartCanvasOperationOptions & {
  requestController: AbortController | null;
  requestToken: number;
};

const errorMessage = (cause: unknown): string => (cause instanceof Error ? cause.message : String(cause));

export const createCanvasOperationController = (deps: CanvasOperationControllerDeps): CanvasOperationController => {
  let state: CanvasOperationState = { status: 'idle' };
  let active: ActiveOperation | null = null;
  let disposed = false;
  const listeners = new Set<() => void>();

  const publishState = (next: CanvasOperationState): void => {
    state = next;
    if (disposed) {
      return;
    }
    for (const listener of listeners) {
      try {
        listener();
      } catch {
        // One faulty subscriber must not strand the controller mid-transition.
      }
    }
  };

  const cleanupPreview = (operation: ActiveOperation): void => {
    try {
      operation.cleanupPreview();
    } catch {
      // Cleanup is best-effort and must not prevent cancellation or invalidation.
    }
  };

  const close = (operation: ActiveOperation): void => {
    if (active !== operation) {
      return;
    }
    operation.requestToken += 1;
    const requestController = operation.requestController;
    operation.requestController = null;
    active = null;
    const idleState: CanvasOperationState = { status: 'idle' };
    state = idleState;
    requestController?.abort();
    cleanupPreview(operation);
    if (active === null && state === idleState) {
      publishState(idleState);
    }
  };

  const interruptProcessing = (operation: ActiveOperation): void => {
    if (active !== operation || disposed) {
      return;
    }
    operation.requestToken += 1;
    const requestController = operation.requestController;
    operation.requestController = null;
    active = null;
    const idleState: CanvasOperationState = { status: 'idle' };
    state = idleState;
    requestController?.abort();
    cleanupPreview(operation);
    if (!disposed && active === null && state === idleState) {
      active = operation;
      publishState({ error: null, identity: operation.identity, phase: 'ready', status: 'active' });
    }
  };

  const isGuardCurrent = (operation: ActiveOperation): boolean => {
    try {
      return deps.isGuardCurrent(operation.guard);
    } catch {
      return false;
    }
  };

  const isCurrentRequest = (operation: ActiveOperation, token: number, controller: AbortController): boolean =>
    !disposed &&
    active === operation &&
    operation.requestToken === token &&
    operation.requestController === controller &&
    !controller.signal.aborted;

  const run = async <T>(
    operation: ActiveOperation,
    work: (signal: AbortSignal) => Promise<T>,
    commitPreview: (result: T) => undefined
  ): Promise<CanvasOperationRunResult> => {
    if (disposed || active !== operation) {
      return 'stale';
    }
    if (!isGuardCurrent(operation)) {
      close(operation);
      return 'stale';
    }

    operation.requestToken += 1;
    const token = operation.requestToken;
    const previousController = operation.requestController;
    operation.requestController = null;
    active = null;
    const idleState: CanvasOperationState = { status: 'idle' };
    state = idleState;
    previousController?.abort();
    cleanupPreview(operation);
    if (disposed || active !== null || state !== idleState) {
      return 'stale';
    }

    const controller = new AbortController();
    operation.requestController = controller;
    active = operation;
    publishState({ error: null, identity: operation.identity, phase: 'running', status: 'active' });
    if (!isCurrentRequest(operation, token, controller)) {
      return 'stale';
    }

    try {
      const result = await work(controller.signal);
      if (!isCurrentRequest(operation, token, controller)) {
        return 'stale';
      }
      if (!isGuardCurrent(operation)) {
        close(operation);
        return 'stale';
      }

      commitPreview(result);
      if (!isCurrentRequest(operation, token, controller)) {
        return 'stale';
      }
      if (!isGuardCurrent(operation)) {
        close(operation);
        return 'stale';
      }

      operation.requestController = null;
      publishState({ error: null, identity: operation.identity, phase: 'ready', status: 'active' });
      return 'published';
    } catch (cause) {
      if (!isCurrentRequest(operation, token, controller)) {
        return 'stale';
      }
      operation.requestController = null;
      publishState({ error: errorMessage(cause), identity: operation.identity, phase: 'error', status: 'active' });
      return 'error';
    }
  };

  const start = (options: StartCanvasOperationOptions): CanvasOperationSession | null => {
    const layerIdentityMatches =
      options.identity.kind === 'filter' &&
      'layerId' in options.guard &&
      options.identity.layerId === options.guard.layerId;
    const compositeIdentityMatches =
      options.identity.kind === 'select-object' &&
      'kind' in options.guard &&
      options.guard.kind === 'select-object-lifecycle';
    if (
      disposed ||
      options.identity.projectId !== options.guard.projectId ||
      (!layerIdentityMatches && !compositeIdentityMatches)
    ) {
      return null;
    }
    try {
      if (!deps.isGuardCurrent(options.guard)) {
        return null;
      }
    } catch {
      return null;
    }

    if (active) {
      close(active);
      if (active) {
        return null;
      }
    }
    const operation: ActiveOperation = {
      ...options,
      requestController: null,
      requestToken: 0,
    };
    active = operation;
    publishState({ error: null, identity: operation.identity, phase: 'ready', status: 'active' });

    return {
      cancel: () => close(operation),
      interruptProcessing: () => interruptProcessing(operation),
      reset: () => interruptProcessing(operation),
      run: <T>(work: (signal: AbortSignal) => Promise<T>, commitPreview: (result: T) => undefined) =>
        run(operation, work, commitPreview),
    };
  };

  const invalidateTarget = (projectId: string, layerId?: string): void => {
    if (
      active?.identity.projectId === projectId &&
      (layerId === undefined || (active.identity.kind === 'select-object' ? true : active.identity.layerId === layerId))
    ) {
      close(active);
    }
  };

  return {
    cancel: () => {
      if (active) {
        close(active);
      }
    },
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      if (active) {
        close(active);
      } else {
        state = { status: 'idle' };
      }
      listeners.clear();
    },
    getSnapshot: () => state,
    invalidateDocument: (projectId) => invalidateTarget(projectId),
    invalidateComposite: (projectId) => {
      if (active?.identity.projectId === projectId && active.identity.kind === 'select-object') {
        close(active);
      }
    },
    invalidateLayer: (projectId, layerId) => invalidateTarget(projectId, layerId),
    invalidateProject: (projectId) => invalidateTarget(projectId),
    invalidateSource: (projectId, layerId) => {
      if (
        active?.identity.kind === 'filter' &&
        active.identity.projectId === projectId &&
        active.identity.layerId === layerId
      ) {
        close(active);
      }
    },
    reset: () => {
      if (active) {
        interruptProcessing(active);
      }
    },
    start,
    subscribe: (listener) => {
      if (!disposed) {
        listeners.add(listener);
      }
      return () => listeners.delete(listener);
    },
  };
};
