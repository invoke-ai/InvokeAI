import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type { CanvasEditGate, CanvasEditLease } from '@workbench/canvas-engine/editGate';

import { createCanvasEditGate } from '@workbench/canvas-engine/editGate';

export type CanvasOperationIdentity =
  | { kind: 'select-object'; projectId: string; layerId: string }
  | { kind: 'filter'; projectId: string; layerId: string };

export type CanvasOperationGuard = LayerExportGuard;

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

export interface StartCanvasOperationOptions {
  identity: CanvasOperationIdentity;
  guard: LayerExportGuard;
  cleanupPreview(): void;
}

export interface CanvasOperationController {
  getSnapshot(): CanvasOperationState;
  subscribe(listener: () => void): () => void;
  start(options: StartCanvasOperationOptions): CanvasOperationSession | null;
  reset(): void;
  cancel(): void;
  invalidateSource(projectId: string, layerId: string): void;
  invalidateLayer(projectId: string, layerId: string): void;
  invalidateProject(projectId: string): void;
  invalidateDocument(projectId: string): void;
  dispose(): void;
}

export interface CanvasOperationControllerDeps {
  isGuardCurrent(guard: CanvasOperationGuard): boolean;
  edits?: CanvasEditGate;
}

type ActiveOperation = StartCanvasOperationOptions & {
  lease: CanvasEditLease;
  onLeaseAbort: () => void;
  requestController: AbortController | null;
  requestToken: number;
};

const errorMessage = (cause: unknown): string => (cause instanceof Error ? cause.message : String(cause));

export const createCanvasOperationController = (deps: CanvasOperationControllerDeps): CanvasOperationController => {
  const ownedEditGate = deps.edits ? null : createCanvasEditGate();
  const edits: CanvasEditGate = deps.edits ?? ownedEditGate!;
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
    operation.lease.signal.removeEventListener('abort', operation.onLeaseAbort);
    operation.lease.release();
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
    operation.lease.signal.removeEventListener('abort', operation.onLeaseAbort);
    operation.lease.release();
    cleanupPreview(operation);
    if (!disposed && active === null && state === idleState) {
      const lease = edits.tryAcquire({ kind: operation.identity.kind, layerId: operation.identity.layerId });
      if (lease) {
        operation.lease = lease;
        lease.signal.addEventListener('abort', operation.onLeaseAbort, { once: true });
        active = operation;
        publishState({ error: null, identity: operation.identity, phase: 'ready', status: 'active' });
      } else {
        publishState(idleState);
      }
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
    operation.lease.signal.removeEventListener('abort', operation.onLeaseAbort);
    operation.lease.release();
    cleanupPreview(operation);
    if (disposed || active !== null || state !== idleState) {
      return 'stale';
    }
    const lease = edits.tryAcquire({ kind: operation.identity.kind, layerId: operation.identity.layerId });
    if (!lease) {
      publishState(idleState);
      return 'stale';
    }
    operation.lease = lease;
    lease.signal.addEventListener('abort', operation.onLeaseAbort, { once: true });

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
    if (
      disposed ||
      options.identity.projectId !== options.guard.projectId ||
      options.identity.layerId !== options.guard.layerId
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
    const lease = edits.tryAcquire({ kind: options.identity.kind, layerId: options.identity.layerId });
    if (!lease) {
      return null;
    }
    let operation!: ActiveOperation;
    const onLeaseAbort = (): void => close(operation);
    operation = {
      ...options,
      lease,
      onLeaseAbort,
      requestController: null,
      requestToken: 0,
    };
    lease.signal.addEventListener('abort', onLeaseAbort, { once: true });
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
    if (active?.identity.projectId === projectId && (layerId === undefined || active.identity.layerId === layerId)) {
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
      ownedEditGate?.dispose();
    },
    getSnapshot: () => state,
    invalidateDocument: (projectId) => invalidateTarget(projectId),
    invalidateLayer: (projectId, layerId) => invalidateTarget(projectId, layerId),
    invalidateProject: (projectId) => invalidateTarget(projectId),
    invalidateSource: (projectId, layerId) => invalidateTarget(projectId, layerId),
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
