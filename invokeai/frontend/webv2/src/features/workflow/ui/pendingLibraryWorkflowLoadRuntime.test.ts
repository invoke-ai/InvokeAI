import { describe, expect, it, vi } from 'vitest';

import type { WorkflowLoadRequest, WorkflowLoadSource } from './workflowUiStore';

import { startPendingWorkflowLoadRuntime } from './pendingLibraryWorkflowLoadRuntime';

const deferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>((res) => {
    resolve = res;
  });

  return { promise, resolve };
};

describe('pending workflow load runtime', () => {
  it('serializes loads, keeps only the latest queued request, and compare-clears completions', async () => {
    const listeners = new Set<() => void>();
    const loads = new Map<string, ReturnType<typeof deferred>>();
    const load = vi.fn((source: WorkflowLoadSource) => {
      const pending = deferred();
      loads.set(source.kind === 'library' ? source.workflowId : source.label, pending);
      return pending.promise;
    });
    const loadedIds = () =>
      load.mock.calls.map(([source]) => (source.kind === 'library' ? source.workflowId : source.label));
    let request: WorkflowLoadRequest | null = null;
    const clearRequest = vi.fn((requestId: number) => {
      if (request?.requestId === requestId) {
        request = null;
      }
    });
    const emit = (next: WorkflowLoadRequest) => {
      request = next;
      listeners.forEach((listener) => listener());
    };
    const stop = startPendingWorkflowLoadRuntime({
      clearRequest,
      getRequest: () => request,
      load,
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    });

    emit({ requestId: 1, source: { kind: 'library', workflowId: 'first' } });
    emit({ requestId: 2, source: { kind: 'library', workflowId: 'superseded' } });
    emit({ requestId: 3, source: { kind: 'document', label: 'latest', raw: {} } });
    expect(loadedIds()).toEqual(['first']);

    loads.get('first')?.resolve();
    await vi.waitFor(() => expect(loadedIds()).toEqual(['first', 'latest']));
    expect(request).toEqual({ requestId: 3, source: { kind: 'document', label: 'latest', raw: {} } });

    loads.get('latest')?.resolve();
    await vi.waitFor(() => expect(request).toBeNull());
    expect(clearRequest.mock.calls.map(([requestId]) => requestId)).toEqual([1, 3]);
    stop();
  });

  it('does not clear shared state when an in-flight load settles after the runtime stops', async () => {
    const pending = deferred();
    const clearRequest = vi.fn();
    const stop = startPendingWorkflowLoadRuntime({
      clearRequest,
      getRequest: () => ({ requestId: 1, source: { kind: 'library', workflowId: 'user-a-workflow' } }),
      load: () => pending.promise,
      subscribe: () => () => undefined,
    });

    stop();
    pending.resolve();
    await pending.promise;
    await Promise.resolve();

    expect(clearRequest).not.toHaveBeenCalled();
  });
});
