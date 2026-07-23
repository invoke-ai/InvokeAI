import { describe, expect, it, vi } from 'vitest';

import type { LibraryWorkflowLoadRequest } from './workflowUiStore';

import { startPendingLibraryWorkflowLoadRuntime } from './pendingLibraryWorkflowLoadRuntime';

const deferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>((res) => {
    resolve = res;
  });

  return { promise, resolve };
};

describe('pending library workflow load runtime', () => {
  it('serializes loads, keeps only the latest queued request, and compare-clears completions', async () => {
    const listeners = new Set<() => void>();
    const loads = new Map<string, ReturnType<typeof deferred>>();
    const load = vi.fn((workflowId: string) => {
      const pending = deferred();
      loads.set(workflowId, pending);
      return pending.promise;
    });
    let request: LibraryWorkflowLoadRequest | null = null;
    const clearRequest = vi.fn((requestId: number) => {
      if (request?.requestId === requestId) {
        request = null;
      }
    });
    const emit = (next: LibraryWorkflowLoadRequest) => {
      request = next;
      listeners.forEach((listener) => listener());
    };
    const stop = startPendingLibraryWorkflowLoadRuntime({
      clearRequest,
      getRequest: () => request,
      load,
      subscribe: (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    });

    emit({ requestId: 1, workflowId: 'first' });
    emit({ requestId: 2, workflowId: 'superseded' });
    emit({ requestId: 3, workflowId: 'latest' });
    expect(load.mock.calls.map(([workflowId]) => workflowId)).toEqual(['first']);

    loads.get('first')?.resolve();
    await vi.waitFor(() => expect(load.mock.calls.map(([workflowId]) => workflowId)).toEqual(['first', 'latest']));
    expect(request).toEqual({ requestId: 3, workflowId: 'latest' });

    loads.get('latest')?.resolve();
    await vi.waitFor(() => expect(request).toBeNull());
    expect(clearRequest.mock.calls.map(([requestId]) => requestId)).toEqual([1, 3]);
    stop();
  });
});
