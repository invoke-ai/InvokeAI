import { describe, expect, it, vi } from 'vitest';

import { createLibraryAutosaver, type LibrarySyncStatus } from './libraryAutosave';

const createManualTimers = () => {
  const pending = new Map<number, () => void>();
  let nextHandle = 1;
  return {
    fire: () => {
      const jobs = [...pending.values()];
      pending.clear();
      for (const job of jobs) {
        job();
      }
    },
    pendingCount: () => pending.size,
    timers: {
      clearTimeout: (handle: number) => void pending.delete(handle),
      setTimeout: (fn: () => void, _ms: number) => {
        const handle = nextHandle++;
        pending.set(handle, fn);
        return handle;
      },
    },
  };
};

const createHarness = (initial: { id?: string; graph?: Record<string, unknown> } = {}) => {
  let current = { libraryWorkflowId: initial.id ?? 'wf-1', serialized: initial.graph ?? { nodes: [1] } };
  const statuses: LibrarySyncStatus[] = [];
  const save = vi.fn(() => Promise.resolve());
  const manual = createManualTimers();
  const autosaver = createLibraryAutosaver({
    onStatus: (status) => void statuses.push(status),
    read: () => current,
    save,
    timers: manual.timers,
  });
  return { autosaver, manual, save, setCurrent: (next: typeof current) => (current = next), statuses };
};

describe('createLibraryAutosaver', () => {
  it('debounces a change and saves the bound workflow', async () => {
    const h = createHarness();
    h.autosaver.notifyGraphChanged();
    expect(h.save).not.toHaveBeenCalled();
    h.manual.fire();
    await h.autosaver.flush();
    expect(h.save).toHaveBeenCalledWith('wf-1', { nodes: [1] });
    expect(h.statuses).toEqual(['dirty', 'saving', 'saved', 'saved']);
  });

  it('skips saving when the serialized graph has not changed since the last save', async () => {
    const h = createHarness();
    h.autosaver.notifyGraphChanged();
    h.manual.fire();
    await h.autosaver.flush();
    h.autosaver.notifyGraphChanged(); // same content
    h.manual.fire();
    await h.autosaver.flush();
    expect(h.save).toHaveBeenCalledTimes(1);
    expect(h.statuses).toEqual(['dirty', 'saving', 'saved', 'saved', 'dirty', 'saved', 'saved']);
  });

  it('markSynced suppresses the echo save after a load/bind', async () => {
    const h = createHarness();
    h.autosaver.markSynced({ nodes: [1] });
    h.autosaver.notifyGraphChanged();
    h.manual.fire();
    await h.autosaver.flush();
    expect(h.save).not.toHaveBeenCalled();
  });

  it('does nothing for an unbound workflow', () => {
    const h = createHarness();
    h.setCurrent({ libraryWorkflowId: undefined as unknown as string, serialized: { nodes: [1] } });
    h.autosaver.notifyGraphChanged();
    expect(h.manual.pendingCount()).toBe(0);
    expect(h.statuses).toEqual([]);
  });

  it('reports error status on a failed save and retries on the next change', async () => {
    const h = createHarness();
    h.save.mockRejectedValueOnce(new Error('offline'));
    h.autosaver.notifyGraphChanged();
    h.manual.fire();
    await h.autosaver.flush();
    // Error is reported, then chained retry succeeds
    expect(h.statuses).toContain('error');
    expect(h.statuses.at(-1)).toBe('saved');
    h.setCurrent({ libraryWorkflowId: 'wf-1', serialized: { nodes: [2] } });
    h.autosaver.notifyGraphChanged();
    h.manual.fire();
    await h.autosaver.flush();
    // Called 3 times: initial (failed), chained retry, new content
    expect(h.save).toHaveBeenCalledTimes(3);
    expect(h.statuses.at(-1)).toBe('saved');
  });

  it('flushes a pending edit on dispose instead of dropping it', () => {
    const h = createHarness();
    h.autosaver.notifyGraphChanged();
    expect(h.save).not.toHaveBeenCalled();

    h.autosaver.dispose();

    expect(h.save).toHaveBeenCalledTimes(1);
    expect(h.manual.pendingCount()).toBe(0);
  });

  it('does not save on dispose when nothing is pending', () => {
    const h = createHarness();
    h.autosaver.dispose();
    expect(h.save).not.toHaveBeenCalled();
  });

  it('a second dispose() is a no-op', () => {
    const h = createHarness();
    h.autosaver.notifyGraphChanged();
    h.autosaver.dispose();
    expect(h.save).toHaveBeenCalledTimes(1);

    h.autosaver.dispose();

    expect(h.save).toHaveBeenCalledTimes(1);
    expect(h.manual.pendingCount()).toBe(0);
  });

  it('flush() after dispose() performs no save and resolves', async () => {
    const h = createHarness();
    h.autosaver.dispose();

    const result = h.autosaver.flush();

    expect(h.save).not.toHaveBeenCalled();
    await expect(result).resolves.toBeUndefined();
  });

  it('dispose does not emit status callbacks for the flush save', async () => {
    const h = createHarness();
    h.autosaver.notifyGraphChanged();
    h.statuses.length = 0; // drop the 'dirty' status from notifyGraphChanged
    h.autosaver.dispose();
    await vi.waitFor(() => expect(h.save).toHaveBeenCalledTimes(1));
    expect(h.statuses).toEqual([]);
  });

  it('disposing while a save is in flight with a newer edit pending still runs the chained rerun', async () => {
    const h = createHarness();
    let resolveSave: () => void;
    const deferred = new Promise<void>((resolve) => {
      resolveSave = resolve;
    });
    h.save.mockReturnValueOnce(deferred);
    h.autosaver.notifyGraphChanged();
    h.manual.fire();
    // First save is now in flight (unresolved).
    h.setCurrent({ libraryWorkflowId: 'wf-1', serialized: { nodes: [2] } });
    h.autosaver.notifyGraphChanged();

    h.autosaver.dispose();
    // The pending debounce timer was flushed into a chained rerun queued behind
    // the in-flight save, so no new timer should remain.
    expect(h.manual.pendingCount()).toBe(0);

    resolveSave!();
    // The in-flight promise is never cancelled by dispose; it settles, then the
    // chained runSave() picks up the newer edit and saves it too.
    await vi.waitFor(() => expect(h.save).toHaveBeenCalledTimes(2));
    expect(h.save).toHaveBeenLastCalledWith('wf-1', { nodes: [2] });
  });

  it('dispose with a pending edit whose save rejects does not throw or emit status', async () => {
    const h = createHarness();
    h.save.mockRejectedValueOnce(new Error('offline'));
    h.autosaver.notifyGraphChanged();
    h.statuses.length = 0;

    h.autosaver.dispose();
    await vi.waitFor(() => expect(h.save).toHaveBeenCalledTimes(1));
    // Give the rejection handler's microtask a chance to run before asserting silence.
    await Promise.resolve();
    expect(h.statuses).toEqual([]);
  });

  it('a change made during an in-flight save is saved afterward', async () => {
    const h = createHarness();
    let resolveSave: () => void;
    const deferred = new Promise<void>((resolve) => {
      resolveSave = resolve;
    });
    h.save.mockReturnValueOnce(deferred);
    h.autosaver.notifyGraphChanged();
    h.manual.fire();
    // Save is now in flight, pending on deferred
    h.setCurrent({ libraryWorkflowId: 'wf-1', serialized: { nodes: [2] } });
    h.autosaver.notifyGraphChanged();
    const flushPromise = h.autosaver.flush();
    // Resolve first save, which should chain another save for the new content
    resolveSave!();
    h.save.mockReturnValueOnce(Promise.resolve());
    await flushPromise;
    expect(h.save).toHaveBeenCalledTimes(2);
    expect(h.save).toHaveBeenLastCalledWith('wf-1', { nodes: [2] });
    expect(h.statuses.at(-1)).toBe('saved');
  });

  it('an edit that dedupes back to the saved content ends in saved, not dirty', async () => {
    const h = createHarness();
    h.autosaver.notifyGraphChanged();
    h.manual.fire();
    await h.autosaver.flush();
    h.autosaver.notifyGraphChanged(); // same content
    h.manual.fire();
    await h.autosaver.flush();
    expect(h.statuses.at(-1)).toBe('saved');
  });
});
