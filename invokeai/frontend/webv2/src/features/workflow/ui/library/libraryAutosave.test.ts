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
    expect(h.statuses).toEqual(['dirty', 'saving', 'saved']);
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
    expect(h.statuses.at(-1)).toBe('error');
    h.setCurrent({ libraryWorkflowId: 'wf-1', serialized: { nodes: [2] } });
    h.autosaver.notifyGraphChanged();
    h.manual.fire();
    await h.autosaver.flush();
    expect(h.save).toHaveBeenCalledTimes(2);
    expect(h.statuses.at(-1)).toBe('saved');
  });

  it('dispose cancels pending work', () => {
    const h = createHarness();
    h.autosaver.notifyGraphChanged();
    h.autosaver.dispose();
    h.manual.fire();
    expect(h.save).not.toHaveBeenCalled();
  });
});
