import { describe, expect, it, vi } from 'vitest';

import type { History, HistoryEntry } from './history';

import { createHistory, HISTORY_BYTE_BUDGET, HISTORY_MAX_ENTRIES, NO_HELD_ASSET_REFS } from './history';

const makeEntry = (label: string, log: string[], bytes = 1): HistoryEntry => ({
  bytes,
  heldAssetRefs: NO_HELD_ASSET_REFS,
  label,
  redo: () => {
    log.push(`redo:${label}`);
  },
  undo: () => {
    log.push(`undo:${label}`);
  },
});

const push = (history: History, entry: HistoryEntry) => {
  const admission = history.admit(entry.bytes);
  if (!admission) {
    throw new Error(`"${entry.label}" was not admitted`);
  }
  return admission.publish(entry);
};

describe('createHistory: replay ordering', () => {
  it('undoes and redoes entries in LIFO order', async () => {
    const log: string[] = [];
    const history = createHistory();
    push(history, makeEntry('a', log));
    push(history, makeEntry('b', log));

    await history.undo();
    await history.undo();
    await history.redo();

    expect(log).toEqual(['undo:b', 'undo:a', 'redo:a']);
    expect(history.entries()).toEqual({ future: ['b'], past: ['a'] });
  });

  it('reports empty stacks without replaying', async () => {
    const history = createHistory();
    await expect(history.undo()).resolves.toEqual({ status: 'empty' });
    await expect(history.redo()).resolves.toEqual({ status: 'empty' });
  });

  it('drops the redo stack when a new entry is published', async () => {
    const log: string[] = [];
    const history = createHistory();
    push(history, makeEntry('a', log));
    await history.undo();
    push(history, makeEntry('b', log));

    expect(history.canRedo()).toBe(false);
    expect(history.entries()).toEqual({ future: [], past: ['b'] });
  });
});

describe('createHistory: entries', () => {
  it('lists past oldest-first and future next-redo-first, tracking every mutation', async () => {
    const log: string[] = [];
    const history = createHistory();
    expect(history.entries()).toEqual({ future: [], past: [] });

    push(history, makeEntry('a', log));
    push(history, makeEntry('b', log));
    push(history, makeEntry('c', log));
    expect(history.entries()).toEqual({ future: [], past: ['a', 'b', 'c'] });

    await history.undo();
    await history.undo();
    expect(history.entries()).toEqual({ future: ['b', 'c'], past: ['a'] });

    await history.redo();
    expect(history.entries()).toEqual({ future: ['c'], past: ['a', 'b'] });

    push(history, makeEntry('d', log));
    expect(history.entries()).toEqual({ future: [], past: ['a', 'b', 'd'] });

    history.clear();
    expect(history.entries()).toEqual({ future: [], past: [] });
  });
});

describe('createHistory: admission', () => {
  it('reserves capacity without clearing redo or evicting retained steps', async () => {
    const log: string[] = [];
    const history = createHistory({ byteBudget: 10 });
    push(history, makeEntry('kept', log, 6));
    push(history, makeEntry('undone', log, 2));
    await history.undo();

    const admission = history.admit(10);

    expect(admission).not.toBeNull();
    expect(history.entries()).toEqual({ future: ['undone'], past: ['kept'] });
    admission!.release();
    admission!.release();
    expect(history.admit(10)).not.toBeNull();
  });

  it('refuses an entry that could never be retained and leaves earlier steps undoable', () => {
    const log: string[] = [];
    const history = createHistory({ byteBudget: 10 });
    push(history, makeEntry('previous', log, 2));

    expect(history.admit(11)).toBeNull();
    expect(history.entries().past).toEqual(['previous']);
    expect(history.canUndo()).toBe(true);
  });

  it('keeps concurrent admissions within the budget so publishing one cannot evict the other', () => {
    const history = createHistory({ byteBudget: 10 });
    const first = history.admit(6)!;
    expect(history.admit(6)).toBeNull();
    const second = history.admit(4)!;

    first.publish(makeEntry('first', [], 6));
    second.publish(makeEntry('second', [], 4));

    expect(history.entries().past).toEqual(['first', 'second']);
  });

  it('grows an admission only while the larger entry still fits', () => {
    const history = createHistory({ byteBudget: 10 });
    const admission = history.admit(4)!;
    const other = history.admit(3)!;

    expect(admission.grow(3)).toBe(true);
    expect(admission.bytes).toBe(7);
    expect(admission.grow(1)).toBe(false);
    expect(admission.bytes).toBe(7);
    other.release();
    expect(admission.grow(3)).toBe(true);
  });

  it('refuses to publish an entry larger than its admission', () => {
    const history = createHistory();
    const admission = history.admit(1)!;
    expect(() => admission.publish(makeEntry('too large', [], 2))).toThrow(/exceeds its admission/);
    expect(history.canUndo()).toBe(false);
  });
});

describe('createHistory: tokens and replacement', () => {
  it('replaces only the expected newest entry and returns a fresh token', () => {
    const log: string[] = [];
    const history = createHistory();
    const first = push(history, makeEntry('nudge', log));
    expect(history.top()).toBe(first);

    const second = history.admit(1)!.publish(makeEntry('nudge again', log), first);

    expect(second).not.toBe(first);
    expect(history.top()).toBe(second);
    expect(history.entries().past).toEqual(['nudge again']);
    expect(() => history.admit(1)!.publish(makeEntry('stale', log), first)).toThrow(/no longer the newest/);
  });
});

describe('createHistory: eviction', () => {
  it('keeps undo and redo media held until their entry is evicted', async () => {
    const history = createHistory({ maxEntries: 1 });
    const first = { ...makeEntry('remove first', []), heldAssetRefs: { images: ['first.png'], videos: [] } };
    const second = { ...makeEntry('remove second', []), heldAssetRefs: { images: ['second.png'], videos: [] } };
    push(history, first);
    await history.undo();
    expect(history.heldAssetRefs().images).toEqual(['first.png']);
    await history.redo();
    push(history, second);
    expect(history.heldAssetRefs().images).toEqual(['second.png']);
    history.clear();
    expect(history.heldAssetRefs().images).toEqual([]);
  });

  it('evicts the oldest entry beyond the entry budget', () => {
    const history = createHistory();
    for (let index = 0; index <= HISTORY_MAX_ENTRIES; index += 1) {
      push(history, makeEntry(`entry-${index}`, []));
    }
    expect(history.entries().past).toHaveLength(HISTORY_MAX_ENTRIES);
    expect(history.entries().past[0]).toBe('entry-1');
  });

  it('evicts oldest entries until under the byte budget, never the newest', () => {
    const history = createHistory({ byteBudget: 10 });
    push(history, makeEntry('a', [], 4));
    push(history, makeEntry('b', [], 4));
    push(history, makeEntry('c', [], 4));

    expect(history.entries().past).toEqual(['b', 'c']);
    expect(history.byteSize()).toBe(8);
    expect(HISTORY_BYTE_BUDGET).toBe(256 * 1024 * 1024);
  });

  it('trims oldest entries by retained bytes while preserving the newest undo history', () => {
    const log: string[] = [];
    const history = createHistory({ byteBudget: 1_000 });
    push(history, makeEntry('oldest', log, 40));
    push(history, makeEntry('middle', log, 50));
    push(history, makeEntry('newest', log, 60));

    history.trimToBytes(110);

    expect(history.byteSize()).toBe(110);
    expect(history.entries().past).toEqual(['middle', 'newest']);
  });

  it('disposes entries whenever stack ownership permanently ends', async () => {
    const disposed: string[] = [];
    const entry = (label: string) => ({ ...makeEntry(label, []), dispose: () => disposed.push(label) });
    const history = createHistory({ maxEntries: 1 });

    const evicted = push(history, entry('evicted'));
    void evicted;
    push(history, entry('redo-cleared'));
    expect(disposed).toEqual(['evicted']);
    await history.undo();
    const replaced = push(history, entry('replaced'));
    expect(disposed).toEqual(['evicted', 'redo-cleared']);
    history.admit(1)!.publish(entry('cleared'), replaced);
    expect(disposed).toEqual(['evicted', 'redo-cleared', 'replaced']);
    history.clear();
    expect(disposed).toEqual(['evicted', 'redo-cleared', 'replaced', 'cleared']);
  });
});

describe('createHistory: failure-atomic asynchronous replay', () => {
  it('keeps a failed undo in place for an exact retry', async () => {
    const history = createHistory();
    let fail = true;
    const undo = vi.fn(() => {
      if (fail) {
        throw new Error('pixels unavailable');
      }
    });
    push(history, { ...makeEntry('stroke', []), undo });

    await expect(history.undo()).resolves.toMatchObject({ label: 'stroke', status: 'failed' });
    expect(history.entries()).toEqual({ future: [], past: ['stroke'] });

    fail = false;
    await expect(history.undo()).resolves.toEqual({ status: 'applied' });
    expect(history.entries()).toEqual({ future: ['stroke'], past: [] });
    expect(undo).toHaveBeenCalledTimes(2);
  });

  it('keeps a rejected asynchronous redo in place', async () => {
    const history = createHistory();
    push(history, { ...makeEntry('merge', []), redo: () => Promise.reject(new Error('over budget')) });
    await history.undo();

    await expect(history.redo()).resolves.toMatchObject({ status: 'failed' });
    expect(history.entries()).toEqual({ future: ['merge'], past: [] });
    expect(history.byteSize()).toBe(1);
  });

  it('refuses a second replay while one is pending and reports replay state', async () => {
    const history = createHistory();
    let finish!: () => void;
    push(history, makeEntry('a', []));
    push(history, {
      ...makeEntry('b', []),
      undo: () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    });

    const pending = history.undo();
    expect(history.isReplaying()).toBe(true);
    await expect(history.undo()).resolves.toEqual({ status: 'busy' });
    finish();
    await expect(pending).resolves.toEqual({ status: 'applied' });
    expect(history.isReplaying()).toBe(false);
    expect(history.entries()).toEqual({ future: ['b'], past: ['a'] });
  });

  it('lets a replay that clears history win instead of resurrecting its entry', async () => {
    const history = createHistory();
    push(history, { ...makeEntry('replace document', []), undo: () => history.clear() });

    await history.undo();

    expect(history.entries()).toEqual({ future: [], past: [] });
  });
});

describe('createHistory: change listener', () => {
  it('fires on publication, replay and clear, isolating faulty observers', async () => {
    const history = createHistory();
    const listener = vi.fn();
    history.subscribe(() => {
      throw new Error('observer failed');
    });
    history.subscribe(listener);

    push(history, makeEntry('a', []));
    await history.undo();
    history.clear();

    expect(listener.mock.calls.length).toBeGreaterThanOrEqual(3);
    expect(history.canUndo()).toBe(false);
  });
});
