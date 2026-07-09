import { describe, expect, it, vi } from 'vitest';

import type { HistoryEntry } from './history';

import { createHistory, HISTORY_BYTE_BUDGET, HISTORY_MAX_ENTRIES } from './history';

/** A tiny entry that records undo/redo calls into a shared log. */
const makeEntry = (label: string, log: string[], bytes = 0): HistoryEntry => ({
  bytes,
  label,
  redo: () => log.push(`redo:${label}`),
  undo: () => log.push(`undo:${label}`),
});

describe('createHistory: push / undo / redo ordering', () => {
  it('undoes and redoes entries in LIFO order', () => {
    const log: string[] = [];
    const history = createHistory();
    history.push(makeEntry('a', log));
    history.push(makeEntry('b', log));
    history.push(makeEntry('c', log));

    history.undo();
    history.undo();
    expect(log).toEqual(['undo:c', 'undo:b']);

    history.redo();
    expect(log).toEqual(['undo:c', 'undo:b', 'redo:b']);

    history.undo();
    history.undo();
    expect(log).toEqual(['undo:c', 'undo:b', 'redo:b', 'undo:b', 'undo:a']);
  });

  it('undo / redo are no-ops on empty stacks', () => {
    const log: string[] = [];
    const history = createHistory();
    history.undo();
    history.redo();
    expect(log).toEqual([]);
    expect(history.canUndo()).toBe(false);
    expect(history.canRedo()).toBe(false);
  });
});

describe('createHistory: redo cleared on push', () => {
  it('drops the redo stack when a new entry is pushed', () => {
    const log: string[] = [];
    const history = createHistory();
    history.push(makeEntry('a', log));
    history.push(makeEntry('b', log));
    history.undo(); // b -> redo
    expect(history.canRedo()).toBe(true);

    history.push(makeEntry('c', log));
    expect(history.canRedo()).toBe(false);

    // Redoing does nothing: the redo stack was cleared by the push.
    history.redo();
    expect(log).toEqual(['undo:b']);
  });
});

describe('createHistory: entry-count eviction', () => {
  it('evicts the oldest entry beyond the 64-entry budget', () => {
    const log: string[] = [];
    const history = createHistory();
    // Push one past the default cap.
    for (let i = 0; i < HISTORY_MAX_ENTRIES + 1; i += 1) {
      history.push(makeEntry(`e${i}`, log));
    }
    // Undo every retained entry: there should be exactly the cap, and the very
    // first entry (e0) should have been evicted (never undone).
    let undos = 0;
    while (history.canUndo()) {
      history.undo();
      undos += 1;
    }
    expect(undos).toBe(HISTORY_MAX_ENTRIES);
    expect(log).not.toContain('undo:e0');
    expect(log).toContain('undo:e1');
    expect(log[log.length - 1]).toBe('undo:e1');
  });

  it('honors a custom maxEntries', () => {
    const log: string[] = [];
    const history = createHistory({ maxEntries: 2 });
    history.push(makeEntry('a', log));
    history.push(makeEntry('b', log));
    history.push(makeEntry('c', log)); // evicts 'a'
    let undos = 0;
    while (history.canUndo()) {
      history.undo();
      undos += 1;
    }
    expect(undos).toBe(2);
    expect(log).toEqual(['undo:c', 'undo:b']);
  });
});

describe('createHistory: byte-budget eviction', () => {
  it('evicts oldest entries until under the byte budget', () => {
    const log: string[] = [];
    // Budget fits two 10-byte entries but not three.
    const history = createHistory({ byteBudget: 25 });
    history.push(makeEntry('a', log, 10));
    history.push(makeEntry('b', log, 10));
    expect(history.canUndo()).toBe(true);
    history.push(makeEntry('c', log, 10)); // total 30 > 25 -> evict 'a'

    let undos = 0;
    while (history.canUndo()) {
      history.undo();
      undos += 1;
    }
    expect(undos).toBe(2);
    expect(log).toEqual(['undo:c', 'undo:b']);
  });

  it('exports the 256 MB default byte budget', () => {
    expect(HISTORY_BYTE_BUDGET).toBe(256 * 1024 * 1024);
  });
});

describe('createHistory: change listener', () => {
  it('fires on push / undo / redo / clear', () => {
    const log: string[] = [];
    const history = createHistory();
    const listener = vi.fn();
    const unsubscribe = history.subscribe(listener);

    history.push(makeEntry('a', log));
    expect(listener).toHaveBeenCalledTimes(1);
    history.undo();
    expect(listener).toHaveBeenCalledTimes(2);
    history.redo();
    expect(listener).toHaveBeenCalledTimes(3);
    history.clear();
    expect(listener).toHaveBeenCalledTimes(4);

    // Clear again with empty stacks: no change, no notification.
    history.clear();
    expect(listener).toHaveBeenCalledTimes(4);

    unsubscribe();
    history.push(makeEntry('b', log));
    expect(listener).toHaveBeenCalledTimes(4);
  });

  it('reflects canUndo / canRedo transitions', () => {
    const log: string[] = [];
    const history = createHistory();
    expect(history.canUndo()).toBe(false);
    history.push(makeEntry('a', log));
    expect(history.canUndo()).toBe(true);
    expect(history.canRedo()).toBe(false);
    history.undo();
    expect(history.canUndo()).toBe(false);
    expect(history.canRedo()).toBe(true);
  });
});

describe('createHistory: amendLast', () => {
  it('replaces the most recent entry in place and adjusts the byte total', () => {
    const log: string[] = [];
    const history = createHistory({ byteBudget: 25 });
    history.push(makeEntry('a', log, 10));
    history.push(makeEntry('b', log, 10)); // burst start

    // Coalesce: replace 'b' with 'b2' (still one burst entry).
    history.amendLast(makeEntry('b2', log, 10));

    let undos = 0;
    while (history.canUndo()) {
      history.undo();
      undos += 1;
    }
    // Two entries retained (a + b2), not three: amend did not grow the stack.
    expect(undos).toBe(2);
    expect(log).toEqual(['undo:b2', 'undo:a']);
  });

  it('pushes when the undo stack is empty', () => {
    const log: string[] = [];
    const history = createHistory();
    history.amendLast(makeEntry('a', log));
    expect(history.canUndo()).toBe(true);
    history.undo();
    expect(log).toEqual(['undo:a']);
  });

  it('clears the redo stack like push', () => {
    const log: string[] = [];
    const history = createHistory();
    history.push(makeEntry('a', log));
    history.push(makeEntry('b', log));
    history.undo(); // b -> redo
    expect(history.canRedo()).toBe(true);
    history.amendLast(makeEntry('a2', log));
    expect(history.canRedo()).toBe(false);
  });

  it('is a no-op while applying', () => {
    const log: string[] = [];
    const history = createHistory();
    const reentrant: HistoryEntry = {
      bytes: 0,
      label: 'reentrant',
      redo: () => {},
      undo: () => history.amendLast(makeEntry('sneaky', log)),
    };
    history.push(reentrant);
    history.undo();
    // The amend during undo was dropped; only the reentrant entry is reachable.
    history.redo();
    let undos = 0;
    while (history.canUndo()) {
      history.undo();
      undos += 1;
    }
    expect(undos).toBe(1);
  });
});

describe('createHistory: re-entrancy guard', () => {
  it('reports isApplying during undo/redo and drops entries pushed while applying', () => {
    const log: string[] = [];
    const history = createHistory();
    const observed: boolean[] = [];

    const reentrant: HistoryEntry = {
      bytes: 0,
      label: 'reentrant',
      redo: () => {
        observed.push(history.isApplying());
        // A replay must not be able to record a new entry.
        history.push(makeEntry('sneaky', log));
      },
      undo: () => {
        observed.push(history.isApplying());
        history.push(makeEntry('sneaky', log));
      },
    };

    expect(history.isApplying()).toBe(false);
    history.push(reentrant);
    history.undo();
    expect(observed).toEqual([true]);
    expect(history.isApplying()).toBe(false);
    // The sneaky push during undo was dropped: redo stack still holds only the
    // reentrant entry, and no 'sneaky' entry is reachable.
    expect(history.canRedo()).toBe(true);

    history.redo();
    expect(observed).toEqual([true, true]);
    // Still exactly one undoable entry (the reentrant one); the sneaky pushes
    // never landed.
    let undos = 0;
    while (history.canUndo()) {
      history.undo();
      undos += 1;
    }
    expect(undos).toBe(1);
  });
});
