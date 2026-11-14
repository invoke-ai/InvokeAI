import { describe, expect, it } from 'vitest';

import type { QueueItem } from './historyTypes';

import {
  getAutoSwitchFollowedLive,
  getQueueActiveSessions,
  getQueueProgressSessions,
  isAutoSwitchPinHeld,
  isGalleryProgressItem,
  pickAutoSwitch,
  selectsLandedResult,
  type AutoSwitchLiveSlot,
  type AutoSwitchMode,
} from './activeSessions';

const run = (id: string, backendItemIds: number[], destination: 'gallery' | 'canvas' = 'gallery'): QueueItem => ({
  id,
  backendItemIds,
  status: 'running',
  cancellable: true,
  snapshot: {
    backendSubmission: { kind: 'invalid', error: 'not submitted by this test' },
    destination,
    filterIntermediateResults: true,
    galleryBoardId: 'destination-board',
    graph: { id: 'workflow', label: 'Multi-output workflow' },
    presentation: { batchCount: 3, width: 512, height: 768 },
    sourceId: 'workflow',
    submittedAt: '2026-09-15T00:00:00Z',
  },
});
const target = (queueItemId: string, itemIndex: number) => ({ queueItemId, itemIndex });

describe('active queue sessions', () => {
  it('shows executing sessions in backend order, without reserving pending slots or workflow outputs', () => {
    const items = [run('newer', [20]), run('batch', [10, 11, 12])];
    const targets = [target('newer', 1), target('batch', 2), target('batch', 1)];
    expect(
      getQueueActiveSessions(items, targets, targets, [target('batch', 2)]).map(
        ({ id, backendItemId, hasOwnFrame, state }) => ({ id, backendItemId, hasOwnFrame, state })
      )
    ).toEqual([
      { id: 'batch:1', backendItemId: 10, hasOwnFrame: false, state: 'running' },
      { id: 'batch:2', backendItemId: 11, hasOwnFrame: true, state: 'running' },
      { id: 'newer:1', backendItemId: 20, hasOwnFrame: false, state: 'running' },
    ]);
  });
  it('keeps a settling session until routing releases it, even after its result enters project state', () => {
    const item = { ...run('batch', [10, 11]), completedBackendItemIds: [10] };
    expect(
      getQueueActiveSessions([item], [target('batch', 2)], [target('batch', 1), target('batch', 2)], [])
    ).toMatchObject([
      { id: 'batch:1', state: 'settling' },
      { id: 'batch:2', state: 'running' },
    ]);
    expect(getQueueActiveSessions([item], [], [], [])).toEqual([]);
  });
  it('filters destinations through presentation policy and rejects targets outside the project or batch', () => {
    const items = [run('gallery', [10]), run('canvas', [11], 'canvas')];
    const targets = [target('gallery', 1), target('canvas', 1), target('other-project', 1), target('gallery', 2)];
    expect(
      getQueueActiveSessions(items.filter(isGalleryProgressItem), targets, targets, []).map((session) => session.id)
    ).toEqual(['gallery:1']);
    expect(getQueueActiveSessions(items, targets, targets, []).map((session) => session.id)).toEqual([
      'gallery:1',
      'canvas:1',
    ]);
  });
});

describe('gallery batch progress slots', () => {
  it('keeps oldest-first order while newest-first batches acquire backend ids', () => {
    const a = { ...run('a', []), backendItemIds: undefined };
    const b = {
      ...run('b', []),
      backendItemIds: undefined,
      snapshot: { ...a.snapshot, submittedAt: '2026-09-15T00:01:00Z' },
    };
    const ids = (items: QueueItem[]) => getQueueProgressSessions(items, []).map(({ id }) => id);
    const expected = ['a:1', 'a:2', 'a:3', 'b:1', 'b:2', 'b:3'];
    expect(ids([b, a])).toEqual(expected);
    expect(ids([b, { ...a, backendItemIds: [10, 11, 12] }])).toEqual(expected);
    expect(
      ids([
        { ...b, backendItemIds: [13, 14, 15] },
        { ...a, backendItemIds: [10, 11, 12] },
      ])
    ).toEqual(expected);
  });

  it('shows all three slots before submission and keeps their identities as execution starts', () => {
    const pending = { ...run('batch', []), backendItemIds: undefined, status: 'pending' as const };
    const queued = getQueueProgressSessions([pending], []);
    expect(queued.map(({ id, state }) => ({ id, state }))).toEqual([
      { id: 'batch:1', state: 'queued' },
      { id: 'batch:2', state: 'queued' },
      { id: 'batch:3', state: 'queued' },
    ]);
    const item = run('batch', [10, 11, 12]);
    const targets = [target('batch', 1)];
    const sessions = getQueueProgressSessions([item], getQueueActiveSessions([item], targets, targets, []));
    expect(sessions.map(({ id }) => id)).toEqual(queued.map(({ id }) => id));
    expect(sessions.map(({ state }) => state)).toEqual(['running', 'queued', 'queued']);
  });
  it('removes completed and cancelled slots, retaining only a settling frame until release', () => {
    const item = { ...run('batch', [10, 11, 12]), completedBackendItemIds: [10], cancelledBackendItemIds: [12] };
    expect(getQueueProgressSessions([item], []).map(({ id }) => id)).toEqual(['batch:2']);
    const active = getQueueActiveSessions([item], [], [target('batch', 1)], []);
    expect(getQueueProgressSessions([item], active).map(({ state }) => state)).toEqual(['settling', 'queued']);
    for (const status of ['completed', 'cancelled', 'failed'] as const) {
      expect(getQueueProgressSessions([{ ...item, status }], [])).toEqual([]);
    }
  });
});

describe('Auto-switch policy', () => {
  const live = (id: string, state: AutoSwitchLiveSlot['state'], hasOwnFrame: boolean): AutoSwitchLiveSlot => ({
    hasOwnFrame,
    id,
    state,
  });
  const modes: AutoSwitchMode[] = ['off', 'latest', 'progress'];

  // Each row: the generating slots in start order and the pin, and what In progress shows. Off and Newest never show a
  // generating slot.
  it.each([
    {
      expected: null,
      lives: [live('a', 'running', false)],
      pinned: null,
      pinHeld: false,
      situation: 'a slot starting without a frame of its own (model load): the previous result stays',
    },
    {
      expected: 'a',
      lives: [live('a', 'running', true)],
      pinned: null,
      pinHeld: false,
      situation: "a slot's first frame of its own",
    },
    {
      expected: 'b',
      lives: [live('a', 'running', true), live('b', 'running', true)],
      pinned: null,
      pinHeld: false,
      situation: 'two running with frames: the newest started',
    },
    {
      expected: 'a',
      lives: [live('a', 'settling', true), live('b', 'running', false)],
      pinned: null,
      pinHeld: false,
      situation: 'a finished slot whose result is on its way, while the next has no frame yet',
    },
    {
      expected: 'b',
      lives: [live('a', 'settling', true), live('b', 'running', true)],
      pinned: null,
      pinHeld: false,
      situation: 'the next slot shows its first frame while the finished one settles',
    },
    {
      expected: null,
      lives: [live('a', 'settling', false)],
      pinned: null,
      pinHeld: false,
      situation: 'a settling slot that never showed a frame',
    },
    {
      expected: 'a',
      lives: [live('a', 'running', false), live('b', 'running', true)],
      pinned: 'a',
      pinHeld: true,
      situation: 'a picked slot that runs, over a newer one with a frame, frameless or not',
    },
    {
      expected: 'b',
      lives: [live('a', 'settling', true), live('b', 'running', true)],
      pinned: 'a',
      pinHeld: false,
      situation: 'a picked slot that has finished releases the pin',
    },
    {
      expected: 'b',
      lives: [live('a', 'queued', false), live('b', 'running', true)],
      pinned: 'a',
      pinHeld: false,
      situation: 'a picked slot waiting to start holds no pin',
    },
    {
      expected: null,
      lives: [],
      pinned: 'gone',
      pinHeld: false,
      situation: 'a picked slot that has gone',
    },
  ])('under each mode, $situation', ({ expected, lives, pinHeld, pinned }) => {
    for (const mode of modes) {
      expect(getAutoSwitchFollowedLive(mode, lives, pinned)?.id ?? null).toBe(mode === 'progress' ? expected : null);
    }
    expect(isAutoSwitchPinHeld(lives, pinned)).toBe(pinHeld);
  });

  it.each([
    { hasSelectedResult: true, mode: 'off', selects: false },
    { hasSelectedResult: false, mode: 'off', selects: true },
    { hasSelectedResult: true, mode: 'latest', selects: true },
    { hasSelectedResult: false, mode: 'latest', selects: true },
    { hasSelectedResult: true, mode: 'progress', selects: true },
    { hasSelectedResult: false, mode: 'progress', selects: true },
  ] as const)(
    'a landed result under $mode, with a result selected: $hasSelectedResult, is selected: $selects',
    ({ hasSelectedResult, mode, selects }) => {
      expect(selectsLandedResult(mode, hasSelectedResult)).toBe(selects);
    }
  );

  it('picking a result pauses Auto-switch, and picking a generating slot follows and pins it, from any mode', () => {
    expect(pickAutoSwitch({ kind: 'result' })).toEqual({ mode: 'off', pinnedId: null });
    expect(pickAutoSwitch({ id: 'b', kind: 'live' })).toEqual({ mode: 'progress', pinnedId: 'b' });
  });
});
