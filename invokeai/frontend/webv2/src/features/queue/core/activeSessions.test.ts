import { describe, expect, it } from 'vitest';

import type { QueueItem } from './historyTypes';

import {
  getFollowedProgressSession,
  getQueueActiveSessions,
  getQueueProgressSessions,
  isGalleryProgressItem,
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
      getQueueActiveSessions(items, targets, targets).map(({ id, backendItemId, state }) => ({
        id,
        backendItemId,
        state,
      }))
    ).toEqual([
      { id: 'batch:1', backendItemId: 10, state: 'running' },
      { id: 'batch:2', backendItemId: 11, state: 'running' },
      { id: 'newer:1', backendItemId: 20, state: 'running' },
    ]);
  });
  it('keeps a settling session until routing releases it, even after its result enters project state', () => {
    const item = { ...run('batch', [10, 11]), completedBackendItemIds: [10] };
    expect(
      getQueueActiveSessions([item], [target('batch', 2)], [target('batch', 1), target('batch', 2)])
    ).toMatchObject([
      { id: 'batch:1', state: 'settling' },
      { id: 'batch:2', state: 'running' },
    ]);
    expect(getQueueActiveSessions([item], [], [])).toEqual([]);
  });
  it('filters destinations through presentation policy and rejects targets outside the project or batch', () => {
    const items = [run('gallery', [10]), run('canvas', [11], 'canvas')];
    const targets = [target('gallery', 1), target('canvas', 1), target('other-project', 1), target('gallery', 2)];
    expect(
      getQueueActiveSessions(items.filter(isGalleryProgressItem), targets, targets).map((session) => session.id)
    ).toEqual(['gallery:1']);
    expect(getQueueActiveSessions(items, targets, targets).map((session) => session.id)).toEqual([
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
    const sessions = getQueueProgressSessions([item], getQueueActiveSessions([item], targets, targets));
    expect(sessions.map(({ id }) => id)).toEqual(queued.map(({ id }) => id));
    expect(sessions.map(({ state }) => state)).toEqual(['running', 'queued', 'queued']);
  });
  it('removes completed and cancelled slots, retaining only a settling frame until release', () => {
    const item = { ...run('batch', [10, 11, 12]), completedBackendItemIds: [10], cancelledBackendItemIds: [12] };
    expect(getQueueProgressSessions([item], []).map(({ id }) => id)).toEqual(['batch:2']);
    const active = getQueueActiveSessions([item], [], [target('batch', 1)]);
    expect(getQueueProgressSessions([item], active).map(({ state }) => state)).toEqual(['settling', 'queued']);
    for (const status of ['completed', 'cancelled', 'failed'] as const) {
      expect(getQueueProgressSessions([{ ...item, status }], [])).toEqual([]);
    }
  });
});

describe('getFollowedProgressSession', () => {
  const sessions = [
    { id: 'queued', state: 'queued' as const },
    { id: 'settling', state: 'settling' as const },
    { id: 'running-1', state: 'running' as const },
    { id: 'running-2', state: 'running' as const },
  ];

  it('prefers the pinned session, then the first running, then the first settling, never a queued one', () => {
    expect(getFollowedProgressSession(sessions, 'running-2')?.id).toBe('running-2');
    expect(getFollowedProgressSession(sessions, null)?.id).toBe('running-1');
    expect(getFollowedProgressSession(sessions, 'gone')?.id).toBe('running-1');
    expect(getFollowedProgressSession(sessions.slice(0, 2), null)?.id).toBe('settling');
    expect(getFollowedProgressSession(sessions.slice(0, 1), null)).toBeNull();
  });
});
