import type { QueueCounts } from '@features/queue/core/types';
import type { TFunction } from 'i18next';

import { describe, expect, it } from 'vitest';

import type { QueueActionScope } from './queueScope';

import { getQueueActionCounts, getQueueActionDescriptors, type QueueActionCounts } from './queueMenuActions';

// Echoes keys with their values, so the copy each action would show can be read back.
const t = ((key: string, values?: Record<string, unknown>) =>
  values ? `${key} ${JSON.stringify(values)}` : key) as unknown as TFunction;

const project: QueueActionScope = { kind: 'project', project: 'Portraits' };
const counts: QueueActionCounts = {
  hasHistory: true,
  history: { canceled: 1, completed: 4, failed: 2, total: 11 },
  inProgress: 1,
  pending: 3,
};

const describeActions = (overrides: Partial<Parameters<typeof getQueueActionDescriptors>[0]> = {}) =>
  getQueueActionDescriptors({
    canManageProcessor: true,
    counts,
    hasRunningItem: true,
    includeOpenQueue: false,
    isConnected: true,
    isPaused: false,
    localUnsubmitted: 0,
    scope: project,
    t,
    ...overrides,
  });

const find = (actions: ReturnType<typeof describeActions>, id: string) => actions.find((action) => action.id === id)!;

describe('queue action model', () => {
  it('stops the running item without asking, and asks before anything that reaches many items', () => {
    const confirmed = describeActions()
      .filter((action) => action.confirmation !== null)
      .map((action) => action.id);

    expect(find(describeActions(), 'cancel-current').confirmation).toBeNull();
    expect(confirmed).toEqual([
      'cancel-all',
      'cancel-all-except-current',
      'clear-failed',
      'clear-finished',
      'clear-queue',
    ]);
  });

  it('names the scope in a whole sentence and says how many items each confirmation reaches', () => {
    const actions = describeActions();

    expect(find(actions, 'cancel-all').confirmation?.body).toBe(
      'widgets.queue.cancelAllConfirmationBody.project {"count":4,"project":"Portraits"}'
    );
    expect(find(actions, 'cancel-all-except-current').confirmation?.body).toContain('"count":3');
    expect(find(actions, 'clear-finished').confirmation?.body).toContain('"count":7');
    expect(find(actions, 'clear-failed').confirmation?.body).toContain('"count":2');
  });

  it('says the running item keeps going only when the server is reachable and an item is running', () => {
    const body = (overrides: Parameters<typeof describeActions>[0]) =>
      find(describeActions({ localUnsubmitted: 1, ...overrides }), 'cancel-all-except-current').confirmation?.body;

    expect(body({})).toContain('widgets.queue.runningItemKeepsGoing');
    expect(body({ isConnected: false })).not.toContain('runningItemKeepsGoing');
    expect(body({ hasRunningItem: false })).not.toContain('runningItemKeepsGoing');
  });

  it('says that clearing the queue also stops what still runs, and leaves that out when nothing does', () => {
    const busy = find(describeActions(), 'clear-queue').confirmation?.body;
    const idle = find(describeActions({ counts: { ...counts, inProgress: 0, pending: 0 } }), 'clear-queue').confirmation
      ?.body;

    expect(busy).toContain('widgets.queue.clearQueueStopsWork {"count":4}');
    expect(busy).toContain('clearQueueConfirmationBody.project {"count":11,"project":"Portraits"}');
    expect(idle).not.toContain('clearQueueStopsWork');
  });

  it('never offers a "Cancel" way out of a cancel', () => {
    for (const action of describeActions()) {
      if (action.confirmation) {
        expect(action.confirmation.cancelLabel).not.toBe('common.cancel');
      }
    }

    expect(find(describeActions(), 'cancel-all').confirmation?.cancelLabel).toBe('widgets.queue.keepRunning');
  });

  it('offers one processor toggle that reflects its state and what waits on it', () => {
    const running = find(describeActions(), 'toggle-processor');
    const paused = find(describeActions({ isPaused: true }), 'toggle-processor');

    expect(running).toMatchObject({ label: 'widgets.queue.pauseProcessor' });
    expect(running.hint).toBeUndefined();
    expect(paused).toMatchObject({
      hint: 'widgets.queue.processorPausedWaiting {"count":3}',
      label: 'widgets.queue.resumeProcessor',
    });
  });

  it('disables what needs the server while it is disconnected, but still opens the queue', () => {
    const offline = describeActions({ includeOpenQueue: true, isConnected: false });

    expect(offline.filter((action) => !action.disabled).map((action) => action.id)).toEqual(['open-queue']);
  });

  it("keeps cancel-all available offline for this tab's runs the backend has not accepted, counting only those", () => {
    const cancelAll = find(describeActions({ isConnected: false, localUnsubmitted: 2 }), 'cancel-all');

    expect(cancelAll.disabled).toBe(false);
    expect(cancelAll.confirmation?.body).toContain('"count":2');
  });

  it("keeps cancel-all-except-current available offline for this tab's unaccepted runs, counting only those", () => {
    const exceptCurrent = find(
      describeActions({ isConnected: false, localUnsubmitted: 2 }),
      'cancel-all-except-current'
    );

    expect(exceptCurrent.disabled).toBe(false);
    expect(exceptCurrent.confirmation?.body).toContain('"count":2');
  });

  it("counts this tab's unaccepted runs with the server's queued items when cancelling all but the current one", () => {
    const connected = find(describeActions({ localUnsubmitted: 2 }), 'cancel-all-except-current');
    const localOnly = find(
      describeActions({ counts: { ...counts, pending: 0 }, localUnsubmitted: 1 }),
      'cancel-all-except-current'
    );

    expect(connected.confirmation?.body).toContain('"count":5');
    expect(localOnly.disabled).toBe(false);
    expect(localOnly.confirmation?.body).toContain('"count":1');
  });

  it('disables bulk actions with nothing to reach', () => {
    const empty = describeActions({
      counts: {
        hasHistory: false,
        history: { canceled: 0, completed: 0, failed: 0, total: 0 },
        inProgress: 0,
        pending: 0,
      },
      hasRunningItem: false,
    });

    expect(empty.filter((action) => !action.disabled).map((action) => action.id)).toEqual(['toggle-processor']);
  });
});

describe('queue action counts', () => {
  const server: QueueCounts = {
    canceled: 5,
    completed: 20,
    failed: 5,
    inProgress: 3,
    pending: 7,
    queueId: 'default',
    total: 40,
    userInProgress: 1,
    userPending: 2,
    waiting: 0,
  };

  it("counts a member's own unfinished work, and no history across all projects, where the server counts everyone's", () => {
    expect(getQueueActionCounts(server, { kind: 'mine' }, 'member')).toEqual({
      hasHistory: true,
      history: null,
      inProgress: 1,
      pending: 2,
    });
  });

  it("counts everyone's work for an administrator, whose cancel-all stops it all", () => {
    expect(getQueueActionCounts(server, { kind: 'allUsers' }, 'admin')).toMatchObject({
      history: { total: 40 },
      inProgress: 3,
      pending: 7,
    });
  });

  it('confirms a member clear across projects without a count it cannot know', () => {
    const body = find(
      describeActions({ counts: getQueueActionCounts(server, { kind: 'mine' }, 'member'), scope: { kind: 'mine' } }),
      'clear-queue'
    ).confirmation?.body;

    expect(body).toMatch(/^widgets\.queue\.clearQueueConfirmationBody\.mine( |$)/);
    expect(body).not.toContain('"count":40');
  });
});
