import type { S } from 'services/api/types';
import { beforeEach, describe, expect, it } from 'vitest';

import {
  getQueueStatusWithObservedEvents,
  getUpdatedQueueStatusOnQueueItemStatusChanged,
  resetObservedQueueStatusEventsForTests,
} from './queueStatusEvents';

const buildQueueStatus = (overrides: Partial<S['SessionQueueStatus']> = {}): S['SessionQueueAndProcessorStatus'] => ({
  queue: {
    queue_id: 'default',
    item_id: 1,
    batch_id: 'batch-1',
    session_id: 'session-1',
    pending: 0,
    in_progress: 1,
    waiting: 0,
    completed: 0,
    failed: 0,
    canceled: 0,
    total: 1,
    ...overrides,
  },
  processor: {
    is_started: true,
    is_processing: true,
  },
});

const buildQueueStatusChangedEvent = (
  overrides: Partial<S['QueueItemStatusChangedEvent']> = {}
): S['QueueItemStatusChangedEvent'] =>
  ({
    item_id: 1,
    status: 'completed',
    status_sequence: 2,
    queue_status: {
      queue_id: 'default',
      item_id: null,
      batch_id: null,
      session_id: null,
      pending: 0,
      in_progress: 0,
      waiting: 0,
      completed: 1,
      failed: 0,
      canceled: 0,
      total: 1,
    },
    ...overrides,
  }) as S['QueueItemStatusChangedEvent'];

beforeEach(() => {
  resetObservedQueueStatusEventsForTests();
});

describe(getUpdatedQueueStatusOnQueueItemStatusChanged.name, () => {
  it('uses the queue status from the socket event while preserving processor status', () => {
    const current = buildQueueStatus();
    const event = buildQueueStatusChangedEvent();

    expect(getUpdatedQueueStatusOnQueueItemStatusChanged(current, event)).toEqual({
      queue: event.queue_status,
      processor: current.processor,
    });
  });
});

describe(getQueueStatusWithObservedEvents.name, () => {
  it('does not let a stale current-item queue status response regress a terminal socket event', () => {
    const current = buildQueueStatus();
    const completedEvent = buildQueueStatusChangedEvent();
    getUpdatedQueueStatusOnQueueItemStatusChanged(current, completedEvent);

    const staleResponse = buildQueueStatus({ item_id: 1, in_progress: 1, completed: 0 });

    expect(getQueueStatusWithObservedEvents(staleResponse)).toEqual({
      ...staleResponse,
      queue: completedEvent.queue_status,
    });
  });

  it('accepts a later status event for the same item when its status sequence increases', () => {
    const current = buildQueueStatus();
    getUpdatedQueueStatusOnQueueItemStatusChanged(current, buildQueueStatusChangedEvent());
    const retryEvent = buildQueueStatusChangedEvent({
      status: 'in_progress',
      status_sequence: 3,
      queue_status: {
        queue_id: 'default',
        item_id: 1,
        batch_id: 'batch-1',
        session_id: 'session-1',
        pending: 0,
        in_progress: 1,
        waiting: 0,
        completed: 1,
        failed: 0,
        canceled: 0,
        total: 2,
      },
    });

    getUpdatedQueueStatusOnQueueItemStatusChanged(current, retryEvent);
    const latestResponse = buildQueueStatus({ item_id: 1, in_progress: 1, completed: 1, total: 2 });

    expect(getQueueStatusWithObservedEvents(latestResponse)).toBe(latestResponse);
  });

  it('ignores an older status event for an item with a newer terminal event', () => {
    const current = buildQueueStatus();
    const completedEvent = buildQueueStatusChangedEvent({ status_sequence: 2 });
    getUpdatedQueueStatusOnQueueItemStatusChanged(current, completedEvent);
    const staleInProgressEvent = buildQueueStatusChangedEvent({
      status: 'in_progress',
      status_sequence: 1,
      queue_status: buildQueueStatus({ item_id: 1, in_progress: 1, completed: 0 }).queue,
    });

    getUpdatedQueueStatusOnQueueItemStatusChanged(current, staleInProgressEvent);

    expect(getQueueStatusWithObservedEvents(buildQueueStatus({ item_id: 1 }))).toEqual({
      ...current,
      queue: completedEvent.queue_status,
    });
  });
});
