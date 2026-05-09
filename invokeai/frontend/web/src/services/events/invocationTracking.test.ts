import { describe, expect, it } from 'vitest';

import {
  clearCompletedInvocationKeysForQueueItem,
  hasCompletedInvocationKey,
  markInvocationAsCompleted,
  shouldIgnoreFinishedQueueItemInvocationEvent,
} from './invocationTracking';

describe(markInvocationAsCompleted.name, () => {
  it('tracks completed invocations per queue item', () => {
    const completedInvocationKeysByItemId = new Map<number, Set<string>>();

    markInvocationAsCompleted(completedInvocationKeysByItemId, 1, 'prepared-node-1');
    markInvocationAsCompleted(completedInvocationKeysByItemId, 2, 'prepared-node-1');

    expect(hasCompletedInvocationKey(completedInvocationKeysByItemId, 1, 'prepared-node-1')).toBe(true);
    expect(hasCompletedInvocationKey(completedInvocationKeysByItemId, 2, 'prepared-node-1')).toBe(true);
  });

  it('clears only the completed invocations for a finished queue item', () => {
    const completedInvocationKeysByItemId = new Map<number, Set<string>>();

    markInvocationAsCompleted(completedInvocationKeysByItemId, 1, 'prepared-node-1');
    markInvocationAsCompleted(completedInvocationKeysByItemId, 2, 'prepared-node-1');

    clearCompletedInvocationKeysForQueueItem(completedInvocationKeysByItemId, 1);

    expect(hasCompletedInvocationKey(completedInvocationKeysByItemId, 1, 'prepared-node-1')).toBe(false);
    expect(hasCompletedInvocationKey(completedInvocationKeysByItemId, 2, 'prepared-node-1')).toBe(true);
  });
});

describe(shouldIgnoreFinishedQueueItemInvocationEvent.name, () => {
  it('ignores late started and progress events for finished queue items', () => {
    const finishedQueueItemIds = new Set<number>([1]);

    expect(shouldIgnoreFinishedQueueItemInvocationEvent('invocation_started', finishedQueueItemIds, 1)).toBe(true);
    expect(shouldIgnoreFinishedQueueItemInvocationEvent('invocation_progress', finishedQueueItemIds, 1)).toBe(true);
  });

  it('does not ignore late error events for finished queue items', () => {
    const finishedQueueItemIds = new Set<number>([1]);

    expect(shouldIgnoreFinishedQueueItemInvocationEvent('invocation_error', finishedQueueItemIds, 1)).toBe(false);
  });
});
