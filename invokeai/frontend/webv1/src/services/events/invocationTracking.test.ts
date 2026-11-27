import { describe, expect, it } from 'vitest';

import {
  clearCompletedInvocationKeysForQueueItem,
  hasCompletedInvocationKey,
  markInvocationAsCompleted,
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
