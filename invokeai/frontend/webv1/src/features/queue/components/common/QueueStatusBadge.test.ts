import { describe, expect, it } from 'vitest';

import { getQueueStatusBadgeState } from './QueueStatusBadge';

describe('QueueStatusBadge', () => {
  it('maps waiting status to the waiting translation key and color', () => {
    expect(getQueueStatusBadgeState('waiting')).toEqual({
      colorScheme: 'purple',
      translationKey: 'queue.waiting',
    });
  });

  it('keeps existing retry-terminal statuses distinct', () => {
    expect(getQueueStatusBadgeState('failed')).toEqual({
      colorScheme: 'red',
      translationKey: 'queue.failed',
    });
    expect(getQueueStatusBadgeState('canceled')).toEqual({
      colorScheme: 'orange',
      translationKey: 'queue.canceled',
    });
  });
});
