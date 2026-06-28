import { describe, expect, it } from 'vitest';

import { getQueueHeaderCancelState } from './QueueHeaderActions';

describe('getQueueHeaderCancelState', () => {
  it('enables cancelling when connected with an in-progress current item', () => {
    expect(getQueueHeaderCancelState({ currentStatus: 'in_progress', isConnected: true })).toEqual({
      disabled: false,
      itemLabel: 'Cancel Current',
    });
  });

  it('disables cancelling when disconnected or there is no current in-progress item', () => {
    expect(getQueueHeaderCancelState({ currentStatus: 'in_progress', isConnected: false }).disabled).toBe(true);
    expect(getQueueHeaderCancelState({ currentStatus: 'pending', isConnected: true }).disabled).toBe(true);
    expect(getQueueHeaderCancelState({ currentStatus: null, isConnected: true }).disabled).toBe(true);
  });
});
