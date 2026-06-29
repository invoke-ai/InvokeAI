import { describe, expect, it, vi } from 'vitest';

import { getQueueMenuActions } from './queueMenuActions';

describe('getQueueMenuActions', () => {
  it('uses the same cancellation and processor actions for topbar and widget menus', () => {
    const actions = getQueueMenuActions({
      cancellableCount: 2,
      canManageProcessor: true,
      hasPendingQueueWork: true,
      hasRunningItem: true,
      isConnected: true,
      onCancelAll: vi.fn(),
      onCancelAllExceptCurrent: vi.fn(),
      onCancelCurrent: vi.fn(),
      onOpenQueue: vi.fn(),
      onPauseProcessor: vi.fn(),
      onResumeProcessor: vi.fn(),
    });

    expect(actions.map((action) => action.label)).toEqual([
      'Cancel Current Item',
      'Cancel All Items',
      'Cancel all except current item',
      'Resume Processor',
      'Pause Processor',
      'Open Queue',
    ]);
  });
});
