import { describe, expect, it, vi } from 'vitest';

import { getQueueMenuActions } from './queueMenuActions';

describe('getQueueMenuActions', () => {
  const buildInputs = (includeOpenQueue: boolean) => ({
    labels: {
      cancelAll: 'Cancel all items',
      cancelAllExceptCurrent: 'Cancel all except current item',
      cancelCurrent: 'Cancel current item',
      openQueue: 'Open queue',
      pauseProcessor: 'Pause processor',
      resumeProcessor: 'Resume processor',
    },
    cancellableCount: 2,
    canManageProcessor: true,
    hasPendingQueueWork: true,
    hasRunningItem: true,
    includeOpenQueue,
    isConnected: true,
    onCancelAll: vi.fn(),
    onCancelAllExceptCurrent: vi.fn(),
    onCancelCurrent: vi.fn(),
    onOpenQueue: vi.fn(),
    onPauseProcessor: vi.fn(),
    onResumeProcessor: vi.fn(),
  });

  it('uses the same cancellation and processor actions for topbar and widget menus', () => {
    const actions = getQueueMenuActions(buildInputs(true));

    expect(actions.map((action) => action.label)).toEqual([
      'Cancel current item',
      'Cancel all items',
      'Cancel all except current item',
      'Resume processor',
      'Pause processor',
      'Open queue',
    ]);
  });

  it('omits the open-queue entry for menus rendered inside the queue widget', () => {
    const actions = getQueueMenuActions(buildInputs(false));

    expect(actions.map((action) => action.label)).not.toContain('Open queue');
  });
});
