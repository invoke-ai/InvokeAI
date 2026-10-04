import { describe, expect, it, vi } from 'vitest';

import { getQueueMenuActions } from './queueMenuActions';

describe('getQueueMenuActions', () => {
  const buildInputs = (includeOpenQueue: boolean) => ({
    labels: {
      cancelAll: 'Cancel All Items',
      cancelAllExceptCurrent: 'Cancel all except current item',
      cancelCurrent: 'Cancel Current Item',
      openQueue: 'Open Queue',
      pauseProcessor: 'Pause Processor',
      resumeProcessor: 'Resume Processor',
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
      'Cancel Current Item',
      'Cancel All Items',
      'Cancel all except current item',
      'Resume Processor',
      'Pause Processor',
      'Open Queue',
    ]);
  });

  it('omits the open-queue entry for menus rendered inside the queue widget', () => {
    const actions = getQueueMenuActions(buildInputs(false));

    expect(actions.map((action) => action.label)).not.toContain('Open Queue');
  });
});
