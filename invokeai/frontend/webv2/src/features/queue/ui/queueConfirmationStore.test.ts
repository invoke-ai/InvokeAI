import { beforeEach, describe, expect, it, vi } from 'vitest';

import { clearQueueConfirmation, getQueueConfirmation, requestQueueConfirmation } from './queueConfirmationStore';

describe('queueConfirmationStore', () => {
  beforeEach(() => {
    clearQueueConfirmation();
  });

  it('keeps confirmation state outside the menu item lifecycle until explicitly cleared', () => {
    const onConfirm = vi.fn();

    requestQueueConfirmation({
      body: 'Clear queued work?',
      confirmLabel: 'Clear Queue',
      onConfirm,
      title: 'Clear queue?',
    });

    expect(getQueueConfirmation()).toMatchObject({
      body: 'Clear queued work?',
      confirmLabel: 'Clear Queue',
      onConfirm,
      title: 'Clear queue?',
    });
  });
});
