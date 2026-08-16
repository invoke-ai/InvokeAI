import type { WorkbenchNotification } from '@workbench/projectContracts';

import { describe, expect, it } from 'vitest';

import { shouldToastNotification } from './toastPolicy';

const note = (overrides: Partial<WorkbenchNotification>): WorkbenchNotification => ({
  createdAt: '2026-08-14T00:00:00.000Z',
  id: 'n-1',
  isRead: false,
  kind: 'success',
  title: 'Invocation queued',
  ...overrides,
});

describe('shouldToastNotification', () => {
  it('toasts enqueue notifications when the preference is on', () => {
    expect(shouldToastNotification(note({ category: 'enqueue' }), { notifyOnEnqueue: true })).toBe(true);
  });

  it('suppresses enqueue toasts when the preference is off', () => {
    expect(shouldToastNotification(note({ category: 'enqueue' }), { notifyOnEnqueue: false })).toBe(false);
  });

  it('always toasts uncategorized notifications', () => {
    expect(shouldToastNotification(note({ kind: 'error', title: 'Error' }), { notifyOnEnqueue: false })).toBe(true);
  });
});
