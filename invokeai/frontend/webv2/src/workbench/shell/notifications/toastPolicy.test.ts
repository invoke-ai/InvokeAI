import type { WorkbenchNotification } from '@workbench/projectContracts';

import { describe, expect, it } from 'vitest';

import { getToastKey, shouldToastNotification } from './toastPolicy';

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

describe('getToastKey', () => {
  it('re-keys coalesced run failures per occurrence but keeps ambient errors on one key', () => {
    expect(getToastKey(note({ category: 'run-outcome', id: 'n-1', occurrenceCount: 2 }))).toBe('n-1:2');
    expect(getToastKey(note({ category: 'run-outcome', id: 'n-1' }))).toBe('n-1:1');
    expect(getToastKey(note({ id: 'n-2', kind: 'error', occurrenceCount: 3 }))).toBe('n-2');
  });
});
