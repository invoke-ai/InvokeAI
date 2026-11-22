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
  it.each([
    ['toasts enqueue notifications when the preference is on', note({ category: 'enqueue' }), true, true],
    ['suppresses enqueue toasts when the preference is off', note({ category: 'enqueue' }), false, false],
    ['always toasts uncategorized notifications', note({ kind: 'error', title: 'Error' }), false, true],
  ])('%s', (_, notification, notifyOnEnqueue, expected) => {
    expect(shouldToastNotification(notification, { notifyOnEnqueue })).toBe(expected);
  });
});

describe('getToastKey', () => {
  it('re-keys coalesced run failures per occurrence but keeps ambient errors on one key', () => {
    expect(getToastKey(note({ category: 'run-outcome', id: 'n-1', occurrenceCount: 2 }))).toBe('n-1:2');
    expect(getToastKey(note({ category: 'run-outcome', id: 'n-1' }))).toBe('n-1:1');
    expect(getToastKey(note({ id: 'n-2', kind: 'error', occurrenceCount: 3 }))).toBe('n-2');
  });
});
