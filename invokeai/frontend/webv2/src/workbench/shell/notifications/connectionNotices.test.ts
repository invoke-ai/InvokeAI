import type { BackendConnectionStatus } from '@platform/transport/types';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { startConnectionNotices } from './connectionNotices';

const setup = (initial: BackendConnectionStatus) => {
  let status = initial;
  const listeners = new Set<() => void>();
  const notices: string[] = [];
  const stop = startConnectionNotices({
    delayMs: 1_000,
    getStatus: () => status,
    notifications: { add: (notice) => notices.push(notice.titleKey ?? notice.title) },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  });
  const set = (next: BackendConnectionStatus) => {
    status = next;
    listeners.forEach((listener) => listener());
  };

  return { notices, set, stop };
};

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('connection notices', () => {
  it('announces an outage once it lasts, and its end once', () => {
    const { notices, set } = setup('connected');

    set('disconnected');
    vi.advanceTimersByTime(999);
    expect(notices).toEqual([]);
    vi.advanceTimersByTime(1);
    set('connecting');
    set('disconnected');
    vi.advanceTimersByTime(5_000);
    expect(notices).toEqual(['notifications.connection.lost']);

    set('connected');
    expect(notices).toEqual(['notifications.connection.lost', 'notifications.connection.restored']);
  });

  it('stays quiet through a brief dropout', () => {
    const { notices, set } = setup('connected');

    for (let flap = 0; flap < 5; flap += 1) {
      set('disconnected');
      vi.advanceTimersByTime(500);
      set('connected');
    }
    vi.advanceTimersByTime(5_000);

    expect(notices).toEqual([]);
  });

  it('does not report the first connection of a session as a reconnect', () => {
    const { notices, set } = setup('connecting');

    vi.advanceTimersByTime(5_000);
    set('connected');

    expect(notices).toEqual([]);
  });

  it('drops a pending announcement when stopped', () => {
    const { notices, set, stop } = setup('connected');

    set('disconnected');
    stop();
    vi.advanceTimersByTime(5_000);

    expect(notices).toEqual([]);
  });
});
