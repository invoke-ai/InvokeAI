import type { AuthSession } from '@features/identity';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it, vi } from 'vitest';

import { watchAccountActivations } from './archiveRecovery';

const createSession = (initial: Partial<AuthSession>) => {
  let snapshot = {
    accountEpoch: accountLifecycle.capture().epoch,
    multiuserEnabled: true,
    phase: 'ready',
    switchingAccount: false,
    user: null,
    ...initial,
  } as AuthSession;
  const listeners = new Set<() => void>();

  return {
    getSnapshot: () => snapshot,
    set: (patch: Partial<AuthSession>) => {
      snapshot = { ...snapshot, ...patch };
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

const user = { user_id: 'user-a' } as AuthSession['user'];

describe('watchAccountActivations', () => {
  it('starts once per account lifetime that reached the app', () => {
    accountLifecycle.invalidate();
    const session = createSession({});
    const start = vi.fn();
    const stop = watchAccountActivations(session, start);

    expect(start).not.toHaveBeenCalled(); // the sign-in screen

    const first = accountLifecycle.activate('user-a');
    session.set({ accountEpoch: first.epoch, switchingAccount: true, user });
    expect(start).not.toHaveBeenCalled();
    session.set({ switchingAccount: false });
    session.set({ user: { ...user! } });
    expect(start.mock.calls).toEqual([[first]]);

    const second = accountLifecycle.activate('user-a');
    session.set({ accountEpoch: second.epoch });
    expect(start.mock.calls).toEqual([[first], [second]]);

    stop();
  });

  it('starts for a single-user session already ready when watching begins', () => {
    const owner = accountLifecycle.activate('single-user');
    const start = vi.fn();

    watchAccountActivations(createSession({ accountEpoch: owner.epoch, multiuserEnabled: false }), start)();

    expect(start.mock.calls).toEqual([[owner]]);
  });
});
