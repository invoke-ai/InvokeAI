import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  beginAuthTransition,
  beginPasswordChange,
  captureAuthGeneration,
  createMediaAuthLock,
  markTokenRefreshAccepted,
  runWithMediaAuthLock,
  shouldAcceptRefreshedToken,
  shouldEndSessionForUnauthorized,
  shouldThrottleRefreshedToken,
  waitForPasswordChange,
} from './authTokenRefresh';

const tokenFor = (userId: string, nonce: number, epoch: number) =>
  `header.${btoa(JSON.stringify({ user_id: userId, nonce, token_epoch: epoch }))}.signature`;

beforeAll(() => {
  const values = new Map<string, string>();
  vi.stubGlobal('localStorage', {
    clear: () => values.clear(),
    getItem: (key: string) => values.get(key) ?? null,
    key: (index: number) => [...values.keys()][index] ?? null,
    get length() {
      return values.size;
    },
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  });
});

describe('refreshed token acceptance', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('accepts a refresh for the unchanged authentication session', () => {
    localStorage.setItem('auth_token', 'token-a');
    const generation = captureAuthGeneration();

    expect(shouldAcceptRefreshedToken('token-a', generation)).toBe(true);
  });

  it('rejects a delayed refresh after logout or another login', () => {
    localStorage.setItem('auth_token', 'token-a');
    const generation = captureAuthGeneration();

    beginAuthTransition();
    localStorage.setItem('auth_token', 'token-b');

    expect(shouldAcceptRefreshedToken('token-a', generation)).toBe(false);
  });

  it('rejects a response superseded by a newer refresh', () => {
    localStorage.setItem('auth_token', 'token-a');
    const generation = captureAuthGeneration();
    localStorage.setItem('auth_token', 'token-newer');

    expect(shouldAcceptRefreshedToken('token-a', generation)).toBe(false);
  });

  it('accepts a password-change replacement after a routine refresh of the same session committed first', () => {
    // The request carried T0; a routine refresh T0' (same user and epoch) committed before the
    // password change's epoch-advancing replacement R arrived.
    const requestToken = tokenFor('user', 1, 0);
    localStorage.setItem('auth_token', requestToken);
    const generation = captureAuthGeneration();
    localStorage.setItem('auth_token', tokenFor('user', 2, 0));

    expect(shouldAcceptRefreshedToken(requestToken, generation, tokenFor('user', 3, 1))).toBe(true);
  });

  it('still rejects a routine refresh superseded by a newer refresh of the same session', () => {
    const requestToken = tokenFor('user', 1, 0);
    localStorage.setItem('auth_token', requestToken);
    const generation = captureAuthGeneration();
    localStorage.setItem('auth_token', tokenFor('user', 2, 0));

    expect(shouldAcceptRefreshedToken(requestToken, generation, tokenFor('user', 3, 0))).toBe(false);
  });

  it('rejects an epoch replacement once the stored token is no longer the same session', () => {
    const requestToken = tokenFor('user', 1, 0);
    localStorage.setItem('auth_token', requestToken);
    const generation = captureAuthGeneration();
    const replacement = tokenFor('user', 3, 1);

    // Already moved past this epoch (e.g. a newer replacement committed first).
    localStorage.setItem('auth_token', tokenFor('user', 4, 2));
    expect(shouldAcceptRefreshedToken(requestToken, generation, replacement)).toBe(false);

    // Another user's token.
    localStorage.setItem('auth_token', tokenFor('other', 5, 0));
    expect(shouldAcceptRefreshedToken(requestToken, generation, replacement)).toBe(false);

    // Logged out.
    localStorage.removeItem('auth_token');
    expect(shouldAcceptRefreshedToken(requestToken, generation, replacement)).toBe(false);
  });

  it('rejects an epoch replacement after a login transition started', () => {
    const requestToken = tokenFor('user', 1, 0);
    localStorage.setItem('auth_token', requestToken);
    const generation = captureAuthGeneration();
    beginAuthTransition();
    localStorage.setItem('auth_token', tokenFor('user', 2, 0));

    expect(shouldAcceptRefreshedToken(requestToken, generation, tokenFor('user', 3, 1))).toBe(false);
  });

  it('recovers from a malformed stored generation', () => {
    localStorage.setItem('auth_generation', 'not-a-number');

    expect(captureAuthGeneration()).toBe(0);
    expect(beginAuthTransition()).toBe(1);
  });

  it('does not throttle the replacement token that advances the current user revocation epoch', () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(100_000);
    markTokenRefreshAccepted();

    expect(shouldThrottleRefreshedToken(tokenFor('user', 1, 0), tokenFor('user', 2, 1))).toBe(false);

    now.mockRestore();
  });

  it('keeps routine, cross-user, and unreadable replacements throttled', () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(200_000);
    markTokenRefreshAccepted();

    expect(shouldThrottleRefreshedToken(tokenFor('user', 1, 1), tokenFor('user', 2, 1))).toBe(true);
    expect(shouldThrottleRefreshedToken(tokenFor('user-a', 1, 0), tokenFor('user-b', 2, 1))).toBe(true);
    expect(shouldThrottleRefreshedToken('opaque-old', 'opaque-new')).toBe(true);

    now.mockRestore();
  });

  it('accepts a routine same-epoch replacement after the throttle window', () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(300_000);
    markTokenRefreshAccepted();
    now.mockReturnValue(360_001);

    expect(shouldThrottleRefreshedToken(tokenFor('user', 1, 1), tokenFor('user', 2, 1))).toBe(false);

    now.mockRestore();
  });

  it('serializes media-cookie writes', async () => {
    const calls: string[] = [];
    let releaseFirst: (() => void) | undefined;
    const first = runWithMediaAuthLock(
      () =>
        new Promise<void>((resolve) => {
          calls.push('first-start');
          releaseFirst = () => {
            calls.push('first-end');
            resolve();
          };
        })
    );
    const second = runWithMediaAuthLock(() => {
      calls.push('second');
    });

    await Promise.resolve();
    expect(calls).toEqual(['first-start']);
    releaseFirst?.();
    await Promise.all([first, second]);
    expect(calls).toEqual(['first-start', 'first-end', 'second']);
  });

  it('serializes fallback media-cookie writes across tabs', async () => {
    vi.stubGlobal('navigator', {});
    const firstTabLock = createMediaAuthLock('tab-a');
    const secondTabLock = createMediaAuthLock('tab-b');
    const calls: string[] = [];
    let releaseFirst: (() => void) | undefined;

    const first = firstTabLock(
      () =>
        new Promise<void>((resolve) => {
          calls.push('first-start');
          releaseFirst = () => {
            calls.push('first-end');
            resolve();
          };
        })
    );
    const second = secondTabLock(() => {
      calls.push('second');
    });

    await vi.waitFor(() => expect(calls).toEqual(['first-start']));
    releaseFirst?.();
    await Promise.all([first, second]);
    expect(calls).toEqual(['first-start', 'first-end', 'second']);
  });

  it('releases the fallback media lock when a write fails', async () => {
    const lock = createMediaAuthLock('tab-a');

    await expect(lock(() => Promise.reject(new Error('write failed')))).rejects.toThrow('write failed');
    await expect(lock(() => 'next write')).resolves.toBe('next write');
  });
});

describe('ending a session over a 401', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('ends it when the token that got the 401 is still the live one', () => {
    localStorage.setItem('auth_token', 'token-a');

    expect(shouldEndSessionForUnauthorized('token-a')).toBe(true);
  });

  it('spares the session that replaced the one the 401 belongs to', () => {
    // Someone else took the tab over while the request was in flight -- here, or in another
    // tab, since localStorage is shared. Their token must survive a stranger's 401.
    localStorage.setItem('auth_token', 'token-b');

    expect(shouldEndSessionForUnauthorized('token-a')).toBe(false);
  });

  it('spares a session whose token a sliding-window refresh replaced', () => {
    // Byte equality, not `isSameAuthContext`: the refreshed token is the same login, but a 401
    // for the token it replaced says nothing about it. The next request settles the question.
    localStorage.setItem('auth_token', 'token-a-refreshed');

    expect(shouldEndSessionForUnauthorized('token-a')).toBe(false);
  });

  it('ignores a 401 for a request that carried no credential', () => {
    // Both forms `dynamicBaseQuery` can hold: no token at all, and the empty string, which
    // sets no Authorization header yet compares equal to itself once stored.
    expect(shouldEndSessionForUnauthorized(null)).toBe(false);
    localStorage.setItem('auth_token', '');
    expect(shouldEndSessionForUnauthorized('')).toBe(false);
  });

  it('waits for a password change published through shared storage, then rechecks the live token', async () => {
    vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout'] });
    try {
      const oldToken = tokenFor('user', 1, 0);
      const replacement = tokenFor('user', 2, 1);
      localStorage.setItem('auth_token', oldToken);
      const finishInOtherTab = beginPasswordChange(oldToken);
      const pending = waitForPasswordChange(oldToken, captureAuthGeneration());
      let settled = false;
      void pending.then(() => {
        settled = true;
      });

      await vi.advanceTimersByTimeAsync(100);
      expect(settled).toBe(false);
      localStorage.setItem('auth_token', replacement);
      finishInOtherTab();
      await vi.advanceTimersByTimeAsync(100);

      await pending;
      expect(shouldEndSessionForUnauthorized(oldToken)).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('announces a legacy password change using the default frontend rotation protocol', () => {
    const finish = beginPasswordChange(tokenFor('user', 1, 0));
    try {
      const keys = Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index));
      const key = keys.find((value) => value?.startsWith('auth_token_rotation:'));
      expect(key).toBeDefined();
      expect(JSON.parse(localStorage.getItem(key!)!)).toEqual({ at: expect.any(Number), userId: 'user' });
    } finally {
      finish();
    }
  });

  it('waits for a default-frontend rotation without depending on the legacy auth generation', async () => {
    vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout'] });
    try {
      vi.setSystemTime(100_000);
      const oldToken = tokenFor('user', 1, 0);
      localStorage.setItem('auth_token', oldToken);
      localStorage.setItem('auth_generation', '7');
      localStorage.setItem('auth_token_rotation:webv2-tab', JSON.stringify({ at: 100_000, userId: 'user' }));
      let settled = false;
      const pending = waitForPasswordChange(oldToken, captureAuthGeneration()).then(() => {
        settled = true;
      });

      await vi.advanceTimersByTimeAsync(100);
      expect(settled).toBe(false);
      localStorage.setItem('auth_token', tokenFor('user', 2, 1));
      localStorage.removeItem('auth_token_rotation:webv2-tab');
      await vi.advanceTimersByTimeAsync(100);
      await pending;
      expect(shouldEndSessionForUnauthorized(oldToken)).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('withdraws only its own rotation when concurrent tabs are changing the same password', () => {
    const otherKey = 'auth_token_rotation:webv2-tab';
    const otherMarker = JSON.stringify({ at: Date.now(), userId: 'user' });
    localStorage.setItem(otherKey, otherMarker);
    const finish = beginPasswordChange(tokenFor('user', 1, 0));
    expect(localStorage.length).toBe(2);
    finish();
    expect(localStorage.length).toBe(1);
    expect(localStorage.getItem(otherKey)).toBe(otherMarker);
  });

  it.each([100_000, 1_000_000])(
    'bounds a default-frontend rotation with timestamp %s to thirty seconds',
    async (at) => {
      vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout'] });
      try {
        vi.setSystemTime(100_000);
        const oldToken = tokenFor('user', 1, 0);
        localStorage.setItem('auth_token', oldToken);
        localStorage.setItem('auth_token_rotation:abandoned', JSON.stringify({ at, userId: 'user' }));
        let settled = false;
        const pending = waitForPasswordChange(oldToken, captureAuthGeneration()).then(() => {
          settled = true;
        });
        await vi.advanceTimersByTimeAsync(29_900);
        expect(settled).toBe(false);
        await vi.advanceTimersByTimeAsync(100);
        await pending;
        expect(shouldEndSessionForUnauthorized(oldToken)).toBe(true);
      } finally {
        vi.useRealTimers();
      }
    }
  );

  it('keeps the rotation wait bounded when the system clock moves backwards', async () => {
    vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout'] });
    try {
      vi.setSystemTime(100_000);
      const oldToken = tokenFor('user', 1, 0);
      localStorage.setItem('auth_token', oldToken);
      localStorage.setItem('auth_token_rotation:abandoned', JSON.stringify({ at: Date.now(), userId: 'user' }));
      let settled = false;
      const pending = waitForPasswordChange(oldToken, captureAuthGeneration()).then(() => {
        settled = true;
      });
      await vi.advanceTimersByTimeAsync(15_000);
      vi.setSystemTime(55_000);
      await vi.advanceTimersByTimeAsync(14_900);
      expect(settled).toBe(false);
      await vi.advanceTimersByTimeAsync(100);
      expect(settled).toBe(true);
      await pending;
      expect(shouldEndSessionForUnauthorized(oldToken)).toBe(true);
    } finally {
      localStorage.clear();
      await vi.advanceTimersByTimeAsync(100);
      vi.useRealTimers();
    }
  });

  it.each([
    JSON.stringify({ at: 69_999, userId: 'user' }),
    JSON.stringify({ at: 100_000, userId: 'someone-else' }),
    JSON.stringify({ at: '100000', userId: 'user' }),
    'null',
    'broken',
  ])('does not defer session expiry for an unrelated, stale or malformed rotation: %s', async (marker) => {
    vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout'] });
    try {
      vi.setSystemTime(100_000);
      const token = tokenFor('user', 1, 0);
      localStorage.setItem('auth_token', token);
      localStorage.setItem('auth_token_rotation:ignored', marker);
      await waitForPasswordChange(token, captureAuthGeneration());
      expect(shouldEndSessionForUnauthorized(token)).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it('stops waiting for an abandoned transition when its marker expires', async () => {
    vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout'] });
    try {
      vi.setSystemTime(100_000);
      const oldToken = tokenFor('user', 1, 0);
      localStorage.setItem('auth_token', oldToken);
      const finish = beginPasswordChange(oldToken);
      const pending = waitForPasswordChange(oldToken, captureAuthGeneration());

      vi.setSystemTime(130_001);
      await vi.advanceTimersByTimeAsync(100);
      await pending;
      expect(shouldEndSessionForUnauthorized(oldToken)).toBe(true);
      finish();
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not wait for a transition belonging to a different login', async () => {
    const otherToken = tokenFor('other-user', 1, 0);
    const currentToken = tokenFor('user', 1, 0);
    const finishOther = beginPasswordChange(otherToken);
    localStorage.setItem('auth_token', currentToken);

    await waitForPasswordChange(currentToken, captureAuthGeneration());
    expect(shouldEndSessionForUnauthorized(currentToken)).toBe(true);
    finishOther();
  });

  it('stops waiting on explicit logout and rejects a late password-change token', async () => {
    vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout'] });
    try {
      const oldToken = tokenFor('user', 1, 0);
      localStorage.setItem('auth_token', oldToken);
      const generation = captureAuthGeneration();
      const finish = beginPasswordChange(oldToken);
      const pending = waitForPasswordChange(oldToken, generation);

      beginAuthTransition();
      localStorage.removeItem('auth_token');
      await vi.advanceTimersByTimeAsync(100);
      await pending;
      expect(shouldAcceptRefreshedToken(oldToken, generation, tokenFor('user', 2, 1))).toBe(false);
      finish();
    } finally {
      vi.useRealTimers();
    }
  });
});
