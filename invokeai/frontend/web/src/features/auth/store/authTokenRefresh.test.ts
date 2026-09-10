import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  beginAuthTransition,
  captureAuthGeneration,
  createMediaAuthLock,
  markTokenRefreshAccepted,
  runWithMediaAuthLock,
  shouldAcceptRefreshedToken,
  shouldEndSessionForUnauthorized,
  shouldThrottleRefreshedToken,
} from './authTokenRefresh';

const tokenFor = (userId: string, nonce: number, epoch: number) =>
  `header.${btoa(JSON.stringify({ user_id: userId, nonce, token_epoch: epoch }))}.signature`;

describe('refreshed token acceptance', () => {
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
});
