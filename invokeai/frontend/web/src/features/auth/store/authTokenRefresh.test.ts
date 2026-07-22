import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  beginAuthTransition,
  captureAuthGeneration,
  runWithMediaAuthLock,
  shouldAcceptRefreshedToken,
} from './authTokenRefresh';

describe('refreshed token acceptance', () => {
  beforeAll(() => {
    const values = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      clear: () => values.clear(),
      getItem: (key: string) => values.get(key) ?? null,
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
});
