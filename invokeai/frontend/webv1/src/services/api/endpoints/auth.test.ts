import { configureStore } from '@reduxjs/toolkit';
import type { BaseQueryApi } from '@reduxjs/toolkit/query';
import { sessionExpiredLogout, tokenRefreshed } from 'features/auth/store/authSlice';
import { beginPasswordChange, markTokenRefreshAccepted } from 'features/auth/store/authTokenRefresh';
import { authApi } from 'services/api/endpoints/auth';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { acceptRefreshedToken, api, buildV1Url, dynamicBaseQuery } from '..';

/**
 * `dynamicBaseQuery` reads the bearer token out of localStorage, and `getDeploymentBaseUrl`
 * reads `window.location.origin`. Neither exists in the default (node) test environment.
 */
const values = new Map<string, string>();

beforeEach(() => {
  values.clear();
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
  vi.stubGlobal('window', { location: { origin: 'http://localhost' } });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const buildStore = () =>
  configureStore({
    reducer: { [api.reducerPath]: api.reducer },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(api.middleware),
  });

const tokenFor = (nonce: number, epoch?: number) =>
  `header.${btoa(
    JSON.stringify({ user_id: 'user-1', nonce, ...(epoch === undefined ? {} : { token_epoch: epoch }) })
  )}.signature`;

describe('refreshed token acceptance', () => {
  it.each([
    ['profile update, old-token 401 first', false, 'auth/me', 'new_password'],
    ['profile update, replacement first', true, 'auth/me', 'new_password'],
    ['admin self-reset, old-token 401 first', false, 'auth/users/user-1', 'password'],
    ['admin self-reset, replacement first', true, 'auth/users/user-1', 'password'],
  ])('keeps a password-change replacement for %s', async (_, replacementFirst, passwordUrl, passwordField) => {
    const requestToken = tokenFor(1, 0);
    const replacement = tokenFor(2, 1);
    localStorage.setItem('auth_token', requestToken);

    let startPasswordChange!: () => void;
    const passwordChangeStarted = new Promise<void>((resolve) => {
      startPasswordChange = resolve;
    });
    let finishPasswordChange!: (response: Response) => void;
    const passwordChangeResponse = new Promise<Response>((resolve) => {
      finishPasswordChange = resolve;
    });
    let startStaleRequest!: () => void;
    const staleRequestStarted = new Promise<void>((resolve) => {
      startStaleRequest = resolve;
    });
    let finishStaleRequest!: (response: Response) => void;
    const staleResponse = new Promise<Response>((resolve) => {
      finishStaleRequest = resolve;
    });
    const dispatch = vi.fn((action: ReturnType<typeof sessionExpiredLogout> | ReturnType<typeof tokenRefreshed>) => {
      if (action.type === sessionExpiredLogout.type) {
        localStorage.removeItem('auth_token');
      } else if (action.type === tokenRefreshed.type) {
        localStorage.setItem('auth_token', action.payload);
      }
    });
    vi.stubGlobal(
      'fetch',
      vi.fn((input: string | URL | Request, init?: RequestInit) => {
        const url = input instanceof Request ? input.url : input.toString();
        if (url.endsWith(`/${passwordUrl}`)) {
          expect((input as Request).headers.get('Authorization')).toBe(`Bearer ${requestToken}`);
          startPasswordChange();
          return passwordChangeResponse;
        }
        if (url.includes('/images/i/example.png')) {
          expect((input as Request).headers.get('Authorization')).toBe(`Bearer ${requestToken}`);
          startStaleRequest();
          return staleResponse;
        }
        if (url.endsWith('/auth/media-cookie')) {
          expect(new Headers(init?.headers).get('Authorization')).toBe(`Bearer ${replacement}`);
          return Promise.resolve(new Response(null, { status: 204 }));
        }
        throw new Error(`Unexpected request: ${url}`);
      })
    );
    const baseQueryApi = {
      dispatch,
      getState: () => ({}),
      signal: new AbortController().signal,
      abort: () => {},
      endpoint: 'updateCurrentUser',
      type: 'mutation',
      forced: false,
      extra: undefined,
    } as unknown as BaseQueryApi;

    const passwordChange = dynamicBaseQuery(
      { url: buildV1Url(passwordUrl), method: 'PATCH', body: { [passwordField]: 'synthetic-test-password' } },
      baseQueryApi,
      {}
    );
    await passwordChangeStarted;
    const staleRequest = dynamicBaseQuery(buildV1Url('images/i/example.png'), baseQueryApi, {});
    await staleRequestStarted;
    const finish = () =>
      finishPasswordChange(
        new Response('{}', {
          headers: { 'content-type': 'application/json', 'X-Refreshed-Token': replacement },
        })
      );
    if (replacementFirst) {
      finish();
      await passwordChange;
      expect(dispatch).toHaveBeenCalledWith(tokenRefreshed(replacement));
      expect(localStorage.getItem('auth_token')).toBe(replacement);
    }
    finishStaleRequest(new Response(null, { status: 401 }));
    if (!replacementFirst) {
      await Promise.resolve();
      expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());
      finish();
      await passwordChange;
    }
    expect((await staleRequest).error?.status).toBe(401);

    expect(localStorage.getItem('auth_token')).toBe(replacement);
    expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());
  });

  it('logs out on an old-token 401 after a failed password change releases the transition', async () => {
    const oldToken = tokenFor(1, 0);
    localStorage.setItem('auth_token', oldToken);
    let finishPasswordChange!: (response: Response) => void;
    const passwordChangeResponse = new Promise<Response>((resolve) => {
      finishPasswordChange = resolve;
    });
    const dispatch = vi.fn((action: ReturnType<typeof sessionExpiredLogout>) => {
      if (action.type === sessionExpiredLogout.type) {
        localStorage.removeItem('auth_token');
      }
    });
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = input instanceof Request ? input.url : input.toString();
      if (url.endsWith('/auth/me')) {
        return passwordChangeResponse;
      }
      if (url.includes('/images/i/example.png')) {
        return Promise.resolve(new Response(null, { status: 401 }));
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const baseQueryApi = {
      dispatch,
      getState: () => ({}),
      signal: new AbortController().signal,
      abort: () => {},
      endpoint: 'updateCurrentUser',
      type: 'mutation',
      forced: false,
      extra: undefined,
    } as unknown as BaseQueryApi;

    const change = dynamicBaseQuery(
      { url: buildV1Url('auth/me'), method: 'PATCH', body: { new_password: 'synthetic-test-password' } },
      baseQueryApi,
      {}
    );
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const stale = dynamicBaseQuery(buildV1Url('images/i/example.png'), baseQueryApi, {});
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());

    finishPasswordChange(new Response(null, { status: 400 }));
    expect((await change).error?.status).toBe(400);
    expect((await stale).error?.status).toBe(401);
    expect(dispatch).toHaveBeenCalledWith(sessionExpiredLogout());
    expect(localStorage.getItem('auth_token')).toBeNull();
  });

  it('does not let this tab log out a password replacement being committed in another tab', async () => {
    const oldToken = tokenFor(1, 0);
    const replacement = tokenFor(2, 1);
    localStorage.setItem('auth_token', oldToken);
    // This tab did not send the password change. A second tab only shares storage with it.
    const finishInOtherTab = beginPasswordChange(oldToken);
    const dispatch = vi.fn((action: ReturnType<typeof sessionExpiredLogout>) => {
      if (action.type === sessionExpiredLogout.type) {
        localStorage.removeItem('auth_token');
      }
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(null, { status: 401 })))
    );
    const request = dynamicBaseQuery(
      buildV1Url('images/i/example.png'),
      {
        dispatch,
        getState: () => ({}),
        signal: new AbortController().signal,
        abort: () => {},
        endpoint: 'getImageDTO',
        type: 'query',
        forced: false,
        extra: undefined,
      } as unknown as BaseQueryApi,
      {}
    );

    await Promise.resolve();
    expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());
    localStorage.setItem('auth_token', replacement);
    finishInOtherTab();
    expect((await request).error?.status).toBe(401);
    expect(localStorage.getItem('auth_token')).toBe(replacement);
    expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());
  });

  it('preserves the shared replacement when the default frontend rotates a password before a legacy 401', async () => {
    vi.useFakeTimers();
    try {
      const oldToken = tokenFor(1, 0);
      const replacement = tokenFor(2, 1);
      localStorage.setItem('auth_token', oldToken);
      // The webv2 token adapter writes this contract, not the legacy beginPasswordChange helper.
      localStorage.setItem('auth_token_rotation:webv2-tab', JSON.stringify({ at: Date.now(), userId: 'user-1' }));
      const dispatch = vi.fn((action: ReturnType<typeof sessionExpiredLogout>) => {
        if (action.type === sessionExpiredLogout.type) {
          localStorage.removeItem('auth_token');
        }
      });
      vi.stubGlobal(
        'fetch',
        vi.fn(() => Promise.resolve(new Response(null, { status: 401 })))
      );
      const request = dynamicBaseQuery(
        buildV1Url('images/i/example.png'),
        {
          dispatch,
          getState: () => ({}),
          signal: new AbortController().signal,
          abort: () => {},
          endpoint: 'getImageDTO',
          type: 'query',
          forced: false,
          extra: undefined,
        } as unknown as BaseQueryApi,
        {}
      );

      // Advance after fetchBaseQuery has read the 401, so early logout cannot hide behind fetch's microtasks.
      await vi.advanceTimersByTimeAsync(100);
      expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());
      localStorage.setItem('auth_token', replacement);
      localStorage.removeItem('auth_token_rotation:webv2-tab');
      await vi.advanceTimersByTimeAsync(100);
      expect((await request).error?.status).toBe(401);
      expect(localStorage.getItem('auth_token')).toBe(replacement);
      expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());
    } finally {
      vi.useRealTimers();
    }
  });

  it('ends an expired session when no replacement is pending', async () => {
    const expiredToken = tokenFor(1, 0);
    localStorage.setItem('auth_token', expiredToken);
    const dispatch = vi.fn((action: ReturnType<typeof sessionExpiredLogout>) => {
      if (action.type === sessionExpiredLogout.type) {
        localStorage.removeItem('auth_token');
      }
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(null, { status: 401 })))
    );

    const result = await dynamicBaseQuery(
      buildV1Url('images/i/example.png'),
      {
        dispatch,
        getState: () => ({}),
        signal: new AbortController().signal,
        abort: () => {},
        endpoint: 'getImageDTO',
        type: 'query',
        forced: false,
        extra: undefined,
      } as unknown as BaseQueryApi,
      {}
    );

    expect(result.error?.status).toBe(401);
    expect(dispatch).toHaveBeenCalledWith(sessionExpiredLogout());
    expect(localStorage.getItem('auth_token')).toBeNull();
  });

  it.each([
    ['auth/me', 'new_password'],
    ['auth/users/user-1', 'password'],
  ])('does not discard a concurrent rotation when the legacy password PATCH to %s gets 401', async (url, field) => {
    vi.useFakeTimers();
    try {
      const oldToken = tokenFor(1, 0);
      const replacement = tokenFor(2, 1);
      const otherKey = 'auth_token_rotation:webv2-tab';
      localStorage.setItem('auth_token', oldToken);
      localStorage.setItem(otherKey, JSON.stringify({ at: Date.now(), userId: 'user-1' }));
      const dispatch = vi.fn((action: ReturnType<typeof sessionExpiredLogout>) => {
        if (action.type === sessionExpiredLogout.type) {
          localStorage.removeItem('auth_token');
        }
      });
      vi.stubGlobal(
        'fetch',
        vi.fn(() => Promise.resolve(new Response(null, { status: 401 })))
      );
      const request = dynamicBaseQuery(
        { url: buildV1Url(url), method: 'PATCH', body: { [field]: 'synthetic-test-password' } },
        {
          dispatch,
          getState: () => ({}),
          signal: new AbortController().signal,
          abort: () => {},
          endpoint: 'updateCurrentUser',
          type: 'mutation',
          forced: false,
          extra: undefined,
        } as unknown as BaseQueryApi,
        {}
      );
      await vi.advanceTimersByTimeAsync(100);
      expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());
      // Withdraw the rejected request's own marker, not the other tab's still-pending rotation.
      const keys = Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index));
      expect(keys.filter((key) => key?.startsWith('auth_token_rotation:'))).toEqual([otherKey]);
      localStorage.setItem('auth_token', replacement);
      localStorage.removeItem(otherKey);
      await vi.advanceTimersByTimeAsync(100);
      expect((await request).error?.status).toBe(401);
      expect(localStorage.getItem('auth_token')).toBe(replacement);
      expect(dispatch).not.toHaveBeenCalledWith(sessionExpiredLogout());
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not defer explicit logout behind another frontend rotation', async () => {
    vi.useFakeTimers();
    try {
      localStorage.setItem('auth_token', tokenFor(1, 0));
      localStorage.setItem('auth_token_rotation:other-tab', JSON.stringify({ at: Date.now(), userId: 'user-1' }));
      const dispatch = vi.fn((action: ReturnType<typeof sessionExpiredLogout>) => {
        if (action.type === sessionExpiredLogout.type) {
          localStorage.removeItem('auth_token');
        }
      });
      vi.stubGlobal(
        'fetch',
        vi.fn(() => Promise.resolve(new Response(null, { status: 401 })))
      );
      const request = dynamicBaseQuery(
        { url: buildV1Url('auth/logout'), method: 'POST' },
        {
          dispatch,
          getState: () => ({}),
          signal: new AbortController().signal,
          abort: () => {},
          endpoint: 'logout',
          type: 'mutation',
          forced: false,
          extra: undefined,
        } as unknown as BaseQueryApi,
        {}
      );
      await vi.advanceTimersByTimeAsync(100);
      expect(dispatch).toHaveBeenCalledWith(sessionExpiredLogout());
      expect(localStorage.getItem('auth_token')).toBeNull();
      expect((await request).error?.status).toBe(401);
    } finally {
      localStorage.clear();
      await vi.advanceTimersByTimeAsync(100);
      vi.useRealTimers();
    }
  });

  it.each([
    ['an explicit epoch-zero token', tokenFor(1, 0)],
    ['a legacy token without an epoch claim', tokenFor(1)],
  ])(
    'accepts an epoch-changing replacement for %s inside the routine refresh throttle window',
    async (_, requestToken) => {
      const refreshedToken = tokenFor(2, 1);
      localStorage.setItem('auth_token', requestToken);
      markTokenRefreshAccepted();

      const events: string[] = [];
      const dispatch = vi.fn(() => events.push('dispatch'));
      const fetchMock = vi.fn((input: string | URL | Request, init?: RequestInit) => {
        const url = input instanceof Request ? input.url : input.toString();
        if (url.endsWith('/api/v1/auth/media-cookie')) {
          events.push('media-cookie');
          expect(new Headers(init?.headers).get('Authorization')).toBe(`Bearer ${refreshedToken}`);
          return Promise.resolve(new Response(null, { status: 204 }));
        }
        return Promise.resolve(
          new Response('{}', {
            headers: { 'content-type': 'application/json', 'X-Refreshed-Token': refreshedToken },
          })
        );
      });
      vi.stubGlobal('fetch', fetchMock);

      await dynamicBaseQuery(
        buildV1Url('images/i/example.png'),
        {
          dispatch,
          getState: () => ({}),
          signal: new AbortController().signal,
          abort: () => {},
          endpoint: 'getImageDTO',
          type: 'query',
          forced: false,
          extra: undefined,
        } as unknown as BaseQueryApi,
        {}
      );

      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(dispatch).toHaveBeenCalledWith(tokenRefreshed(refreshedToken));
      expect(events).toEqual(['media-cookie', 'dispatch']);
    }
  );

  it('keeps a same-epoch replacement inside the routine refresh throttle window', async () => {
    const requestToken = tokenFor(1, 1);
    const refreshedToken = tokenFor(2, 1);
    localStorage.setItem('auth_token', requestToken);
    markTokenRefreshAccepted();

    const dispatch = vi.fn();
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        new Response('{}', {
          headers: { 'content-type': 'application/json', 'X-Refreshed-Token': refreshedToken },
        })
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    await dynamicBaseQuery(
      buildV1Url('images/i/example.png'),
      {
        dispatch,
        getState: () => ({}),
        signal: new AbortController().signal,
        abort: () => {},
        endpoint: 'getImageDTO',
        type: 'query',
        forced: false,
        extra: undefined,
      } as unknown as BaseQueryApi,
      {}
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('commits a same-epoch replacement after the routine refresh throttle window', async () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(300_000);
    const requestToken = tokenFor(1, 1);
    const refreshedToken = tokenFor(2, 1);
    localStorage.setItem('auth_token', requestToken);
    markTokenRefreshAccepted();
    now.mockReturnValue(360_001);

    const dispatch = vi.fn();
    const fetchMock = vi.fn((input: string | URL | Request, init?: RequestInit) => {
      const url = input instanceof Request ? input.url : input.toString();
      if (url.endsWith('/api/v1/auth/media-cookie')) {
        expect(new Headers(init?.headers).get('Authorization')).toBe(`Bearer ${refreshedToken}`);
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      return Promise.resolve(
        new Response('{}', {
          headers: { 'content-type': 'application/json', 'X-Refreshed-Token': refreshedToken },
        })
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    await dynamicBaseQuery(
      buildV1Url('images/i/example.png'),
      {
        dispatch,
        getState: () => ({}),
        signal: new AbortController().signal,
        abort: () => {},
        endpoint: 'getImageDTO',
        type: 'query',
        forced: false,
        extra: undefined,
      } as unknown as BaseQueryApi,
      {}
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(dispatch).toHaveBeenCalledWith(tokenRefreshed(refreshedToken));
  });

  describe('a password-change replacement racing a routine refresh of the same session', () => {
    // The request carried T0. A routine refresh T0' (same epoch) and the password change's
    // epoch-advancing replacement R were both minted for it, and T0' commits first.
    const requestToken = tokenFor(1, 0);
    const routineToken = tokenFor(2, 0);
    const replacementToken = tokenFor(3, 1);

    // Commits like the real `tokenRefreshed` reducer, so the second acceptance sees T0' stored.
    const dispatch = vi.fn((action: ReturnType<typeof tokenRefreshed>) => {
      localStorage.setItem('auth_token', action.payload);
    });

    beforeEach(() => {
      // Put the throttle past its window, so the routine refresh is accepted and commits.
      const now = vi.spyOn(Date, 'now').mockReturnValue(300_000);
      markTokenRefreshAccepted();
      now.mockReturnValue(360_001);
      dispatch.mockClear();
      localStorage.setItem('auth_token', requestToken);
      vi.stubGlobal(
        'fetch',
        vi.fn(() => Promise.resolve(new Response(null, { status: 204 })))
      );
    });

    it('keeps the replacement when the routine refresh committed before it arrived', async () => {
      await acceptRefreshedToken(routineToken, requestToken, 0, dispatch);
      expect(localStorage.getItem('auth_token')).toBe(routineToken);

      await acceptRefreshedToken(replacementToken, requestToken, 0, dispatch);

      expect(localStorage.getItem('auth_token')).toBe(replacementToken);
    });

    it('keeps the replacement when it queued on the lock behind the routine refresh', async () => {
      // Both pass the check before the lock while T0 is still stored; the replacement then waits
      // for the routine refresh to commit and re-checks inside the lock.
      const routine = acceptRefreshedToken(routineToken, requestToken, 0, dispatch);
      const replacement = acceptRefreshedToken(replacementToken, requestToken, 0, dispatch);
      await Promise.all([routine, replacement]);

      expect(dispatch.mock.calls.map(([action]) => action.payload)).toEqual([routineToken, replacementToken]);
      expect(localStorage.getItem('auth_token')).toBe(replacementToken);
    });
  });
});

describe('getCurrentUser', () => {
  it('does not let a replacement session read the 401 of the token it replaced', async () => {
    // The sequence this exists for: a tab page-loads with an expired token and asks who it is;
    // another tab logs the same user back in, so localStorage now holds a new token and this tab
    // adopts it — an adoption that deliberately keeps the API cache, since the user did not
    // change. The first request's 401 arrives in between. Shared across logins, one cache entry
    // would hand that 401 to the adopted session, and `ProtectedRoute` ends the session on a 401
    // from this query: the token the user just obtained would be deleted out of localStorage,
    // taking the tab that minted it down too.
    const requests: (string | null)[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn((request: Request) => {
        const sent = request.headers.get('Authorization');
        requests.push(sent);
        if (sent === 'Bearer token-expired') {
          return Promise.resolve(new Response(null, { status: 401 }));
        }
        return Promise.resolve(
          new Response(JSON.stringify({ user_id: 'user-1', email: 'user@example.com', is_admin: false }), {
            headers: { 'content-type': 'application/json' },
          })
        );
      })
    );
    const store = buildStore();

    localStorage.setItem('auth_token', 'token-expired');
    await store.dispatch(authApi.endpoints.getCurrentUser.initiate('token-expired'));
    expect(authApi.endpoints.getCurrentUser.select('token-expired')(store.getState()).error).toBeDefined();

    localStorage.setItem('auth_token', 'token-fresh');
    await store.dispatch(authApi.endpoints.getCurrentUser.initiate('token-fresh'));

    // A second request went out, under the new credential, and its result is what the adopted
    // session reads. The superseded entry keeps its own 401 and is no longer anybody's answer.
    expect(requests).toEqual(['Bearer token-expired', 'Bearer token-fresh']);
    const fresh = authApi.endpoints.getCurrentUser.select('token-fresh')(store.getState());
    expect(fresh.error).toBeUndefined();
    expect(fresh.data).toMatchObject({ user_id: 'user-1' });
    // And the superseded entry is untouched, so this is a separate answer rather than one
    // overwritten in place: an adopted session reads `token-fresh`'s and never sees the 401.
    expect(authApi.endpoints.getCurrentUser.select('token-expired')(store.getState()).error).toBeDefined();
  });
});
