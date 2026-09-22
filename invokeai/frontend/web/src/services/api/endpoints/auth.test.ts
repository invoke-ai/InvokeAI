import { configureStore } from '@reduxjs/toolkit';
import { authApi } from 'services/api/endpoints/auth';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '..';

/**
 * `dynamicBaseQuery` reads the bearer token out of localStorage, and `getDeploymentBaseUrl`
 * reads `window.location.origin`. Neither exists in the default (node) test environment.
 */
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
  vi.stubGlobal('window', { location: { origin: 'http://localhost' } });
});

beforeEach(() => {
  localStorage.clear();
});

const buildStore = () =>
  configureStore({
    reducer: { [api.reducerPath]: api.reducer },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(api.middleware),
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
