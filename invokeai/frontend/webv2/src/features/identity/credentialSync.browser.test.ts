import type * as httpModule from '@platform/transport/http';

import { createAccountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import type * as sessionModule from './session';

/**
 * Two documents sharing real browser storage: this page is the tab under test, and a same-origin frame stands in for
 * a second tab. The frame's writes reach this page as genuine `storage` events.
 */

const TOKEN_KEY = 'auth_token';
const ROTATION_KEY = 'auth_token_rotation';

const userFor = (userId: string) => ({
  created_at: '2026-07-25T12:00:00Z',
  display_name: userId,
  email: `${userId}@example.com`,
  is_active: true,
  is_admin: false,
  last_login_at: null,
  updated_at: '2026-07-25T12:00:00Z',
  user_id: userId,
});

/** An unsigned token carrying the backend's `user_id` claim; the stub backend reads the same claim. */
const tokenFor = (userId: string, nonce: string): string =>
  `e30.${btoa(JSON.stringify({ nonce, user_id: userId })).replaceAll('=', '')}.signature`;

const userIdOf = (authorization: string | null): string | null =>
  authorization === null
    ? null
    : (JSON.parse(atob(authorization.replace('Bearer ', '').split('.')[1]!)) as { user_id: string }).user_id;

interface SentRequest {
  authorization: string | null;
  method: string;
  path: string;
}

const json = (body: unknown, init: { refreshedToken?: string; status?: number } = {}): Response =>
  new Response(JSON.stringify(body), {
    headers: init.refreshedToken ? { 'X-Refreshed-Token': init.refreshedToken } : undefined,
    status: init.status ?? 200,
  });

let http: typeof httpModule;
let session: typeof sessionModule;
let sent: SentRequest[];
let pendingWrite: ((response: Response) => void) | null;
let stopSync: () => void;
let otherTab: HTMLIFrameElement;

/** The backend authenticates each request by its bearer token, as the real one does. */
const backend = (request: SentRequest): Response | Promise<Response> => {
  const principal = userIdOf(request.authorization);

  switch (request.path) {
    case '/api/v1/auth/status':
      return json({
        admin_email: null,
        multiuser_enabled: true,
        setup_required: false,
        strict_password_checking: false,
      });
    case '/api/v1/auth/login':
      return json({ expires_in: 86400, token: tokenFor('user-a', 'login'), user: userFor('user-a') });
    case '/api/v1/auth/me':
      return principal ? json(userFor(principal)) : json({ detail: 'Not authenticated' }, { status: 401 });
    case '/api/v1/boards/revoked':
      return json({ detail: 'revoked' }, { status: 401 });
    case '/api/v1/client_state/default/set_by_key':
      if (pendingWrite === null) {
        return json({});
      }
      return new Promise((resolve) => {
        pendingWrite = resolve;
      });
    default:
      return json({});
  }
};

const writeClientState = (): Promise<Response> =>
  http.apiFetch('/api/v1/client_state/default/set_by_key', { body: '{}', method: 'POST' });

beforeEach(async () => {
  vi.resetModules();
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(ROTATION_KEY);
  sent = [];
  pendingWrite = null;
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    const request = {
      authorization: new Headers(init?.headers).get('Authorization'),
      method: init?.method ?? 'GET',
      path: new URL(String(input)).pathname,
    };

    sent.push(request);

    return Promise.resolve(backend(request));
  });
  otherTab = document.createElement('iframe');
  document.body.append(otherTab);

  [http, session] = await Promise.all([import('@platform/transport/http'), import('./session')]);
  session.configureIdentityAccountLifecycle(createAccountLifecycle());
  http.configureHttpAuth(session.identityTransportAuthAdapter);
  stopSync = session.startIdentityCredentialSync();
  await session.ensureAuthSession();
  await session.loginWithCredentials('user-a@example.com', 'password', false);
});

afterEach(() => {
  stopSync();
  otherTab.remove();
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(ROTATION_KEY);
  vi.unstubAllGlobals();
});

const otherTabStorage = (): Storage => otherTab.contentWindow!.localStorage;

/** Storage events cross documents asynchronously; allow for a loaded CI machine. */
const CROSS_DOCUMENT = { timeout: 5_000 };

it.each([
  {
    name: 'signs out, then signs in as another user',
    switchAccount: async () => {
      otherTabStorage().removeItem(TOKEN_KEY);
      await vi.waitFor(() => {
        expect(session.getAuthSession().user).toBeNull();
      }, CROSS_DOCUMENT);
      otherTabStorage().setItem(TOKEN_KEY, tokenFor('user-b', 'login'));
    },
  },
  {
    name: 'replaces the stored token with another user’s',
    switchAccount: () => {
      otherTabStorage().setItem(TOKEN_KEY, tokenFor('user-b', 'login'));
      return Promise.resolve();
    },
  },
])('keeps user, storage scope and bearer token in agreement when another tab $name', async ({ switchAccount }) => {
  expect(session.getUserStorageScope()).toBe(':user:user-a');

  // A write for account A is in flight when the other tab switches accounts.
  pendingWrite = () => undefined;
  const writeForA = writeClientState();
  await vi.waitFor(() => {
    expect(sent.at(-1)?.path).toBe('/api/v1/client_state/default/set_by_key');
  });
  const sentBeforeSwitch = sent.length;

  await switchAccount();

  await vi.waitFor(() => {
    expect(session.getAuthSession().user?.user_id).toBe('user-b');
  }, CROSS_DOCUMENT);
  // A's late response, renewal included, belongs to a lifetime that has ended.
  pendingWrite?.(json({}, { refreshedToken: tokenFor('user-a', 'renewed') }));
  await expect(writeForA).rejects.toBeInstanceOf(http.HttpRequestIdentityExpiredError);

  pendingWrite = null;
  await writeClientState();

  expect(session.getUserStorageScope()).toBe(':user:user-b');
  expect(userIdOf(sent.at(-1)!.authorization)).toBe('user-b');
  // No request carried A's token once the other tab switched accounts.
  expect(sent.slice(sentBeforeSwitch).map((request) => userIdOf(request.authorization))).not.toContain('user-a');
  expect(window.localStorage.getItem(TOKEN_KEY)).toBe(tokenFor('user-b', 'login'));
});

it('signs out when another tab signs out, without reporting an expired session', async () => {
  otherTabStorage().removeItem(TOKEN_KEY);

  await vi.waitFor(() => {
    expect(session.getAuthSession().user).toBeNull();
  }, CROSS_DOCUMENT);
  expect(session.getAuthSession().sessionExpired).toBe(false);
  expect(session.identityTransportAuthAdapter.capture().token).toBeNull();
});

it('adopts another tab’s renewal of the same user without starting a new lifetime', async () => {
  const { accountEpoch } = session.getAuthSession();
  const renewed = tokenFor('user-a', 'renewed');

  otherTabStorage().setItem(TOKEN_KEY, renewed);

  await vi.waitFor(() => {
    expect(session.identityTransportAuthAdapter.capture().token).toBe(renewed);
  }, CROSS_DOCUMENT);
  await writeClientState();

  expect(session.getAuthSession()).toMatchObject({ accountEpoch, user: { user_id: 'user-a' } });
  expect(sent.at(-1)!.authorization).toBe(`Bearer ${renewed}`);
  expect(sent.some((request) => request.path === '/api/v1/auth/me')).toBe(false);
});

it('holds a 401 while another tab announces a password change, then adopts the replacement it stores', async () => {
  const { accountEpoch } = session.getAuthSession();
  const replacement = tokenFor('user-a', 'epoch-2');

  otherTabStorage().setItem(ROTATION_KEY, JSON.stringify({ at: Date.now(), userId: 'user-a' }));
  await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });

  expect(session.getAuthSession()).toMatchObject({ accountEpoch, sessionExpired: false, user: { user_id: 'user-a' } });

  otherTabStorage().setItem(TOKEN_KEY, replacement);
  otherTabStorage().removeItem(ROTATION_KEY);
  await vi.waitFor(() => {
    expect(session.identityTransportAuthAdapter.capture().token).toBe(replacement);
  }, CROSS_DOCUMENT);
  await writeClientState();

  expect(session.getAuthSession()).toMatchObject({ accountEpoch, sessionExpired: false, user: { user_id: 'user-a' } });
  expect(sent.at(-1)!.authorization).toBe(`Bearer ${replacement}`);
});

it('expires the held 401 once the announcing tab withdraws without a replacement', async () => {
  otherTabStorage().setItem(ROTATION_KEY, JSON.stringify({ at: Date.now(), userId: 'user-a' }));
  await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
  expect(session.getAuthSession().user?.user_id).toBe('user-a');

  otherTabStorage().removeItem(ROTATION_KEY);

  await vi.waitFor(() => {
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });
  }, CROSS_DOCUMENT);
  expect(window.localStorage.getItem(TOKEN_KEY)).toBeNull();
});
