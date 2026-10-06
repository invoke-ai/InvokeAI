import type * as httpModule from '@platform/transport/http';

import { createAccountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type * as sessionModule from './session';

/**
 * The credential contract between the real HTTP transport and Identity, against a stubbed backend. Only browser
 * storage is replaced, by an in-memory store.
 */

const storage = vi.hoisted(() => {
  const listeners = new Set<() => void>();
  let stored: string | null = null;
  let rotation: { at: number; userId: string } | null = null;
  const log: string[] = [];
  const notify = (): void => {
    for (const listener of listeners) {
      listener();
    }
  };

  return {
    adapter: {
      clear: () => {
        stored = null;
      },
      clearRotation: () => {
        log.push('clearRotation');
        rotation = null;
      },
      read: () => stored,
      readRotation: () => rotation,
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      write: (token: string) => {
        log.push(`write:${token}`);
        stored = token;
      },
      writeRotation: (marker: { at: number; userId: string }) => {
        log.push('writeRotation');
        rotation = marker;
      },
    },
    get: () => stored,
    getRotation: () => rotation,
    /** This tab's own writes, in order. */
    log,
    /** Another tab announced or withdrew a credential rotation; this tab hears about it through a storage event. */
    otherTabAnnouncesRotation: (marker: { at: number; userId: string } | null) => {
      rotation = marker;
      notify();
    },
    /** Another tab changed the shared token; this tab hears about it through a storage event. */
    otherTabStores: (token: string | null) => {
      stored = token;
      notify();
    },
    reset: () => {
      listeners.clear();
      log.length = 0;
      stored = null;
      rotation = null;
    },
    /** A token left by an earlier page load. */
    seed: (token: string) => {
      stored = token;
    },
  };
});

vi.mock('./core/tokenStorage', () => ({ browserIdentityTokenAdapter: storage.adapter }));

const user = {
  created_at: '2026-07-25T12:00:00Z',
  display_name: 'Ada',
  email: 'ada@example.com',
  is_active: true,
  is_admin: false,
  last_login_at: '2026-07-25T12:00:00Z',
  updated_at: '2026-07-25T12:00:00Z',
  user_id: 'user-a',
};

const userB = { ...user, display_name: 'Grace', email: 'grace@example.com', user_id: 'user-b' };

/** An unsigned token carrying the backend's `user_id` claim, which is all the client reads. */
const tokenFor = (userId: string, nonce: string): string =>
  `e30.${btoa(JSON.stringify({ nonce, user_id: userId })).replaceAll('=', '')}.signature`;

interface SentRequest {
  authorization: string | null;
  method: string;
  path: string;
  signal: AbortSignal | null;
}

const json = (body: unknown, init: { refreshedToken?: string; status?: number } = {}): Response =>
  new Response(JSON.stringify(body), {
    headers: init.refreshedToken ? { 'X-Refreshed-Token': init.refreshedToken } : undefined,
    status: init.status ?? 200,
  });

const authRoutes = (request: SentRequest): Response => {
  switch (request.path) {
    case '/api/v1/auth/status':
      return json({
        admin_email: null,
        multiuser_enabled: true,
        setup_required: false,
        strict_password_checking: false,
      });
    case '/api/v1/auth/login':
      return json({ expires_in: 86400, token: 'token-a', user });
    case '/api/v1/auth/media-cookie':
      return json({ success: true });
    case '/api/v1/auth/me':
      if (request.method !== 'GET') {
        return json({});
      }
      if (request.authorization === null) {
        return json({ detail: 'Not authenticated' }, { status: 401 });
      }
      return json(request.authorization.includes(tokenFor(userB.user_id, 'login')) ? userB : user);
    default:
      return json({});
  }
};

const createDeferredResponse = (): { promise: Promise<Response>; resolve: (response: Response) => void } => {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

let http: typeof httpModule;
let session: typeof sessionModule;
let sent: SentRequest[];
let route: (request: SentRequest) => Response | Promise<Response>;
let accountCacheClears: number;
let stopSync: () => void;
let page: EventTarget & { visibilityState: DocumentVisibilityState };

/** The bearer token the next ordinary request carries. */
const nextRequestToken = async (): Promise<string | null> => {
  await http.apiFetch('/api/v1/boards/');

  return sent.at(-1)?.authorization?.replace('Bearer ', '') ?? null;
};

const signIn = async (): Promise<void> => {
  await session.ensureAuthSession();
  await session.loginWithCredentials(user.email, 'password', false);
  accountCacheClears = 0;
};

const sentTo = (path: string, method = 'GET'): SentRequest[] =>
  sent.filter((request) => request.path === path && request.method === method);

const untilPatchSent = (path: string): Promise<void> =>
  vi.waitFor(() => {
    expect(sentTo(path, 'PATCH')).toHaveLength(1);
  });

beforeEach(async () => {
  vi.resetModules();
  storage.reset();
  sent = [];
  route = authRoutes;
  accountCacheClears = 0;
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    const request = {
      authorization: new Headers(init?.headers).get('Authorization'),
      method: init?.method ?? 'GET',
      path: new URL(String(input)).pathname,
      signal: init?.signal ?? null,
    };

    sent.push(request);

    return Promise.resolve(route(request));
  });
  vi.stubGlobal('window', Object.assign(new EventTarget(), { location: { origin: 'http://localhost' } }));
  page = Object.assign(new EventTarget(), { visibilityState: 'visible' as DocumentVisibilityState });
  vi.stubGlobal('document', page);

  [http, session] = await Promise.all([import('@platform/transport/http'), import('./session')]);
  const lifecycle = createAccountLifecycle();

  lifecycle.register({
    clear: () => {
      accountCacheClears += 1;
    },
    name: 'account-cache',
  });
  session.configureIdentityAccountLifecycle(lifecycle);
  http.configureHttpAuth(session.identityTransportAuthAdapter);
  stopSync = session.startIdentityCredentialSync();
});

afterEach(() => {
  stopSync();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('credential renewal', () => {
  beforeEach(signIn);

  it('sends a renewal on the next request and persists it without starting a new lifetime', async () => {
    const { accountEpoch } = session.getAuthSession();
    route = (request) =>
      request.path === '/api/v1/images/upload' ? json({}, { refreshedToken: 'token-a2' }) : authRoutes(request);

    await http.apiFetchRaw('/api/v1/images/upload', { method: 'POST' });

    expect(await nextRequestToken()).toBe('token-a2');
    expect(storage.get()).toBe('token-a2');
    expect(session.getAuthSession()).toMatchObject({ accountEpoch, user });
    expect(accountCacheClears).toBe(0);
  });

  it('ignores a renewal for a request sent with a token that has since been replaced', async () => {
    const first = createDeferredResponse();
    const second = createDeferredResponse();
    route = (request) => (request.path === '/api/v1/boards/first' ? first.promise : second.promise);

    const firstRequest = http.apiFetch('/api/v1/boards/first', { method: 'POST' });
    const secondRequest = http.apiFetch('/api/v1/boards/second', { method: 'POST' });
    first.resolve(json({}, { refreshedToken: 'token-a2' }));
    await firstRequest;
    second.resolve(json({}, { refreshedToken: 'token-a-late' }));
    await secondRequest;
    route = authRoutes;

    expect(await nextRequestToken()).toBe('token-a2');
    expect(storage.get()).toBe('token-a2');
  });

  it('ignores a renewal from an identity lifetime that has ended', async () => {
    const late = createDeferredResponse();
    route = (request) =>
      request.path === '/api/v1/boards/' && request.method === 'POST' ? late.promise : authRoutes(request);

    const lateRequest = http.apiFetch('/api/v1/boards/', { method: 'POST' });
    await session.logoutSession();
    route = (request) =>
      request.path === '/api/v1/auth/login'
        ? json({ expires_in: 86400, token: 'token-b', user: { ...user, user_id: 'user-b' } })
        : authRoutes(request);
    await session.loginWithCredentials('grace@example.com', 'password', false);
    late.resolve(json({}, { refreshedToken: 'token-a2' }));

    await expect(lateRequest).rejects.toBeInstanceOf(http.HttpRequestIdentityExpiredError);
    expect(await nextRequestToken()).toBe('token-b');
    expect(storage.get()).toBe('token-b');
  });

  it('does not expire the session for a 401 to a token that a renewal already replaced', async () => {
    const rejected = createDeferredResponse();
    route = (request) => (request.path === '/api/v1/boards/slow' ? rejected.promise : authRoutes(request));
    const staleRequest = http.apiFetch('/api/v1/boards/slow');

    route = (request) =>
      request.path === '/api/v1/boards/renew' ? json({}, { refreshedToken: 'token-a2' }) : authRoutes(request);
    await http.apiFetch('/api/v1/boards/renew', { method: 'POST' });
    rejected.resolve(json({ detail: 'expired' }, { status: 401 }));

    await expect(staleRequest).rejects.toMatchObject({ status: 401 });
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user });
    expect(await nextRequestToken()).toBe('token-a2');
  });

  it('renews the media cookie with an accepted renewal at most every five minutes', async () => {
    let now = Date.now();
    vi.spyOn(Date, 'now').mockImplementation(() => now);
    let renewals = 0;
    route = (request) =>
      request.path === '/api/v1/boards/' && request.method === 'POST'
        ? json({}, { refreshedToken: `token-a-renewal-${(renewals += 1)}` })
        : authRoutes(request);
    const mediaCookieTokens = (): (string | null)[] =>
      sent.filter((request) => request.path === '/api/v1/auth/media-cookie').map((request) => request.authorization);

    // Login set the cookie moments ago.
    await http.apiFetch('/api/v1/boards/', { method: 'POST' });
    now += 5 * 60_000;
    await http.apiFetch('/api/v1/boards/', { method: 'POST' });
    await http.apiFetch('/api/v1/boards/', { method: 'POST' });
    await vi.waitFor(() => {
      expect(mediaCookieTokens()).toEqual(['Bearer token-a-renewal-2']);
    });
  });
});

describe('own password change', () => {
  beforeEach(signIn);

  const changePassword = (): Promise<unknown> =>
    session.updateOwnProfile({ current_password: 'old-password', new_password: 'new-password' });

  it('keeps the session and sends the replacement on the following request', async () => {
    const { accountEpoch } = session.getAuthSession();
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH'
        ? json(user, { refreshedToken: 'token-a-epoch-2' })
        : authRoutes(request);

    await changePassword();

    expect(session.getAuthSession()).toMatchObject({ accountEpoch, sessionExpired: false, user });
    expect(accountCacheClears).toBe(0);
    expect(await nextRequestToken()).toBe('token-a-epoch-2');
    expect(storage.get()).toBe('token-a-epoch-2');
  });

  it('waits for the change before acting on a 401 that the revocation caused, and drops older renewals', async () => {
    const change = createDeferredResponse();
    route = (request) => {
      if (request.path === '/api/v1/auth/me' && request.method === 'PATCH') {
        return change.promise;
      }
      if (request.path === '/api/v1/boards/revoked') {
        return json({ detail: 'revoked' }, { status: 401 });
      }
      if (request.path === '/api/v1/boards/renewed') {
        // Minted from the old epoch just before the server committed the change.
        return json({}, { refreshedToken: 'token-a-epoch-1-renewed' });
      }
      return authRoutes(request);
    };

    const pendingChange = changePassword();
    await vi.waitFor(() => {
      expect(sent.at(-1)?.method).toBe('PATCH');
    });
    await http.apiFetch('/api/v1/boards/renewed', { method: 'POST' });
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });

    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user });

    change.resolve(json(user, { refreshedToken: 'token-a-epoch-2' }));
    await pendingChange;
    route = authRoutes;

    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user });
    expect(await nextRequestToken()).toBe('token-a-epoch-2');
  });

  it("lets the replacement win over another tab's renewal of the revoked token", async () => {
    const change = createDeferredResponse();
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH' ? change.promise : authRoutes(request);
    // An unsigned token with the backend's user_id claim, renewed by another tab just before the change committed.
    const otherTabRenewal = `e30.${btoa(JSON.stringify({ user_id: user.user_id })).replaceAll('=', '')}.signature`;

    const pendingChange = changePassword();
    await vi.waitFor(() => {
      expect(sent.at(-1)?.method).toBe('PATCH');
    });
    storage.otherTabStores(otherTabRenewal);
    change.resolve(json(user, { refreshedToken: 'token-a-epoch-2' }));
    await pendingChange;
    route = authRoutes;

    expect(await nextRequestToken()).toBe('token-a-epoch-2');
    expect(storage.get()).toBe('token-a-epoch-2');
  });

  it('expires the session for that 401 once the change fails without a replacement', async () => {
    const change = createDeferredResponse();
    route = (request) => {
      if (request.path === '/api/v1/auth/me' && request.method === 'PATCH') {
        return change.promise;
      }
      return request.path === '/api/v1/boards/revoked' ? json({}, { status: 401 }) : authRoutes(request);
    };

    const pendingChange = changePassword().catch((error: unknown) => error);
    await vi.waitFor(() => {
      expect(sent.at(-1)?.method).toBe('PATCH');
    });
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
    change.resolve(json({ detail: 'Current password is incorrect' }, { status: 400 }));

    expect(await pendingChange).toMatchObject({ status: 400 });
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });
    expect(storage.get()).toBeNull();
  });

  it("applies another tab's sign-out at once and never stores the change's late replacement", async () => {
    const change = createDeferredResponse();
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH' ? change.promise : authRoutes(request);

    const pendingChange = changePassword().catch((error: unknown) => error);
    await untilPatchSent('/api/v1/auth/me');
    storage.otherTabStores(null);

    // A deliberate sign-out on a shared machine must end this tab too, whatever the change delivers.
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user: null });
    expect(sentTo('/api/v1/auth/me', 'PATCH')[0]?.signal?.aborted).toBe(true);
    change.resolve(json(user, { refreshedToken: 'token-a-epoch-2' }));

    expect(await pendingChange).toBeInstanceOf(http.HttpRequestIdentityExpiredError);
    expect(storage.get()).toBeNull();
    expect(storage.getRotation()).toBeNull();
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user: null });
    expect(await nextRequestToken()).toBeNull();
  });

  it('announces the rotation to other tabs while the change is in flight', async () => {
    const change = createDeferredResponse();
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH' ? change.promise : authRoutes(request);

    const pendingChange = changePassword();
    await untilPatchSent('/api/v1/auth/me');

    expect(storage.getRotation()).toMatchObject({ userId: user.user_id });

    storage.log.length = 0;
    change.resolve(json(user, { refreshedToken: 'token-a-epoch-2' }));
    await pendingChange;

    // A waiting tab adopts whatever is stored when the announcement is withdrawn, so the replacement comes first.
    expect(storage.log).toEqual(['write:token-a-epoch-2', 'clearRotation']);
    expect(storage.getRotation()).toBeNull();
  });

  it('withdraws only its own announcement, not one another tab of the same user made meanwhile', async () => {
    const change = createDeferredResponse();
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH' ? change.promise : authRoutes(request);
    const otherTabMarker = { at: Date.now() + 5, userId: user.user_id };

    const pendingChange = changePassword();
    await untilPatchSent('/api/v1/auth/me');
    storage.otherTabAnnouncesRotation(otherTabMarker);
    change.resolve(json(user, { refreshedToken: 'token-a-epoch-2' }));
    await pendingChange;

    expect(storage.getRotation()).toBe(otherTabMarker);
  });

  it('withdraws the announcement when this tab leaves or signs out mid-change', async () => {
    // The change never answers; only the abort that ends its lifetime releases it.
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH'
        ? new Promise<Response>((_resolve, reject) => {
            request.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
          })
        : authRoutes(request);

    void changePassword().catch(() => undefined);
    await untilPatchSent('/api/v1/auth/me');
    expect(storage.getRotation()).not.toBeNull();
    window.dispatchEvent(new Event('pagehide'));
    expect(storage.getRotation()).toBeNull();

    await session.logoutSession();
    await signIn();
    void changePassword().catch(() => undefined);
    await vi.waitFor(() => {
      expect(storage.getRotation()).not.toBeNull();
    });
    await session.logoutSession();
    expect(storage.getRotation()).toBeNull();
    expect(storage.get()).toBeNull();
  });

  it('withdraws the announcement when the change fails', async () => {
    const change = createDeferredResponse();
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH' ? change.promise : authRoutes(request);

    const pendingChange = changePassword().catch((error: unknown) => error);
    await untilPatchSent('/api/v1/auth/me');
    change.resolve(json({ detail: 'Current password is incorrect' }, { status: 400 }));
    await pendingChange;

    expect(storage.getRotation()).toBeNull();
    expect(storage.get()).toBe('token-a');
  });

  it("follows another tab's account switch at once and discards the change's late replacement", async () => {
    const change = createDeferredResponse();
    const tokenB = tokenFor(userB.user_id, 'login');
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH' ? change.promise : authRoutes(request);

    const pendingChange = changePassword().catch((error: unknown) => error);
    await untilPatchSent('/api/v1/auth/me');
    storage.otherTabStores(tokenB);
    await vi.waitFor(() => {
      expect(session.getAuthSession().user).toEqual(userB);
    });
    change.resolve(json(user, { refreshedToken: 'token-a-epoch-2' }));

    expect(await pendingChange).toBeInstanceOf(http.HttpRequestIdentityExpiredError);
    expect(sentTo('/api/v1/auth/me', 'PATCH')[0]?.signal?.aborted).toBe(true);
    expect(storage.get()).toBe(tokenB);
    expect(await nextRequestToken()).toBe(tokenB);
  });

  it('sends each queued change with the replacement the previous one delivered', async () => {
    const first = createDeferredResponse();
    let changes = 0;
    route = (request) => {
      if (request.path !== '/api/v1/auth/me' || request.method !== 'PATCH') {
        return authRoutes(request);
      }
      changes += 1;
      return changes === 1 ? first.promise : json(user, { refreshedToken: 'token-a-epoch-3' });
    };

    const firstChange = changePassword();
    const secondChange = changePassword();
    await untilPatchSent('/api/v1/auth/me');
    first.resolve(json(user, { refreshedToken: 'token-a-epoch-2' }));
    await Promise.all([firstChange, secondChange]);
    route = authRoutes;

    expect(sentTo('/api/v1/auth/me', 'PATCH').map((request) => request.authorization)).toEqual([
      'Bearer token-a',
      'Bearer token-a-epoch-2',
    ]);
    expect(await nextRequestToken()).toBe('token-a-epoch-3');
  });

  it('stops deferring for a later lifetime when a stalled change outlives its own', async () => {
    // The change never answers, and this stub ignores aborts, so only the lifetime check releases its hold.
    route = (request) =>
      request.path === '/api/v1/auth/me' && request.method === 'PATCH'
        ? new Promise<Response>(() => {
            // Never answers.
          })
        : authRoutes(request);
    void changePassword();
    await untilPatchSent('/api/v1/auth/me');
    await session.logoutSession();
    await session.loginWithCredentials(user.email, 'password', false);

    route = (request) =>
      request.path === '/api/v1/boards/renew' ? json({}, { refreshedToken: 'token-a2' }) : authRoutes(request);
    await http.apiFetch('/api/v1/boards/renew', { method: 'POST' });
    expect(await nextRequestToken()).toBe('token-a2');

    route = (request) => (request.path === '/api/v1/boards/revoked' ? json({}, { status: 401 }) : authRoutes(request));
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });
  });
});

describe("another tab's password change", () => {
  beforeEach(signIn);

  const revokedRoute = (request: SentRequest): Response =>
    request.path === '/api/v1/boards/revoked' ? json({ detail: 'revoked' }, { status: 401 }) : authRoutes(request);

  it('holds a 401 the change caused until the replacement is stored, then adopts it without a new lifetime', async () => {
    const { accountEpoch } = session.getAuthSession();
    const replacement = tokenFor(user.user_id, 'epoch-2');
    route = revokedRoute;

    storage.otherTabAnnouncesRotation({ at: Date.now(), userId: user.user_id });
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });

    expect(session.getAuthSession()).toMatchObject({ accountEpoch, sessionExpired: false, user });
    expect(accountCacheClears).toBe(0);
    expect(storage.get()).toBe('token-a');

    storage.otherTabStores(replacement);
    storage.otherTabAnnouncesRotation(null);
    route = authRoutes;
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(session.getAuthSession()).toMatchObject({ accountEpoch, sessionExpired: false, user });
    expect(accountCacheClears).toBe(0);
    expect(await nextRequestToken()).toBe(replacement);
  });

  it('expires the session for that 401 once the change is withdrawn without a replacement', async () => {
    route = revokedRoute;

    storage.otherTabAnnouncesRotation({ at: Date.now(), userId: user.user_id });
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
    expect(session.getAuthSession().user).toEqual(user);

    storage.otherTabAnnouncesRotation(null);
    await vi.waitFor(() => {
      expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });
    });
    expect(storage.get()).toBeNull();
  });

  it('expires the session when the announcing tab never settles its change', async () => {
    vi.useFakeTimers();
    try {
      route = revokedRoute;

      storage.otherTabAnnouncesRotation({ at: Date.now(), userId: user.user_id });
      await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
      await vi.advanceTimersByTimeAsync(29_000);
      expect(session.getAuthSession().user).toEqual(user);

      await vi.advanceTimersByTimeAsync(1_000);
      expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });
      expect(storage.get()).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('adopts the replacement over a renewal of the revoked token that lands during the hold', async () => {
    const renewal = createDeferredResponse();
    const replacement = tokenFor(user.user_id, 'epoch-2');
    route = (request) => (request.path === '/api/v1/boards/renew' ? renewal.promise : revokedRoute(request));

    // Sent before the change committed; its renewal still carries the revoked epoch.
    const renewing = http.apiFetch('/api/v1/boards/renew', { method: 'POST' });
    storage.otherTabAnnouncesRotation({ at: Date.now(), userId: user.user_id });
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
    renewal.resolve(json({}, { refreshedToken: 'token-a-epoch-1-renewed' }));
    await renewing;
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });

    storage.otherTabStores(replacement);
    storage.otherTabAnnouncesRotation(null);
    route = authRoutes;
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user });
    expect(storage.get()).toBe(replacement);
    expect(await nextRequestToken()).toBe(replacement);
  });

  it('holds an announcement dated in the future no longer than the wait', async () => {
    vi.useFakeTimers();
    try {
      route = revokedRoute;

      storage.otherTabAnnouncesRotation({ at: Date.now() + 600_000, userId: user.user_id });
      await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
      await vi.advanceTimersByTimeAsync(30_000);

      expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });
    } finally {
      vi.useRealTimers();
    }
  });

  it('ignores an announcement for another principal or one older than the wait', async () => {
    route = revokedRoute;

    storage.otherTabAnnouncesRotation({ at: Date.now(), userId: userB.user_id });
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });

    await signIn();
    route = revokedRoute;
    storage.otherTabAnnouncesRotation({ at: Date.now() - 31_000, userId: user.user_id });
    await expect(http.apiFetch('/api/v1/boards/revoked')).rejects.toMatchObject({ status: 401 });
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });
  });
});

describe('administrator edits', () => {
  beforeEach(signIn);

  it("adopts the replacement for an administrator's own password reset without a new lifetime", async () => {
    const { accountEpoch } = session.getAuthSession();
    route = (request) =>
      request.path === '/api/v1/auth/users/user-a'
        ? json(user, { refreshedToken: 'token-a-epoch-2' })
        : authRoutes(request);

    await session.updateManagedUser(user.user_id, { password: 'new-password' });

    expect(session.getAuthSession()).toMatchObject({ accountEpoch, user });
    expect(accountCacheClears).toBe(0);
    expect(await nextRequestToken()).toBe('token-a-epoch-2');
  });

  it.each([
    { changes: { password: 'new-password' }, name: "another user's password reset", userId: 'user-b' },
    { changes: { display_name: 'Ada L.' }, name: 'an own edit without a password', userId: 'user-a' },
  ])('treats $name as an ordinary mutation that rotates nothing', async ({ changes, userId }) => {
    const edit = createDeferredResponse();
    const path = `/api/v1/auth/users/${userId}`;
    route = (request) => {
      if (request.path === path) {
        return edit.promise;
      }
      return request.path === '/api/v1/boards/renew' ? json({}, { refreshedToken: 'token-a2' }) : authRoutes(request);
    };

    const pendingEdit = session.updateManagedUser(userId, changes);
    await untilPatchSent(path);
    // A rotation would drop this concurrent renewal and adopt the edit's header instead.
    await http.apiFetch('/api/v1/boards/renew', { method: 'POST' });
    edit.resolve(json(user, { refreshedToken: 'token-a3' }));
    await pendingEdit;
    route = authRoutes;

    expect(await nextRequestToken()).toBe('token-a2');
  });
});

describe('signing in', () => {
  it('leaves the stored token alone while an attempt is pending and when it fails', async () => {
    await session.ensureAuthSession();
    const otherTabToken = tokenFor(userB.user_id, 'login');
    const attempt = createDeferredResponse();
    storage.seed(otherTabToken);
    route = (request) => (request.path === '/api/v1/auth/login' ? attempt.promise : authRoutes(request));

    const pendingLogin = session.loginWithCredentials(user.email, 'wrong-password', false);
    await vi.waitFor(() => {
      expect(sentTo('/api/v1/auth/login', 'POST')).toHaveLength(1);
    });
    expect(storage.get()).toBe(otherTabToken);

    attempt.resolve(json({ detail: 'Invalid credentials' }, { status: 401 }));
    await expect(pendingLogin).rejects.toMatchObject({ status: 401 });

    expect(storage.get()).toBe(otherTabToken);
    expect(session.getAuthSession().user).toBeNull();
  });
});

describe('restoring and following sessions through the transport', () => {
  it('restores from the replacement when the stored token is rejected under an announced rotation', async () => {
    const original = tokenFor(user.user_id, 'epoch-1');
    const replacement = tokenFor(user.user_id, 'epoch-2');
    storage.seed(original);
    storage.otherTabAnnouncesRotation({ at: Date.now(), userId: user.user_id });
    route = (request) =>
      request.authorization === `Bearer ${original}`
        ? json({ detail: 'revoked' }, { status: 401 })
        : authRoutes(request);

    const restoring = session.ensureAuthSession();
    await vi.waitFor(() => {
      expect(sentTo('/api/v1/auth/me')).toHaveLength(1);
    });
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(storage.get()).toBe(original);

    storage.otherTabStores(replacement);
    storage.otherTabAnnouncesRotation(null);

    expect(await restoring).toMatchObject({ phase: 'ready', sessionExpired: false, user });
    expect(sentTo('/api/v1/auth/me').map((request) => request.authorization)).toEqual([
      `Bearer ${original}`,
      `Bearer ${replacement}`,
    ]);
    expect(await nextRequestToken()).toBe(replacement);
  });

  it('expires a rejected stored token once its announced rotation is withdrawn without a replacement', async () => {
    const original = tokenFor(user.user_id, 'epoch-1');
    storage.seed(original);
    storage.otherTabAnnouncesRotation({ at: Date.now(), userId: user.user_id });
    route = (request) =>
      request.path === '/api/v1/auth/me' ? json({ detail: 'revoked' }, { status: 401 }) : authRoutes(request);

    const restoring = session.ensureAuthSession();
    await vi.waitFor(() => {
      expect(sentTo('/api/v1/auth/me')).toHaveLength(1);
    });
    storage.otherTabAnnouncesRotation(null);

    expect(await restoring).toMatchObject({ phase: 'ready', sessionExpired: true, user: null });
    expect(storage.get()).toBeNull();
    expect(await nextRequestToken()).toBeNull();
  });

  it('stays signed out, not expired, when another tab signs out during the announced rotation', async () => {
    const original = tokenFor(user.user_id, 'epoch-1');
    storage.seed(original);
    storage.otherTabAnnouncesRotation({ at: Date.now(), userId: user.user_id });
    route = (request) =>
      request.path === '/api/v1/auth/me' ? json({ detail: 'revoked' }, { status: 401 }) : authRoutes(request);

    const restoring = session.ensureAuthSession();
    await vi.waitFor(() => {
      expect(sentTo('/api/v1/auth/me')).toHaveLength(1);
    });
    storage.otherTabStores(null);
    storage.otherTabAnnouncesRotation(null);

    expect(await restoring).toMatchObject({ phase: 'ready', sessionExpired: false, user: null });
    expect(storage.get()).toBeNull();
  });

  it('follows a stored token rejected under an announced rotation only once the announcement settles', async () => {
    await signIn();
    const original = tokenFor(userB.user_id, 'epoch-1');
    const replacement = tokenFor(userB.user_id, 'epoch-2');
    route = (request) => {
      if (request.authorization === `Bearer ${original}`) {
        return json({ detail: 'revoked' }, { status: 401 });
      }
      return request.path === '/api/v1/auth/me' && request.authorization === `Bearer ${replacement}`
        ? json(userB)
        : authRoutes(request);
    };

    storage.otherTabAnnouncesRotation({ at: Date.now(), userId: userB.user_id });
    storage.otherTabStores(original);
    await vi.waitFor(() => {
      expect(sentTo('/api/v1/auth/me')).toHaveLength(1);
    });
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(storage.get()).toBe(original);
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user: null });

    storage.otherTabStores(replacement);
    storage.otherTabAnnouncesRotation(null);
    await vi.waitFor(() => {
      expect(session.getAuthSession().user).toEqual(userB);
    });
    expect(await nextRequestToken()).toBe(replacement);
  });

  it('sends a restored token for the principal lookup and every later request', async () => {
    storage.seed('stored-token');

    await session.ensureAuthSession();

    expect(sentTo('/api/v1/auth/me')[0]?.authorization).toBe('Bearer stored-token');
    expect(session.getAuthSession()).toMatchObject({ phase: 'ready', user });
    expect(await nextRequestToken()).toBe('stored-token');
  });

  it('expires a restored token the backend rejects and forgets it', async () => {
    storage.seed('revoked-token');
    route = (request) =>
      request.path === '/api/v1/auth/me' ? json({ detail: 'revoked' }, { status: 401 }) : authRoutes(request);

    await session.ensureAuthSession();

    expect(session.getAuthSession()).toMatchObject({ phase: 'ready', sessionExpired: true, user: null });
    expect(storage.get()).toBeNull();
    expect(await nextRequestToken()).toBeNull();
  });

  it("expires the session when the backend rejects another tab's token", async () => {
    await signIn();
    const revoked = tokenFor(userB.user_id, 'revoked');
    route = (request) =>
      request.authorization === `Bearer ${revoked}`
        ? json({ detail: 'revoked' }, { status: 401 })
        : authRoutes(request);

    storage.otherTabStores(revoked);
    await vi.waitFor(() => {
      expect(session.getAuthSession().sessionExpired).toBe(true);
    });

    expect(session.getAuthSession().user).toBeNull();
    expect(storage.get()).toBeNull();
    expect(await nextRequestToken()).toBeNull();
  });

  it("stays signed out with the token kept when another tab's principal cannot be resolved, and retries on return", async () => {
    await signIn();
    const tokenB = tokenFor(userB.user_id, 'login');
    route = (request) =>
      request.path === '/api/v1/auth/me' ? json({ detail: 'unavailable' }, { status: 503 }) : authRoutes(request);

    storage.otherTabStores(tokenB);
    await vi.waitFor(() => {
      expect(sentTo('/api/v1/auth/me')).toHaveLength(1);
    });
    await new Promise((resolve) => {
      setTimeout(resolve, 20);
    });

    expect(sentTo('/api/v1/auth/me')).toHaveLength(1);
    expect(session.getAuthSession()).toMatchObject({ phase: 'ready', sessionExpired: false, user: null });
    expect(storage.get()).toBe(tokenB);
    expect(await nextRequestToken()).toBeNull();

    route = authRoutes;
    page.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => {
      expect(session.getAuthSession().user).toEqual(userB);
    });
    expect(await nextRequestToken()).toBe(tokenB);
  });
});
