import { createAccountLifecycle, type AccountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { LoginResult } from './data/api';
import type * as sessionModule from './session';

const testState = vi.hoisted(() => {
  const events: string[] = [];
  const listeners = new Set<() => void>();
  let stored: string | null = null;
  let rotation: { at: number; userId: string } | null = null;
  let blocked = false;

  const tokenAdapter = {
    clear: vi.fn(() => {
      events.push('token.clear');
      if (!blocked) {
        stored = null;
      }
    }),
    clearRotation: () => {
      rotation = null;
    },
    read: vi.fn(() => (blocked ? undefined : stored)),
    readRotation: () => (blocked ? undefined : rotation),
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    write: vi.fn((nextToken: string) => {
      events.push(`token.set:${nextToken}`);
      if (!blocked) {
        stored = nextToken;
      }
    }),
    writeRotation: (marker: { at: number; userId: string }) => {
      if (!blocked) {
        rotation = marker;
      }
    },
  };

  return {
    blockStorage: () => {
      blocked = true;
    },
    events,
    getToken: () => stored,
    /** Another tab changed the shared token; this tab hears about it through a storage event. */
    otherTabStores: (token: string | null) => {
      stored = token;
      for (const listener of listeners) {
        listener();
      }
    },
    reset: () => {
      events.length = 0;
      listeners.clear();
      stored = null;
      rotation = null;
      blocked = false;
      tokenAdapter.clear.mockClear();
      tokenAdapter.read.mockClear();
      tokenAdapter.write.mockClear();
    },
    /** Storage changed without an event reaching this tab, e.g. while it sat in the back/forward cache. */
    seed: (token: string | null) => {
      stored = token;
    },
    tokenAdapter,
  };
});

const api = vi.hoisted(() => ({
  getAuthStatus: vi.fn(),
  getCurrentUser: vi.fn(),
  login: vi.fn(),
  logout: vi.fn(),
  refreshMediaCookie: vi.fn(),
  setupAdmin: vi.fn(),
  updateCurrentUser: vi.fn(),
  updateUser: vi.fn(),
}));

vi.mock('./core/tokenStorage', () => ({
  browserIdentityTokenAdapter: testState.tokenAdapter,
}));

vi.mock('./data/api', () => api);

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

const userB = {
  ...user,
  display_name: 'Grace',
  email: 'grace@example.com',
  user_id: 'user-b',
};

const createDeferred = <Value>(): {
  promise: Promise<Value>;
  reject: (reason?: unknown) => void;
  resolve: (value: Value) => void;
} => {
  let reject!: (reason?: unknown) => void;
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>((resolvePromise, rejectPromise) => {
    reject = rejectPromise;
    resolve = resolvePromise;
  });

  return { promise, reject, resolve };
};

let session: typeof sessionModule;

const heldToken = (): string | null => session.identityTransportAuthAdapter.capture().token;

/** What the transport reports when a request carrying the held credential is rejected. */
const rejectHeldCredential = (): void => {
  session.identityTransportAuthAdapter.onUnauthorized(session.identityTransportAuthAdapter.capture());
};

const createObservedLifecycle = (): {
  lifecycle: AccountLifecycle;
  port: sessionModule.IdentityAccountLifecycle;
} => {
  const lifecycle = createAccountLifecycle();

  lifecycle.register({
    clear: () => {
      testState.events.push('cache.clear');
    },
    name: 'test-account-cache',
  });
  const port = {
    activate: (accountId: string, storageSuffix?: string) => {
      testState.events.push(`lifecycle.activate:${accountId}`);
      return lifecycle.activate(accountId, storageSuffix);
    },
    capture: () => lifecycle.capture(),
    invalidate: () => {
      testState.events.push('lifecycle.invalidate');
      return lifecycle.invalidate();
    },
  };

  return {
    lifecycle,
    port,
  };
};

const resolveSignedOutMultiuserSession = async (): Promise<ReturnType<typeof createObservedLifecycle>> => {
  const observed = createObservedLifecycle();

  session.configureIdentityAccountLifecycle(observed.port);
  await session.ensureAuthSession();
  testState.events.length = 0;

  return observed;
};

beforeEach(async () => {
  vi.resetModules();
  testState.reset();
  api.getAuthStatus.mockReset();
  api.getCurrentUser.mockReset();
  api.login.mockReset();
  api.logout.mockReset();
  api.refreshMediaCookie.mockReset();
  api.setupAdmin.mockReset();
  api.updateCurrentUser.mockReset();
  api.updateUser.mockReset();
  api.refreshMediaCookie.mockImplementation(() => {
    testState.events.push('api.refreshMediaCookie');
    return Promise.resolve({ success: true });
  });

  api.getAuthStatus.mockResolvedValue({
    admin_email: 'admin@example.com',
    multiuser_enabled: true,
    setup_required: false,
    strict_password_checking: true,
  });
  api.login.mockImplementation(() => {
    testState.events.push('api.login');
    return Promise.resolve({ expires_in: 3600, token: 'token-a', user });
  });
  api.logout.mockResolvedValue({ success: true });

  session = await import('./session');
});

describe('identity account transitions', () => {
  it('invalidates old account state before installing a token and publishing a login', async () => {
    await resolveSignedOutMultiuserSession();
    const unsubscribe = session.subscribeAuthSession(() => {
      const snapshot = session.getAuthSession();
      testState.events.push(`publish:${snapshot.user?.user_id ?? 'signed-out'}:${snapshot.accountEpoch}`);
    });

    await session.loginWithCredentials(user.email, 'password', true);

    const accountEpoch = session.getAuthSession().accountEpoch;
    expect(testState.events).toEqual([
      'lifecycle.invalidate',
      'cache.clear',
      expect.stringMatching(/^publish:signed-out:\d+$/),
      'api.login',
      'token.set:token-a',
      'lifecycle.activate:user-a',
      'cache.clear',
      `publish:user-a:${accountEpoch}`,
    ]);
    expect(testState.getToken()).toBe('token-a');
    unsubscribe();
  });

  it('clears account-owned state and the token before publishing transport expiry', async () => {
    await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    testState.events.length = 0;
    const unsubscribe = session.subscribeAuthSession(() => {
      const snapshot = session.getAuthSession();
      testState.events.push(`publish:${snapshot.user?.user_id ?? 'signed-out'}:${snapshot.accountEpoch}`);
    });

    rejectHeldCredential();

    const snapshot = session.getAuthSession();
    expect(testState.events).toEqual([
      'lifecycle.invalidate',
      'cache.clear',
      'token.clear',
      `publish:signed-out:${snapshot.accountEpoch}`,
    ]);
    expect(snapshot).toMatchObject({ sessionExpired: true, user: null });
    expect(testState.getToken()).toBeNull();
    unsubscribe();
  });

  it('completes local logout without waiting for the best-effort server request', async () => {
    await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    testState.events.length = 0;
    let resolveLogout: ((result: { success: boolean }) => void) | undefined;
    api.logout.mockImplementation(() => {
      testState.events.push('api.logout');
      return new Promise((resolve) => {
        resolveLogout = resolve;
      });
    });
    const unsubscribe = session.subscribeAuthSession(() => {
      const snapshot = session.getAuthSession();
      testState.events.push(`publish:${snapshot.user?.user_id ?? 'signed-out'}:${snapshot.accountEpoch}`);
    });

    await session.logoutSession();

    const snapshot = session.getAuthSession();
    expect(testState.events).toEqual([
      'api.logout',
      'lifecycle.invalidate',
      'cache.clear',
      'token.clear',
      `publish:signed-out:${snapshot.accountEpoch}`,
    ]);
    expect(snapshot).toMatchObject({ sessionExpired: false, user: null });
    resolveLogout?.({ success: true });
    unsubscribe();
  });

  it('rotates the epoch when the same user authenticates again', async () => {
    const { lifecycle } = await resolveSignedOutMultiuserSession();

    await session.loginWithCredentials(user.email, 'password', true);
    const firstSession = session.getAuthSession();
    const firstScope = lifecycle.capture();

    api.login.mockResolvedValueOnce({ expires_in: 3600, token: 'token-a-refreshed', user });
    await session.loginWithCredentials(user.email, 'new-password', false);

    const secondSession = session.getAuthSession();
    const secondScope = lifecycle.capture();
    expect(secondSession.user?.user_id).toBe(firstSession.user?.user_id);
    expect(secondSession.accountEpoch).toBeGreaterThan(firstSession.accountEpoch);
    expect(secondScope.accountId).toBe(firstScope.accountId);
    expect(secondScope).not.toBe(firstScope);
    expect(firstScope.signal.aborted).toBe(true);
    expect(lifecycle.isCurrent(firstScope)).toBe(false);
    expect(lifecycle.isCurrent(secondScope)).toBe(true);
    expect(testState.getToken()).toBe('token-a-refreshed');
  });

  it('neither expires nor renews a lifetime from a capture taken before a same-token re-login', async () => {
    await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    const earlierCapture = session.identityTransportAuthAdapter.capture();

    await session.loginWithCredentials(user.email, 'password', true);
    // The backend handed back the same token string, so only the lifetime tells the captures apart.
    expect(heldToken()).toBe(earlierCapture.token);
    session.identityTransportAuthAdapter.onRefreshedToken(earlierCapture, 'token-a-renewed');
    session.identityTransportAuthAdapter.onUnauthorized(earlierCapture);

    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user });
    expect(heldToken()).toBe('token-a');
    expect(testState.getToken()).toBe('token-a');
  });

  it('rejects a profile completion owned by an expired account epoch', async () => {
    await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    const update = createDeferred<{ refreshedToken: string | null; user: typeof user }>();
    api.updateCurrentUser.mockReturnValueOnce(update.promise);

    const lateUpdate = session.updateOwnProfile({ display_name: 'Late User A' });
    await session.logoutSession();
    update.resolve({ refreshedToken: null, user: { ...user, display_name: 'Late User A' } });
    await lateUpdate;

    expect(session.getAuthSession().user).toBeNull();
  });

  it('fails closed when auth status is unavailable, even with a stored token', async () => {
    testState.seed('stored-token');
    testState.events.length = 0;
    api.getAuthStatus.mockRejectedValue(new Error('backend unavailable'));
    const { lifecycle } = createObservedLifecycle();
    session.configureIdentityAccountLifecycle({
      activate: (accountId, storageSuffix) => lifecycle.activate(accountId, storageSuffix),
      capture: () => lifecycle.capture(),
      invalidate: () => lifecycle.invalidate(),
    });

    const snapshot = await session.ensureAuthSession();

    expect(snapshot).toMatchObject({
      multiuserEnabled: false,
      phase: 'unavailable',
      user: null,
    });
    expect(lifecycle.capture()).toMatchObject({ accountId: null, storageSuffix: '' });
    expect(testState.getToken()).toBe('stored-token');
    // No lifetime owns the stored token until its principal resolves, so nothing sends it.
    expect(heldToken()).toBeNull();
    expect(api.getCurrentUser).not.toHaveBeenCalled();
    expect(() => session.getUserStorageScope()).toThrow(session.AuthSessionUnavailableError);
    await expect(session.loginWithCredentials(user.email, 'password', true)).rejects.toBeInstanceOf(
      session.AuthSessionUnavailableError
    );
    expect(api.login).not.toHaveBeenCalled();
    expect(testState.getToken()).toBe('stored-token');
    await expect(session.ensureReadyAuthSession()).rejects.toBeInstanceOf(session.AuthSessionUnavailableError);
  });

  it('never sends a restored token whose principal could not be resolved', async () => {
    testState.seed('stored-token');
    api.getCurrentUser.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    session.configureIdentityAccountLifecycle(createObservedLifecycle().port);

    expect((await session.ensureAuthSession()).phase).toBe('unavailable');
    expect(heldToken()).toBeNull();
    expect(testState.getToken()).toBe('stored-token');
  });

  it('recovers an unavailable stored-token session only after status and principal both resolve', async () => {
    testState.seed('stored-token');
    api.getAuthStatus.mockRejectedValueOnce(new Error('backend unavailable'));
    api.getCurrentUser.mockResolvedValue(user);
    const { lifecycle, port } = createObservedLifecycle();
    session.configureIdentityAccountLifecycle(port);

    expect((await session.ensureAuthSession()).phase).toBe('unavailable');

    const recovered = await session.ensureReadyAuthSession();

    expect(recovered).toMatchObject({
      multiuserEnabled: true,
      phase: 'ready',
      sessionExpired: false,
      user,
    });
    expect(lifecycle.capture()).toMatchObject({ accountId: user.user_id, storageSuffix: `:user:${user.user_id}` });
    expect(session.getUserStorageScope()).toBe(`:user:${user.user_id}`);
    expect(testState.getToken()).toBe('stored-token');
  });

  it('clears a stored token rejected after an auth-status outage and recovers to login', async () => {
    const { ApiError } = await import('@platform/transport/http');
    testState.seed('stored-token');
    api.getAuthStatus.mockRejectedValueOnce(new Error('backend unavailable'));
    api.getCurrentUser.mockRejectedValueOnce(new ApiError('Unauthorized', 401));
    const { lifecycle, port } = createObservedLifecycle();
    session.configureIdentityAccountLifecycle(port);

    expect((await session.ensureAuthSession()).phase).toBe('unavailable');

    const recovered = await session.ensureReadyAuthSession();

    expect(recovered).toMatchObject({
      multiuserEnabled: true,
      phase: 'ready',
      sessionExpired: true,
      user: null,
    });
    expect(lifecycle.capture().accountId).toBeNull();
    expect(testState.getToken()).toBeNull();
    expect(() => session.getUserStorageScope()).toThrow(/without an authenticated user/);
  });

  it('lets the newest concurrent login win when an older response arrives last', async () => {
    await resolveSignedOutMultiuserSession();
    const first = createDeferred<LoginResult>();
    const second = createDeferred<LoginResult>();
    let firstSignal: AbortSignal | undefined;
    api.login
      .mockImplementationOnce((_request, signal: AbortSignal | undefined) => {
        firstSignal = signal;
        return first.promise;
      })
      .mockImplementationOnce(() => second.promise);

    const firstOutcome = session.loginWithCredentials(user.email, 'password-a', true).catch((error) => error);
    const secondLogin = session.loginWithCredentials(userB.email, 'password-b', false);

    expect(firstSignal?.aborted).toBe(true);
    second.resolve({ expires_in: 3600, token: 'token-b', user: userB });
    await secondLogin;
    first.resolve({ expires_in: 3600, token: 'token-a-late', user });

    expect(await firstOutcome).toBeInstanceOf(session.LoginAttemptSupersededError);
    expect(session.getAuthSession().user).toEqual(userB);
    expect(testState.getToken()).toBe('token-b');
  });

  it('does not let a pending login undo logout', async () => {
    await resolveSignedOutMultiuserSession();
    const pending = createDeferred<LoginResult>();
    let signal: AbortSignal | undefined;
    api.login.mockImplementationOnce((_request, nextSignal: AbortSignal | undefined) => {
      signal = nextSignal;
      return pending.promise;
    });
    const loginOutcome = session.loginWithCredentials(user.email, 'password', true).catch((error) => error);

    await session.logoutSession();
    pending.resolve({ expires_in: 3600, token: 'late-token', user });

    expect(signal?.aborted).toBe(true);
    expect(await loginOutcome).toBeInstanceOf(session.LoginAttemptSupersededError);
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user: null });
    expect(testState.getToken()).toBeNull();
  });
});

describe('media cookie recovery on restore', () => {
  it('re-issues the media cookie when a stored token restores a session', async () => {
    testState.seed('stored-token');
    api.getCurrentUser.mockResolvedValue(user);
    const observed = createObservedLifecycle();
    session.configureIdentityAccountLifecycle(observed.port);

    await session.ensureAuthSession();

    // Restored JWT sessions need a media cookie refresh; login already sets it.
    expect(api.refreshMediaCookie).toHaveBeenCalledTimes(1);
    expect(session.getAuthSession()).toMatchObject({ phase: 'ready', user });
  });

  it('still restores the session when the media cookie refresh fails', async () => {
    testState.seed('stored-token');
    api.getCurrentUser.mockResolvedValue(user);
    api.refreshMediaCookie.mockRejectedValue(new Error('network down'));
    const observed = createObservedLifecycle();
    session.configureIdentityAccountLifecycle(observed.port);

    await session.ensureAuthSession();

    // Broken media is a degraded gallery; it must not present the user as signed out.
    expect(session.getAuthSession()).toMatchObject({ phase: 'ready', sessionExpired: false, user });
  });

  it('does not request a media cookie when there is no stored token to restore', async () => {
    await resolveSignedOutMultiuserSession();

    expect(api.refreshMediaCookie).not.toHaveBeenCalled();
  });
});

describe('protected media cookie recovery', () => {
  it('shares one in-flight refresh between concurrent callers in the same account epoch', async () => {
    await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    const refresh = createDeferred<{ success: boolean }>();
    api.refreshMediaCookie.mockReturnValueOnce(refresh.promise);

    const first = session.refreshProtectedMediaCookie();
    const second = session.refreshProtectedMediaCookie();

    expect(api.refreshMediaCookie).toHaveBeenCalledTimes(1);

    refresh.resolve({ success: true });

    await expect(Promise.all([first, second])).resolves.toEqual([true, true]);
  });

  it('aborts the old cookie request before a new account transport starts and gives the new epoch a fresh flight', async () => {
    await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    const oldRefresh = createDeferred<{ success: boolean }>();
    const newRefresh = createDeferred<{ success: boolean }>();
    let oldSignal: AbortSignal | undefined;
    let newSignal: AbortSignal | undefined;
    api.refreshMediaCookie
      .mockImplementationOnce((signal?: AbortSignal) => {
        oldSignal = signal;
        signal?.addEventListener(
          'abort',
          () => {
            testState.events.push('refresh.abort');
          },
          { once: true }
        );
        return oldRefresh.promise;
      })
      .mockImplementationOnce((signal?: AbortSignal) => {
        newSignal = signal;
        return newRefresh.promise;
      });

    const staleOutcome = session.refreshProtectedMediaCookie();
    expect(oldSignal?.aborted).toBe(false);

    testState.events.length = 0;
    api.login.mockImplementationOnce(() => {
      testState.events.push('api.login:new-account');
      return Promise.resolve({ expires_in: 3600, token: 'token-b', user: userB });
    });
    await session.loginWithCredentials(userB.email, 'password', true);

    expect(oldSignal?.aborted).toBe(true);
    expect(testState.events.indexOf('refresh.abort')).toBeGreaterThanOrEqual(0);
    expect(testState.events.indexOf('refresh.abort')).toBeLessThan(testState.events.indexOf('api.login:new-account'));

    const currentOutcome = session.refreshProtectedMediaCookie();

    expect(api.refreshMediaCookie).toHaveBeenCalledTimes(2);
    expect(newSignal?.aborted).toBe(false);
    expect(newSignal).not.toBe(oldSignal);

    newRefresh.resolve({ success: true });
    await expect(currentOutcome).resolves.toBe(true);

    oldRefresh.resolve({ success: true });
    await expect(staleOutcome).resolves.toBe(false);
  });

  it('returns false without throwing when the refresh request fails', async () => {
    await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    api.refreshMediaCookie.mockRejectedValueOnce(new Error('network down'));

    await expect(session.refreshProtectedMediaCookie()).resolves.toBe(false);
  });
});

/** An unsigned token carrying the backend's `user_id` claim; only the claim matters to the client. */
const tokenFor = (userId: string, nonce: string): string => {
  const encode = (value: object): string =>
    btoa(JSON.stringify(value)).replaceAll('=', '').replaceAll('+', '-').replaceAll('/', '_');

  return `${encode({ alg: 'HS256' })}.${encode({ nonce, user_id: userId })}.signature`;
};

describe('cross-tab credential reconciliation', () => {
  let stopSync: (() => void) | null = null;
  let page: EventTarget;
  let visibleDocument: EventTarget & { visibilityState: DocumentVisibilityState };

  const startSync = (): void => {
    page = new EventTarget();
    visibleDocument = Object.assign(new EventTarget(), { visibilityState: 'visible' as DocumentVisibilityState });
    vi.stubGlobal('window', page);
    vi.stubGlobal('document', visibleDocument);
    stopSync = session.startIdentityCredentialSync();
  };

  const recordPublications = (): (() => void) =>
    session.subscribeAuthSession(() => {
      const snapshot = session.getAuthSession();
      testState.events.push(`publish:${snapshot.user?.user_id ?? 'signed-out'}`);
    });

  const signInAsUserA = async (): Promise<ReturnType<typeof createObservedLifecycle>> => {
    const observed = await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    startSync();
    testState.events.length = 0;

    return observed;
  };

  afterEach(() => {
    stopSync?.();
    stopSync = null;
    vi.unstubAllGlobals();
  });

  it("adopts another tab's renewal for the same principal within the current lifetime", async () => {
    const { lifecycle } = await signInAsUserA();
    const scope = lifecycle.capture();
    const renewed = tokenFor(user.user_id, 'renewed');

    testState.otherTabStores(renewed);

    expect(heldToken()).toBe(renewed);
    expect(lifecycle.capture()).toBe(scope);
    expect(testState.events).toEqual([]);
    expect(api.getCurrentUser).not.toHaveBeenCalled();
  });

  it('signs out locally, without expiring the session, when another tab signs out', async () => {
    await signInAsUserA();
    const unsubscribe = recordPublications();

    testState.otherTabStores(null);

    expect(testState.events).toEqual(['lifecycle.invalidate', 'cache.clear', 'publish:signed-out']);
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user: null });
    expect(heldToken()).toBeNull();
    expect(api.logout).not.toHaveBeenCalled();
    unsubscribe();
  });

  it("ends the old lifetime before resolving another principal's token, then activates it like a restore", async () => {
    const { lifecycle } = await signInAsUserA();
    const scopeA = lifecycle.capture();
    const tokenB = tokenFor(userB.user_id, 'b');
    const principal = createDeferred<typeof userB>();
    let requestCredential: ReturnType<typeof session.identityTransportAuthAdapter.capture> | undefined;
    api.getCurrentUser.mockImplementationOnce(() => {
      testState.events.push('api.getCurrentUser');
      requestCredential = session.identityTransportAuthAdapter.capture();
      return principal.promise;
    });
    const unsubscribe = recordPublications();

    testState.otherTabStores(tokenB);

    expect(scopeA.signal.aborted).toBe(true);
    expect(requestCredential).toEqual({ identity: lifecycle.capture(), token: tokenB });
    expect(lifecycle.capture().accountId).toBeNull();

    principal.resolve(userB);
    await vi.waitFor(() => {
      expect(session.getAuthSession().user).toEqual(userB);
    });

    expect(testState.events).toEqual([
      'lifecycle.invalidate',
      'cache.clear',
      'publish:signed-out',
      'api.getCurrentUser',
      'api.refreshMediaCookie',
      'lifecycle.activate:user-b',
      'cache.clear',
      'publish:user-b',
    ]);
    expect(lifecycle.capture().accountId).toBe(userB.user_id);
    expect(session.getUserStorageScope()).toBe(`:user:${userB.user_id}`);
    expect(heldToken()).toBe(tokenB);
    expect(testState.getToken()).toBe(tokenB);
    unsubscribe();
  });

  it('lets the latest stored credential win while an earlier transition is still resolving', async () => {
    await signInAsUserA();
    const userC = { ...userB, user_id: 'user-c' };
    const principalB = createDeferred<typeof userB>();
    const principalC = createDeferred<typeof userC>();
    let signalB: AbortSignal | undefined;
    api.getCurrentUser
      .mockImplementationOnce((signal?: AbortSignal) => {
        signalB = signal;
        return principalB.promise;
      })
      .mockImplementationOnce(() => principalC.promise);

    testState.otherTabStores(tokenFor(userB.user_id, 'b'));
    testState.otherTabStores(tokenFor(userC.user_id, 'c'));

    expect(signalB?.aborted).toBe(true);
    principalB.resolve(userB);
    await Promise.resolve();
    expect(session.getAuthSession().user).toBeNull();

    principalC.resolve(userC);
    await vi.waitFor(() => {
      expect(session.getAuthSession().user).toEqual(userC);
    });
    expect(heldToken()).toBe(tokenFor(userC.user_id, 'c'));
  });

  it('stays signed out when another tab signs out while a transition is resolving', async () => {
    await signInAsUserA();
    const principalB = createDeferred<typeof userB>();
    api.getCurrentUser.mockReturnValueOnce(principalB.promise);

    testState.otherTabStores(tokenFor(userB.user_id, 'b'));
    testState.otherTabStores(null);
    principalB.resolve(userB);
    await principalB.promise;
    await Promise.resolve();

    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user: null });
    expect(heldToken()).toBeNull();
  });

  it('signs out as expired when the backend rejects the followed token', async () => {
    const { ApiError } = await import('@platform/transport/http');
    await signInAsUserA();
    const rejected = tokenFor(userB.user_id, 'revoked');
    api.getCurrentUser.mockRejectedValueOnce(new ApiError('Unauthorized', 401));

    testState.otherTabStores(rejected);
    await vi.waitFor(() => {
      expect(session.getAuthSession().sessionExpired).toBe(true);
    });

    expect(session.getAuthSession().user).toBeNull();
    expect(testState.getToken()).toBeNull();
  });

  it("supersedes this tab's pending sign-in when another tab signs in first", async () => {
    await resolveSignedOutMultiuserSession();
    startSync();
    const pendingLogin = createDeferred<LoginResult>();
    api.login.mockReturnValueOnce(pendingLogin.promise);
    api.getCurrentUser.mockResolvedValueOnce(userB);

    const loginOutcome = session.loginWithCredentials(user.email, 'password', true).catch((error) => error);
    testState.otherTabStores(tokenFor(userB.user_id, 'b'));
    pendingLogin.resolve({ expires_in: 3600, token: 'token-a', user });

    expect(await loginOutcome).toBeInstanceOf(session.LoginAttemptSupersededError);
    await vi.waitFor(() => {
      expect(session.getAuthSession().user).toEqual(userB);
    });
    expect(heldToken()).toBe(tokenFor(userB.user_id, 'b'));
    expect(testState.getToken()).toBe(tokenFor(userB.user_id, 'b'));
  });

  it('reconciles after returning to view or from the back/forward cache without misreading its own writes', async () => {
    await signInAsUserA();

    visibleDocument.dispatchEvent(new Event('visibilitychange'));
    page.dispatchEvent(Object.assign(new Event('pageshow'), { persisted: true }));
    expect(session.getAuthSession().user).toEqual(user);
    expect(testState.events).toEqual([]);

    // Another tab signed out while this page sat in the back/forward cache, so no storage event reached it.
    testState.seed(null);
    page.dispatchEvent(Object.assign(new Event('pageshow'), { persisted: true }));

    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user: null });
  });

  it('reconciles a missed change when the tab becomes visible, not while hidden or on a fresh page show', async () => {
    await signInAsUserA();
    // Another tab signed out while this one was hidden and its storage event was missed.
    testState.seed(null);

    visibleDocument.visibilityState = 'hidden';
    visibleDocument.dispatchEvent(new Event('visibilitychange'));
    page.dispatchEvent(Object.assign(new Event('pageshow'), { persisted: false }));
    expect(session.getAuthSession().user).toEqual(user);

    visibleDocument.visibilityState = 'visible';
    visibleDocument.dispatchEvent(new Event('visibilitychange'));
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: false, user: null });
  });

  it('is inert before the session resolves and in single-user mode', async () => {
    api.getAuthStatus.mockResolvedValue({
      admin_email: null,
      multiuser_enabled: false,
      setup_required: false,
      strict_password_checking: false,
    });
    const observed = createObservedLifecycle();
    session.configureIdentityAccountLifecycle(observed.port);
    startSync();

    testState.otherTabStores(tokenFor(user.user_id, 'early'));
    await session.ensureAuthSession();
    testState.events.length = 0;
    testState.otherTabStores(tokenFor(userB.user_id, 'b'));

    expect(testState.events).toEqual([]);
    expect(api.getCurrentUser).not.toHaveBeenCalled();
    expect(heldToken()).toBeNull();
  });

  it('follows a change another tab made while this tab was still restoring its session', async () => {
    testState.seed(tokenFor(user.user_id, 'a'));
    const restore = createDeferred<typeof user>();
    api.getCurrentUser.mockReturnValueOnce(restore.promise).mockResolvedValueOnce(userB);
    const observed = createObservedLifecycle();
    session.configureIdentityAccountLifecycle(observed.port);
    startSync();

    const resolved = session.ensureAuthSession();
    await vi.waitFor(() => {
      expect(api.getCurrentUser).toHaveBeenCalledOnce();
    });
    testState.otherTabStores(tokenFor(userB.user_id, 'b'));
    restore.resolve(user);
    await resolved;

    await vi.waitFor(() => {
      expect(session.getAuthSession().user).toEqual(userB);
    });
    expect(heldToken()).toBe(tokenFor(userB.user_id, 'b'));
  });

  it('keeps a sign-in working for the tab lifetime when storage is unavailable', async () => {
    testState.blockStorage();
    await signInAsUserA();

    visibleDocument.dispatchEvent(new Event('visibilitychange'));

    expect(session.getAuthSession().user).toEqual(user);
    expect(heldToken()).toBe('token-a');
  });

  it('keeps an own password change working when storage is unavailable', async () => {
    testState.blockStorage();
    await signInAsUserA();
    const { accountEpoch } = session.getAuthSession();
    api.updateCurrentUser.mockResolvedValueOnce({ refreshedToken: 'token-a-epoch-2', user });

    await session.updateOwnProfile({ current_password: 'old', new_password: 'new' });

    expect(session.getAuthSession()).toMatchObject({ accountEpoch, sessionExpired: false, user });
    expect(heldToken()).toBe('token-a-epoch-2');
  });
});
