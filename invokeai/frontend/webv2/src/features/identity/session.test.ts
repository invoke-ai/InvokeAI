import { createAccountLifecycle, type AccountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { LoginResult } from './data/api';
import type * as sessionModule from './session';

const testState = vi.hoisted(() => {
  const events: string[] = [];
  let token: string | null = null;

  const tokenAdapter = {
    clear: vi.fn(() => {
      events.push('token.clear');
      token = null;
    }),
    get: vi.fn(() => token),
    set: vi.fn((nextToken: string) => {
      events.push(`token.set:${nextToken}`);
      token = nextToken;
    }),
  };

  return {
    events,
    getToken: () => token,
    reset: () => {
      events.length = 0;
      token = null;
      tokenAdapter.clear.mockClear();
      tokenAdapter.get.mockClear();
      tokenAdapter.set.mockClear();
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

  return {
    lifecycle,
    port: {
      activate: (accountId, storageSuffix) => {
        testState.events.push(`lifecycle.activate:${accountId}`);
        return lifecycle.activate(accountId, storageSuffix);
      },
      invalidate: () => {
        testState.events.push('lifecycle.invalidate');
        return lifecycle.invalidate();
      },
    },
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
      'token.clear',
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

    session.handleUnauthorizedResponse();

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

  it('rejects a profile completion owned by an expired account epoch', async () => {
    await resolveSignedOutMultiuserSession();
    await session.loginWithCredentials(user.email, 'password', true);
    const oldEpoch = session.getAuthSession().accountEpoch;

    await session.logoutSession();
    session.setSessionUser({ ...user, display_name: 'Late User A' }, oldEpoch);

    expect(session.getAuthSession().user).toBeNull();
  });

  it('fails closed when auth status is unavailable, even with a stored token', async () => {
    testState.tokenAdapter.set('stored-token');
    testState.events.length = 0;
    api.getAuthStatus.mockRejectedValue(new Error('backend unavailable'));
    const { lifecycle } = createObservedLifecycle();
    session.configureIdentityAccountLifecycle({
      activate: (accountId, storageSuffix) => lifecycle.activate(accountId, storageSuffix),
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
    expect(api.getCurrentUser).not.toHaveBeenCalled();
    expect(() => session.getUserStorageScope()).toThrow(session.AuthSessionUnavailableError);
    await expect(session.loginWithCredentials(user.email, 'password', true)).rejects.toBeInstanceOf(
      session.AuthSessionUnavailableError
    );
    expect(api.login).not.toHaveBeenCalled();
    expect(testState.getToken()).toBe('stored-token');
    await expect(session.ensureReadyAuthSession()).rejects.toBeInstanceOf(session.AuthSessionUnavailableError);
  });

  it('recovers an unavailable stored-token session only after status and principal both resolve', async () => {
    testState.tokenAdapter.set('stored-token');
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
    testState.tokenAdapter.set('stored-token');
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

  it('treats a later 401 as rejection of the current stored credential while auth mode is unavailable', async () => {
    testState.tokenAdapter.set('stored-token');
    api.getAuthStatus.mockRejectedValue(new Error('backend unavailable'));
    const { lifecycle, port } = createObservedLifecycle();
    session.configureIdentityAccountLifecycle(port);
    await session.ensureAuthSession();

    session.handleUnauthorizedResponse();

    expect(session.getAuthSession()).toMatchObject({ phase: 'unavailable', sessionExpired: true, user: null });
    expect(lifecycle.capture().accountId).toBeNull();
    expect(testState.getToken()).toBeNull();

    api.getAuthStatus.mockResolvedValue({
      admin_email: 'admin@example.com',
      multiuser_enabled: true,
      setup_required: false,
      strict_password_checking: true,
    });
    await expect(session.ensureReadyAuthSession()).resolves.toMatchObject({
      phase: 'ready',
      sessionExpired: true,
      user: null,
    });
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

  it('does not let a pending login undo transport expiry', async () => {
    await resolveSignedOutMultiuserSession();
    const pending = createDeferred<LoginResult>();
    let signal: AbortSignal | undefined;
    api.login.mockImplementationOnce((_request, nextSignal: AbortSignal | undefined) => {
      signal = nextSignal;
      return pending.promise;
    });
    const loginOutcome = session.loginWithCredentials(user.email, 'password', true).catch((error) => error);

    session.handleUnauthorizedResponse();
    pending.resolve({ expires_in: 3600, token: 'late-token', user });

    expect(signal?.aborted).toBe(true);
    expect(await loginOutcome).toBeInstanceOf(session.LoginAttemptSupersededError);
    expect(session.getAuthSession()).toMatchObject({ sessionExpired: true, user: null });
    expect(testState.getToken()).toBeNull();
  });
});

describe('media cookie recovery on restore', () => {
  it('re-issues the media cookie when a stored token restores a session', async () => {
    testState.tokenAdapter.set('stored-token');
    api.getCurrentUser.mockResolvedValue(user);
    const observed = createObservedLifecycle();
    session.configureIdentityAccountLifecycle(observed.port);

    await session.ensureAuthSession();

    // A restored session has a valid JWT but no media cookie, so every <img> would 401
    // without this call. Login needs no equivalent — the backend sets the cookie there.
    expect(api.refreshMediaCookie).toHaveBeenCalledTimes(1);
    expect(session.getAuthSession()).toMatchObject({ phase: 'ready', user });
  });

  it('still restores the session when the media cookie refresh fails', async () => {
    testState.tokenAdapter.set('stored-token');
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
