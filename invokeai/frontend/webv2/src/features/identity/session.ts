import { createExternalStore } from '@platform/state/externalStore';
import { ApiError } from '@platform/transport/http';

import { shouldExpireUnauthorizedSession } from './core/sessionPolicy';
import { browserIdentityTokenAdapter } from './core/tokenStorage';
import {
  getAuthStatus,
  getCurrentUser,
  login,
  logout,
  refreshMediaCookie,
  setupAdmin,
  type AuthStatus,
  type UserDTO,
} from './data/api';

/**
 * Session-lived auth state shared by the router guards and shell chrome. When
 * multi-user is disabled on the backend, the resolved snapshot keeps the
 * entire auth surface — login screen, user menu, expiry handling — dormant.
 * Until that mode is known, `phase` keeps the otherwise conservative boolean
 * defaults from being interpreted as a single-user grant.
 */
export interface AuthSession {
  /** Remount key for every authenticated lifetime, including same-user logins. */
  accountEpoch: number;
  /** Auth mode is never inferred when `/auth/status` cannot be resolved. */
  phase: 'unknown' | 'unavailable' | 'ready';
  multiuserEnabled: boolean;
  setupRequired: boolean;
  strictPasswordChecking: boolean;
  user: UserDTO | null;
  /** Set when a stored token is rejected mid-session; shown on the login screen. */
  sessionExpired: boolean;
}

const store = createExternalStore<AuthSession>({
  accountEpoch: 0,
  multiuserEnabled: false,
  phase: 'unknown',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  user: null,
});

export interface IdentityAccountLifecycle {
  activate(accountId: string, storageSuffix?: string): { readonly epoch: number };
  capture(): { readonly epoch: number; readonly signal: AbortSignal };
  invalidate(): { readonly epoch: number };
}

let accountLifecycle: IdentityAccountLifecycle | null = null;

/** Selected once by the App composition root before session resolution. */
export const configureIdentityAccountLifecycle = (lifecycle: IdentityAccountLifecycle): void => {
  accountLifecycle = lifecycle;
};

const getAccountLifecycle = (): IdentityAccountLifecycle => {
  if (!accountLifecycle) {
    throw new Error('Identity account lifecycle has not been configured by the App composition root.');
  }

  return accountLifecycle;
};

export const useAuthSession = (): AuthSession => store.useSnapshot();

/**
 * Imperative read for non-reactive callers (e.g. the widget registry). Safe
 * inside the workbench: the route guard resolves the session before mounting,
 * and a user change remounts the whole workbench route.
 */
export const getAuthSession = (): AuthSession => store.getSnapshot();

/** Stable subscription for App-composed capability read ports. */
export const subscribeAuthSession = (listener: () => void): (() => void) => store.subscribe(listener);

/**
 * Compatibility scope for synchronous settings/storage callers. Long-lived
 * async work must capture `AccountScope.storageSuffix` from the App-composed
 * lifecycle instead of consulting this mutable value after an await.
 */
let activeUserScope = '';

const setActiveUserScope = (user: UserDTO | null): void => {
  activeUserScope = user === null ? '' : `:user:${user.user_id}`;
};

const activateResolvedAccount = (multiuserEnabled: boolean, user: UserDTO | null): number => {
  if (!multiuserEnabled) {
    return getAccountLifecycle().activate('single-user').epoch;
  }

  return user
    ? getAccountLifecycle().activate(user.user_id, `:user:${user.user_id}`).epoch
    : getAccountLifecycle().invalidate().epoch;
};

/**
 * Suffix for localStorage keys holding user-owned state (projects, account
 * preferences, personal API keys — see the spec's State Ownership section).
 * Empty in single-user mode, so existing keys keep working unchanged.
 */
export const getUserStorageScope = (): string => {
  const session = store.getSnapshot();

  if (session.phase !== 'ready') {
    throw new AuthSessionUnavailableError();
  }

  if (!session.multiuserEnabled) {
    return '';
  }

  if (session.user === null || activeUserScope === '') {
    throw new Error('Account-owned storage is unavailable without an authenticated user.');
  }

  return activeUserScope;
};

export class AuthSessionUnavailableError extends Error {
  constructor() {
    super('The backend authentication mode is currently unavailable.');
    this.name = 'AuthSessionUnavailableError';
  }
}

export class LoginAttemptSupersededError extends Error {
  constructor() {
    super('This sign-in attempt was superseded by a newer identity transition.');
    this.name = 'LoginAttemptSupersededError';
  }
}

export const isLoginAttemptSupersededError = (error: unknown): error is LoginAttemptSupersededError =>
  error instanceof LoginAttemptSupersededError;

interface LoginAttempt {
  readonly controller: AbortController;
}

let activeLoginAttempt: LoginAttempt | null = null;

const beginLoginAttempt = (): LoginAttempt => {
  activeLoginAttempt?.controller.abort();

  const attempt: LoginAttempt = { controller: new AbortController() };
  activeLoginAttempt = attempt;

  return attempt;
};

const cancelLoginAttempt = (): boolean => {
  if (activeLoginAttempt === null) {
    return false;
  }

  const attempt = activeLoginAttempt;
  activeLoginAttempt = null;
  attempt.controller.abort();

  return true;
};

const isCurrentLoginAttempt = (attempt: LoginAttempt): boolean =>
  activeLoginAttempt === attempt && !attempt.controller.signal.aborted;

const publishUnavailableSession = (): AuthSession => {
  cancelLoginAttempt();
  setActiveUserScope(null);
  const accountEpoch = getAccountLifecycle().invalidate().epoch;
  store.patchSnapshot({
    accountEpoch,
    multiuserEnabled: false,
    phase: 'unavailable',
    setupRequired: false,
    strictPasswordChecking: false,
    user: null,
  });

  return store.getSnapshot();
};

const resolveSession = async (): Promise<AuthSession> => {
  let status: AuthStatus;

  try {
    status = await getAuthStatus();
  } catch {
    // A network failure says nothing about the backend's auth mode. Publishing
    // an unavailable state keeps every authenticated route and storage owner
    // unmounted while allowing a later route retry to resolve the session.
    return publishUnavailableSession();
  }

  let user: UserDTO | null = null;
  let sessionExpired = false;

  if (status.multiuser_enabled && !status.setup_required && browserIdentityTokenAdapter.get()) {
    try {
      user = await getCurrentUser();
      // This branch is the restore-from-stored-token path, the one case that holds a valid JWT
      // without the media cookie login would have set. Re-issue it before anything renders an
      // <img>, or every thumbnail 401s. Awaited (a small POST) so the first paint already has
      // the cookie, but never fatal: failing here leaves media broken, not the session.
      await refreshMediaCookie().catch(() => undefined);
    } catch (error) {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        browserIdentityTokenAdapter.clear();
        sessionExpired = true;
      } else {
        // The auth mode is known, but the principal is not. Treat this like a
        // status outage instead of incorrectly presenting the user as signed
        // out and making unscoped storage reachable.
        return publishUnavailableSession();
      }
    }
  } else if (status.multiuser_enabled && !status.setup_required) {
    // Preserve an expiry raised while auth status was unavailable so recovery
    // can explain why the remembered session returned to the login screen.
    sessionExpired = store.getSnapshot().sessionExpired;
  }

  setActiveUserScope(user);
  const accountEpoch = activateResolvedAccount(status.multiuser_enabled, user);
  store.patchSnapshot({
    accountEpoch,
    multiuserEnabled: status.multiuser_enabled,
    phase: 'ready',
    sessionExpired,
    setupRequired: status.setup_required,
    strictPasswordChecking: status.strict_password_checking,
    user,
  });

  return store.getSnapshot();
};

let pendingResolve: Promise<AuthSession> | null = null;

/** Resolve once while pending; an unavailable snapshot is retryable. */
export const ensureAuthSession = (): Promise<AuthSession> => {
  const current = store.getSnapshot();

  if (current.phase === 'ready') {
    return Promise.resolve(current);
  }

  pendingResolve ??= resolveSession().finally(() => {
    pendingResolve = null;
  });

  return pendingResolve;
};

interface ProtectedMediaCookieRefresh {
  scope: ReturnType<IdentityAccountLifecycle['capture']>;
  promise: Promise<boolean>;
}

let pendingProtectedMediaCookieRefresh: ProtectedMediaCookieRefresh | null = null;

/**
 * Re-issue the cookie used by native media elements without leaking request
 * failures or completions across authenticated account lifetimes.
 */
export const refreshProtectedMediaCookie = (): Promise<boolean> => {
  const session = store.getSnapshot();

  if (session.phase !== 'ready' || (session.multiuserEnabled && session.user === null)) {
    return Promise.resolve(false);
  }

  const lifecycle = getAccountLifecycle();
  const scope = lifecycle.capture();

  if (scope.signal.aborted || scope.epoch !== session.accountEpoch) {
    return Promise.resolve(false);
  }

  if (pendingProtectedMediaCookieRefresh?.scope === scope) {
    return pendingProtectedMediaCookieRefresh.promise;
  }

  const promise = refreshMediaCookie(scope.signal)
    .then(
      (result) =>
        result.success &&
        !scope.signal.aborted &&
        lifecycle.capture() === scope &&
        store.getSnapshot().accountEpoch === scope.epoch
    )
    .catch(() => false)
    .finally(() => {
      if (pendingProtectedMediaCookieRefresh?.promise === promise) {
        pendingProtectedMediaCookieRefresh = null;
      }
    });

  pendingProtectedMediaCookieRefresh = { promise, scope };
  return promise;
};

export type ReadyAuthSession = AuthSession & { phase: 'ready' };

/**
 * Route-facing resolver. Unknown auth mode is an availability error, never an
 * implicit single-user grant. Retrying the route retries session resolution.
 */
export const ensureReadyAuthSession = async (): Promise<ReadyAuthSession> => {
  const session = await ensureAuthSession();

  if (session.phase !== 'ready') {
    throw new AuthSessionUnavailableError();
  }

  return session as ReadyAuthSession;
};

export const loginWithCredentials = async (email: string, password: string, rememberMe: boolean): Promise<void> => {
  const session = store.getSnapshot();

  if (session.phase !== 'ready' || !session.multiuserEnabled) {
    throw new AuthSessionUnavailableError();
  }

  const attempt = beginLoginAttempt();

  // Invalidate the old epoch before touching its token. This also protects a
  // same-user reauthentication from completions started by the prior login.
  const invalidatedEpoch = getAccountLifecycle().invalidate().epoch;
  browserIdentityTokenAdapter.clear();
  setActiveUserScope(null);
  store.patchSnapshot({ accountEpoch: invalidatedEpoch, sessionExpired: false, user: null });

  try {
    const result = await login({ email, password, remember_me: rememberMe }, attempt.controller.signal);

    if (!isCurrentLoginAttempt(attempt)) {
      throw new LoginAttemptSupersededError();
    }

    browserIdentityTokenAdapter.set(result.token);
    setActiveUserScope(result.user);
    const accountEpoch = getAccountLifecycle().activate(result.user.user_id, `:user:${result.user.user_id}`).epoch;
    store.patchSnapshot({ accountEpoch, sessionExpired: false, user: result.user });
  } catch (error) {
    if (!isCurrentLoginAttempt(attempt)) {
      throw new LoginAttemptSupersededError();
    }

    throw error;
  } finally {
    if (activeLoginAttempt === attempt) {
      activeLoginAttempt = null;
    }
  }
};

export const logoutSession = (): Promise<void> => {
  cancelLoginAttempt();

  // apiFetch captures the current token synchronously. Let the best-effort
  // request finish in the old account while local sign-out proceeds at once.
  void logout().catch(() => {
    // Tokens are stateless on the backend; local sign-out always wins.
  });

  const accountEpoch = getAccountLifecycle().invalidate().epoch;
  browserIdentityTokenAdapter.clear();
  setActiveUserScope(null);
  store.patchSnapshot({ accountEpoch, sessionExpired: false, user: null });

  return Promise.resolve();
};

/** Create the initial admin account, then sign straight in with it. */
export const completeAdminSetup = async (
  email: string,
  displayName: string | null,
  password: string
): Promise<void> => {
  await setupAdmin({ display_name: displayName, email, password });
  store.patchSnapshot({ setupRequired: false });
  await loginWithCredentials(email, password, false);
};

/** Reflect a profile edit only into the authenticated lifetime that started it. */
export const setSessionUser = (user: UserDTO, expectedAccountEpoch: number): void => {
  const session = store.getSnapshot();

  if (session.accountEpoch !== expectedAccountEpoch || session.user?.user_id !== user.user_id) {
    return;
  }

  store.patchSnapshot({ user });
};

// A 401 for the current stored credential invalidates its account lifetime.
// Login requests carry no stored token, and known single-user mode stays inert.
export const handleUnauthorizedResponse = (): void => {
  const canceledLogin = cancelLoginAttempt();
  const session = store.getSnapshot();

  if (!canceledLogin && !shouldExpireUnauthorizedSession(session)) {
    return;
  }

  const accountEpoch = getAccountLifecycle().invalidate().epoch;
  browserIdentityTokenAdapter.clear();
  setActiveUserScope(null);
  store.patchSnapshot({ accountEpoch, sessionExpired: true, user: null });
};
