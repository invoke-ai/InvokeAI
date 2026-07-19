import { createExternalStore } from '@platform/state/externalStore';
import { ApiError } from '@platform/transport/http';
import { socketHub } from '@platform/transport/socketHub';

import { shouldExpireUnauthorizedSession } from './core/sessionPolicy';
import { browserIdentityTokenAdapter } from './core/tokenStorage';
import { getAuthStatus, getCurrentUser, login, logout, setupAdmin, type AuthStatus, type UserDTO } from './data/api';

/**
 * Session-lived auth state shared by the router guards and shell chrome. When
 * multi-user is disabled on the backend the snapshot stays at its defaults
 * (`multiuserEnabled: false`, `user: null`) and the entire auth surface — login
 * screen, user menu, expiry handling — stays dormant.
 */
export interface AuthSession {
  /** 'unknown' until the first `/auth/status` round-trip resolves. */
  phase: 'unknown' | 'ready';
  multiuserEnabled: boolean;
  setupRequired: boolean;
  strictPasswordChecking: boolean;
  user: UserDTO | null;
  /** Set when a stored token is rejected mid-session; shown on the login screen. */
  sessionExpired: boolean;
}

const store = createExternalStore<AuthSession>({
  multiuserEnabled: false,
  phase: 'unknown',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  user: null,
});

export const useAuthSession = (): AuthSession => store.useSnapshot();

/**
 * Imperative read for non-reactive callers (e.g. the widget registry). Safe
 * inside the workbench: the route guard resolves the session before mounting,
 * and a user change remounts the whole workbench route.
 */
export const getAuthSession = (): AuthSession => store.getSnapshot();

/**
 * Sticky across sign-out: a debounced autosave can still fire during the
 * logout transition, and it must land in the bucket of the user whose data
 * was loaded — never the shared single-user bucket.
 */
let activeUserScope = '';

const setActiveUserScope = (user: UserDTO | null): void => {
  if (user !== null) {
    activeUserScope = `:user:${user.user_id}`;
  }
};

/**
 * Suffix for localStorage keys holding user-owned state (projects, account
 * preferences, personal API keys — see the spec's State Ownership section).
 * Empty in single-user mode, so existing keys keep working unchanged.
 */
export const getUserStorageScope = (): string => (store.getSnapshot().multiuserEnabled ? activeUserScope : '');

const fetchAuthStatus = async (): Promise<AuthStatus> => {
  try {
    return await getAuthStatus();
  } catch {
    // Backend unreachable: let the shell mount in single-user shape. The
    // connection banner reports the outage, and a 401 once the backend returns
    // in multi-user mode routes through the session-expiry path to login.
    return { admin_email: null, multiuser_enabled: false, setup_required: false, strict_password_checking: false };
  }
};

const resolveSession = async (): Promise<AuthSession> => {
  const status = await fetchAuthStatus();
  let user: UserDTO | null = null;

  if (status.multiuser_enabled && !status.setup_required && browserIdentityTokenAdapter.get()) {
    try {
      user = await getCurrentUser();
    } catch (error) {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        browserIdentityTokenAdapter.clear();
      }
      // Any other failure leaves `user` null; the route guard lands on login.
    }
  }

  setActiveUserScope(user);
  store.patchSnapshot({
    multiuserEnabled: status.multiuser_enabled,
    phase: 'ready',
    setupRequired: status.setup_required,
    strictPasswordChecking: status.strict_password_checking,
    user,
  });

  return store.getSnapshot();
};

let pendingResolve: Promise<AuthSession> | null = null;

/** Resolve the session exactly once per app load; later calls reuse the snapshot. */
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

export const loginWithCredentials = async (email: string, password: string, rememberMe: boolean): Promise<void> => {
  // A stale token must not ride along on the login request itself.
  browserIdentityTokenAdapter.clear();

  const result = await login({ email, password, remember_me: rememberMe });

  browserIdentityTokenAdapter.set(result.token);
  setActiveUserScope(result.user);
  store.patchSnapshot({ sessionExpired: false, user: result.user });
};

export const logoutSession = async (): Promise<void> => {
  try {
    await logout();
  } catch {
    // Tokens are stateless on the backend; local sign-out always wins.
  }

  browserIdentityTokenAdapter.clear();
  // Tear down the shared socket so it does not linger authenticated as the
  // previous user; the next authenticated mount reconnects with a fresh token.
  socketHub.disconnect();
  store.patchSnapshot({ sessionExpired: false, user: null });
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

/** Reflect a profile edit (display name, password) into the session snapshot. */
export const setSessionUser = (user: UserDTO): void => {
  store.patchSnapshot({ user });
};

// A 401 on any authenticated request means the stored token expired or was
// revoked. Only react while an authenticated multi-user session is live, so
// failed login attempts and single-user mode never trip it.
export const handleUnauthorizedResponse = (): void => {
  const session = store.getSnapshot();

  if (!shouldExpireUnauthorizedSession(session)) {
    return;
  }

  browserIdentityTokenAdapter.clear();
  socketHub.disconnect();
  store.patchSnapshot({ sessionExpired: true, user: null });
};
