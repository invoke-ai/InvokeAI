import { createUuid } from '@platform/browser/randomUuid';
import { createExternalStore } from '@platform/state/externalStore';
import {
  ApiError,
  type HttpAuthAdapter,
  type HttpCredential,
  HttpRequestIdentityExpiredError,
} from '@platform/transport/http';

import { shouldExpireUnauthorizedSession } from './core/sessionPolicy';
import { classifyStoredCredential, readTokenExpiry, readTokenUserId } from './core/storedCredential';
import { browserIdentityTokenAdapter, type CredentialRotationMarker } from './core/tokenStorage';
import {
  getAuthStatus,
  getCurrentUser,
  login,
  logout,
  refreshMediaCookie,
  setupAdmin,
  updateCurrentUser,
  updateUser,
  type AuthStatus,
  type ProfileUpdateRequest,
  type UserDTO,
  type UserUpdateRequest,
} from './data/api';

/** Auth surfaces stay dormant in single-user mode; an unresolved phase never reads conservative defaults as a grant. */
export interface AuthSession {
  /** Remount key for every authenticated lifetime, including same-user logins. */
  accountEpoch: number;
  /** Auth mode is never inferred when `/auth/status` cannot be resolved. */
  phase: 'unknown' | 'unavailable' | 'ready';
  multiuserEnabled: boolean;
  setupRequired: boolean;
  /** During setup only: work made without an account exists, which the account being created will not see. */
  hasSingleUserWork: boolean;
  strictPasswordChecking: boolean;
  user: UserDTO | null;
  sessionExpired: boolean;
  /** The expired account, until the next sign-in or sign-out; only its sign-in returns where it was. */
  expiredUserId: string | null;
  /** While this tab follows another tab's sign-in to a different account (the old lifetime already ended). */
  switchingAccount: boolean;
}

const store = createExternalStore<AuthSession>({
  accountEpoch: 0,
  multiuserEnabled: false,
  expiredUserId: null,
  hasSingleUserWork: false,
  phase: 'unknown',
  sessionExpired: false,
  setupRequired: false,
  switchingAccount: false,
  strictPasswordChecking: false,
  user: null,
});

export interface IdentityAccountLifecycle {
  activate(accountId: string, storageSuffix?: string): { readonly epoch: number };
  capture(): { readonly epoch: number; readonly signal: AbortSignal };
  invalidate(): { readonly epoch: number };
}

type LifetimeScope = ReturnType<IdentityAccountLifecycle['capture']>;

let accountLifecycle: IdentityAccountLifecycle | null = null;

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

/** Workbench imperative reads follow route-guard resolution; changing users remounts the route. */
export const getAuthSession = (): AuthSession => store.getSnapshot();

/** Single-user servers tag events with a system user id, so only multi-user sessions can be fenced. */
export const getSessionUserId = (): string | null => {
  const session = getAuthSession();
  return session.multiuserEnabled ? (session.user?.user_id ?? null) : null;
};

export const subscribeAuthSession = (listener: () => void): (() => void) => store.subscribe(listener);

/** For synchronous callers; async work captures `AccountScope.storageSuffix` instead of reading this after an await. */
let activeUserScope = '';

const setActiveUserScope = (user: UserDTO | null): void => {
  activeUserScope = user === null ? '' : `:user:${user.user_id}`;
};

/** Single-user mode keeps the existing unsuffixed keys. */
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

// The in-memory token is authoritative per tab and dies with its lifetime; storage is read on resolve and sync only.

const tokenStore = browserIdentityTokenAdapter;

let heldCredential: { readonly scope: LifetimeScope; readonly token: string } | null = null;

/** The value this tab wrote or adopted; anything else in storage came from another tab; `undefined` if unknown. */
let observedStoredToken: string | null | undefined;

const credentialListeners = new Set<() => void>();

const captureCredential = (): HttpCredential => {
  const scope = getAccountLifecycle().capture();

  return { identity: scope, token: heldCredential?.scope === scope ? heldCredential.token : null };
};

const getHeldToken = (): string | null => captureCredential().token;

const isCredentialCurrent = (credential: HttpCredential): boolean => {
  const current = captureCredential();

  return credential.token !== null && credential.identity === current.identity && credential.token === current.token;
};

/** Bind `token` to the current lifetime, which must already be the lifetime it authenticates. */
const bindHeldToken = (token: string | null): void => {
  heldCredential = token === null ? null : { scope: getAccountLifecycle().capture(), token };
};

const persistToken = (token: string | null): void => {
  if (token === null) {
    tokenStore.clear();
  } else {
    tokenStore.write(token);
  }

  // A failed write leaves storage as it was; a concurrent foreign value is not this tab's to have observed.
  if (tokenStore.read() === token) {
    observedStoredToken = token;
  }
};

/** A rejection removes only the rejected token, never a newer one another tab stored meanwhile. */
const forgetStoredToken = (token: string | null): void => {
  if (token !== null && tokenStore.read() === token) {
    persistToken(null);
  }
};

/** Replace the token within the current lifetime: no remount, no account-owned cleanup. */
const replaceHeldToken = (token: string, persist: boolean): void => {
  bindHeldToken(token);

  if (persist) {
    persistToken(token);
  }

  for (const listener of credentialListeners) {
    listener();
  }
};

/** Sliding renewals do not re-issue the media cookie, whose lifetime ends with the token it was issued from. */
const MEDIA_COOKIE_RENEWAL_INTERVAL_MS = 5 * 60_000;

let lastMediaCookieIssue: { readonly at: number; readonly scope: LifetimeScope } | null = null;

const noteMediaCookieIssued = (): void => {
  lastMediaCookieIssue = { at: Date.now(), scope: getAccountLifecycle().capture() };
};

const renewMediaCookie = (): void => {
  const scope = getAccountLifecycle().capture();

  if (
    lastMediaCookieIssue?.scope === scope &&
    Date.now() - lastMediaCookieIssue.at < MEDIA_COOKIE_RENEWAL_INTERVAL_MS
  ) {
    return;
  }

  noteMediaCookieIssued();
  void refreshProtectedMediaCookie();
};

interface CredentialRotation {
  announced: CredentialRotationMarker | null;
  deferredStorageSync: boolean;
  deferredUnauthorized: HttpCredential | null;
  readonly scope: LifetimeScope;
}

let rotation: CredentialRotation | null = null;
let rotationQueue: Promise<unknown> = Promise.resolve();

/** The own-credential rotation in flight for the current lifetime; one started by an ended lifetime no longer counts. */
const getPendingRotation = (): CredentialRotation | null =>
  rotation?.scope === getAccountLifecycle().capture() ? rotation : null;

/** A password change hashes twice and round-trips once; an older marker was left by a tab that unloaded mid-change. */
const ANNOUNCED_ROTATION_WAIT_MS = 30_000;

/** A future-dated marker counts from its first read (`firstSeen`), so it never holds longer than the wait. */
const readAnnouncedRotation = (
  userId: string | null,
  firstSeen?: Map<string, number>
): CredentialRotationMarker | null => {
  const markers = userId === null ? undefined : tokenStore.readRotations();
  const now = Date.now();
  let latest: CredentialRotationMarker | null = null;

  for (const marker of markers ?? []) {
    if (marker.userId !== userId) {
      continue;
    }

    const seen = firstSeen?.get(marker.id) ?? now;
    const at = Math.min(marker.at, seen);

    firstSeen?.set(marker.id, seen);

    if (now - at < ANNOUNCED_ROTATION_WAIT_MS && (latest === null || at > latest.at)) {
      latest = { ...marker, at };
    }
  }

  return latest;
};

/** Expired markers are pruned first; a future-dated orphan ages out once real time passes it. */
const announceRotation = (marker: CredentialRotationMarker): void => {
  for (const stale of tokenStore.readRotations() ?? []) {
    if (marker.at - stale.at >= ANNOUNCED_ROTATION_WAIT_MS) {
      tokenStore.clearRotation(stale.id);
    }
  }

  tokenStore.writeRotation(marker);
};

/** Waits on every live announcement, not just `marker`, so one failed concurrent change cannot free another's 401. */
const untilAnnouncedRotationSettles = (marker: CredentialRotationMarker): Promise<void> =>
  new Promise((resolve) => {
    // Remembered for the whole wait, so re-reading a future-dated marker cannot keep restarting its wait.
    const firstSeen = new Map([[marker.id, marker.at]]);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const settle = (): void => {
      clearTimeout(timer);
      unsubscribe();
      resolve();
    };
    const holdUntilWaitEnds = (latest: CredentialRotationMarker): void => {
      clearTimeout(timer);
      timer = setTimeout(settle, Math.max(0, latest.at + ANNOUNCED_ROTATION_WAIT_MS - Date.now()));
    };
    const unsubscribe = tokenStore.subscribe(() => {
      const latest = readAnnouncedRotation(marker.userId, firstSeen);

      if (latest === null) {
        settle();
      } else {
        holdUntilWaitEnds(latest);
      }
    });

    holdUntilWaitEnds(marker);
  });

let awaitingRotation: HttpCredential | null = null;

/** A 401 while another tab rotates this principal's credential is likely that revocation: wait for the replacement. */
const deferUnauthorizedToAnnouncedRotation = (credential: HttpCredential, marker: CredentialRotationMarker): void => {
  if (awaitingRotation?.token === credential.token) {
    return;
  }

  awaitingRotation = credential;
  void untilAnnouncedRotationSettles(marker).then(() => {
    if (awaitingRotation?.token === credential.token) {
      awaitingRotation = null;
    }

    if (!isCredentialCurrent(credential)) {
      return;
    }

    syncStoredCredential();

    if (isCredentialCurrent(credential)) {
      handleUnauthorizedResponse();
    }
  });
};

const acceptRefreshedToken = (credential: HttpCredential, token: string): void => {
  // A rotation adopts only its own replacement; a renewal minted before it would carry the revoked epoch.
  if (getPendingRotation() !== null || token === credential.token) {
    return;
  }

  // Another tab may have replaced the shared token first; follow it rather than overwrite it.
  syncStoredCredential();

  if (!isCredentialCurrent(credential)) {
    return;
  }

  replaceHeldToken(token, true);
  renewMediaCookie();
};

const handleTransportUnauthorized = (credential: HttpCredential): void => {
  if (!isCredentialCurrent(credential)) {
    return;
  }

  const pendingRotation = getPendingRotation();

  if (pendingRotation !== null) {
    // The pending rotation decides: its replacement makes this token superseded, its failure leaves it current.
    pendingRotation.deferredUnauthorized = credential;
    return;
  }

  syncStoredCredential();

  if (!isCredentialCurrent(credential)) {
    return;
  }

  // Before the session resolves, the token's own claim names the principal; it only decides whether to wait.
  const announced = readAnnouncedRotation(
    store.getSnapshot().user?.user_id ?? (credential.token === null ? null : readTokenUserId(credential.token))
  );

  if (announced !== null) {
    // A restore or follow in progress decides what its own rejection means once the announcement settles.
    if (store.getSnapshot().phase === 'ready' && activeTransition === null) {
      deferUnauthorizedToAnnouncedRotation(credential, announced);
    }

    return;
  }

  handleUnauthorizedResponse();
};

const subscribeCredential = (listener: () => void): (() => void) => {
  credentialListeners.add(listener);

  return () => {
    credentialListeners.delete(listener);
  };
};

/** Sliding renewal means only an idle session gets this close to expiry. */
export const SESSION_EXPIRY_WARNING_LEAD_MS = 5 * 60_000;
const MAX_TIMER_DELAY_MS = 2 ** 31 - 1;

export interface SessionExpiryWarning {
  expiresAt: number;
  /** Aborts once the warning is moot: the token was renewed or replaced, or its lifetime ended. */
  signal: AbortSignal;
}

/** Once per held token; a token without an expiry claim never warns, and renewal re-arms for the new token. */
export const watchSessionExpiry = (
  onWarning: (warning: SessionExpiryWarning) => void,
  { leadMs = SESSION_EXPIRY_WARNING_LEAD_MS, now = Date.now }: { leadMs?: number; now?: () => number } = {}
): (() => void) => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let watched: { controller: AbortController; credential: HttpCredential } | null = null;

  const disarm = (): void => {
    clearTimeout(timer);
    watched?.controller.abort();
    watched = null;
  };
  const reconcile = (): void => {
    const credential = captureCredential();

    if (
      watched &&
      watched.credential.token === credential.token &&
      watched.credential.identity === credential.identity
    ) {
      return;
    }

    disarm();
    const expiresAt = credential.token === null ? null : readTokenExpiry(credential.token);

    if (expiresAt === null) {
      return;
    }

    const controller = new AbortController();
    const arm = (): void => {
      const remaining = expiresAt - leadMs - now();

      if (remaining > 0) {
        // Timers cap near 25 days; a longer wait re-arms when it fires.
        timer = setTimeout(arm, Math.min(remaining, MAX_TIMER_DELAY_MS));
      } else if (expiresAt > now() && !controller.signal.aborted) {
        onWarning({ expiresAt, signal: controller.signal });
      }
    };

    watched = { controller, credential };
    arm();
  };
  const unsubscribeCredential = subscribeCredential(reconcile);
  const unsubscribeSession = store.subscribe(reconcile);

  reconcile();

  return () => {
    unsubscribeCredential();
    unsubscribeSession();
    disarm();
  };
};

export const identityTransportAuthAdapter: HttpAuthAdapter = {
  capture: captureCredential,
  onRefreshedToken: acceptRefreshedToken,
  onUnauthorized: handleTransportUnauthorized,
  subscribe: subscribeCredential,
};

/**
 * An own password change revokes every earlier token; until it settles, other replacements are ignored and 401s wait.
 * Rotations run one at a time; one still queued when its lifetime ends is rejected unsent, one in flight aborted.
 */
const rotateOwnCredential = <Result extends { refreshedToken: string | null }>(
  request: (signal: AbortSignal) => Promise<Result>
): Promise<Result> => {
  const scope = getAccountLifecycle().capture();
  const userId = store.getSnapshot().user?.user_id ?? null;
  const run = async (): Promise<Result> => {
    if (getAccountLifecycle().capture() !== scope) {
      throw new HttpRequestIdentityExpiredError();
    }

    const pending: CredentialRotation = {
      announced: userId === null ? null : { at: Date.now(), id: createUuid(), userId },
      deferredStorageSync: false,
      deferredUnauthorized: null,
      scope,
    };

    rotation = pending;

    if (pending.announced !== null) {
      announceRotation(pending.announced);
    }

    try {
      // The transport captures the same credential synchronously when `request` starts.
      const credential = captureCredential();
      const result = await request(scope.signal);

      if (result.refreshedToken !== null) {
        // Another tab switching accounts ends this lifetime first; its renewals and sign-outs stay deferred.
        syncStoredCredential();

        if (isCredentialCurrent(credential)) {
          replaceHeldToken(result.refreshedToken, true);
          noteMediaCookieIssued();
        }
      }

      return result;
    } finally {
      rotation = null;
      // Withdraw before handling a deferred 401, so this tab does not wait on its own settled announcement.
      withdrawAnnouncedRotation(pending);

      if (pending.deferredStorageSync) {
        syncStoredCredential();
      }

      if (pending.deferredUnauthorized !== null) {
        handleTransportUnauthorized(pending.deferredUnauthorized);
      }
    }
  };
  const result = rotationQueue.then(run);

  rotationQueue = result.catch(() => undefined);

  return result;
};

// Each transition invalidates the old lifetime before touching its credential; one async transition runs at a time.

interface IdentityTransition {
  readonly controller: AbortController;
}

let activeTransition: IdentityTransition | null = null;

const beginTransition = (): IdentityTransition => {
  activeTransition?.controller.abort();

  const transition: IdentityTransition = { controller: new AbortController() };
  activeTransition = transition;

  return transition;
};

const cancelTransition = (): boolean => {
  if (activeTransition === null) {
    return false;
  }

  const transition = activeTransition;
  activeTransition = null;
  transition.controller.abort();

  return true;
};

const isCurrentTransition = (transition: IdentityTransition): boolean =>
  activeTransition === transition && !transition.controller.signal.aborted;

const endTransition = (transition: IdentityTransition): void => {
  if (activeTransition === transition) {
    activeTransition = null;
  }
};

const activateAccount = (user: UserDTO, token: string): number => {
  const { epoch } = getAccountLifecycle().activate(user.user_id, `:user:${user.user_id}`);

  bindHeldToken(token);
  setActiveUserScope(user);
  noteMediaCookieIssued();

  return epoch;
};

/** Clears only this rotation's marker; another tab of the same user may have announced its own. */
const withdrawAnnouncedRotation = (pending: CredentialRotation): void => {
  if (pending.announced !== null) {
    tokenStore.clearRotation(pending.announced.id);
    pending.announced = null;
  }
};

/** End the current lifetime; a rotation it announced is abandoned with it, so no other tab keeps waiting on it. */
const endLifetime = (): number => {
  const pendingRotation = getPendingRotation();

  if (pendingRotation !== null) {
    withdrawAnnouncedRotation(pendingRotation);
  }

  const { epoch } = getAccountLifecycle().invalidate();

  return epoch;
};

/** An expiry remembers the account signed in here so its own sign-in can return where it was. */
const signOut = (sessionExpired: boolean, forget: (token: string | null) => void): void => {
  const token = getHeldToken();
  const expiredUserId = sessionExpired ? (store.getSnapshot().user?.user_id ?? null) : null;
  const accountEpoch = endLifetime();

  heldCredential = null;
  forget(token);
  setActiveUserScope(null);
  store.patchSnapshot({ accountEpoch, expiredUserId, sessionExpired, switchingAccount: false, user: null });
};

const publishUnavailableSession = (): AuthSession => {
  cancelTransition();
  setActiveUserScope(null);
  const accountEpoch = endLifetime();
  store.patchSnapshot({
    accountEpoch,
    hasSingleUserWork: false,
    multiuserEnabled: false,
    phase: 'unavailable',
    setupRequired: false,
    strictPasswordChecking: false,
    switchingAccount: false,
    user: null,
  });

  return store.getSnapshot();
};

type PrincipalResolution = { kind: 'user'; user: UserDTO } | { kind: 'rejected' } | { kind: 'unavailable' };

const resolvePrincipal = async (signal?: AbortSignal): Promise<PrincipalResolution> => {
  try {
    const user = await getCurrentUser(signal);
    // A media cookie refresh failure breaks media but must not invalidate the session.
    await refreshMediaCookie(signal).catch(() => undefined);

    return { kind: 'user', user };
  } catch (error) {
    return error instanceof ApiError && (error.status === 401 || error.status === 403)
      ? { kind: 'rejected' }
      : { kind: 'unavailable' };
  }
};

/** A rejection during another tab's announced rotation waits for it, then continues from whatever storage holds. */
const resolveStoredPrincipal = async (
  token: string
): Promise<{ principal: PrincipalResolution | null; token: string | null }> => {
  bindHeldToken(token);
  const principal = await resolvePrincipal();
  const announced = principal.kind === 'rejected' ? readAnnouncedRotation(readTokenUserId(token)) : null;

  if (announced === null) {
    return { principal, token };
  }

  await untilAnnouncedRotationSettles(announced);
  const stored = tokenStore.read();

  if (stored === null) {
    heldCredential = null;
    observedStoredToken = null;

    return { principal: null, token: null };
  }

  if (typeof stored !== 'string' || stored === token) {
    return { principal, token };
  }

  observedStoredToken = stored;
  bindHeldToken(stored);

  return { principal: await resolvePrincipal(), token: stored };
};

const resolveSession = async (): Promise<AuthSession> => {
  let status: AuthStatus;

  try {
    status = await getAuthStatus();
  } catch {
    // Network failure cannot establish auth mode; keep routes and storage owners unmounted until a successful retry.
    return publishUnavailableSession();
  }

  const authenticates = status.multiuser_enabled && !status.setup_required;
  let user: UserDTO | null = null;
  let token: string | null = null;
  let sessionExpired = false;
  let expiredUserId: string | null = null;

  if (status.multiuser_enabled) {
    observedStoredToken = tokenStore.read();
    token = authenticates ? (observedStoredToken ?? null) : null;
  }

  if (token !== null) {
    const restored = await resolveStoredPrincipal(token);

    token = restored.token;

    if (restored.principal?.kind === 'unavailable') {
      // Unknown principal remains unavailable rather than signed out, preventing access to unscoped storage.
      return publishUnavailableSession();
    }

    if (restored.principal?.kind === 'user') {
      user = restored.principal.user;
    } else if (restored.principal?.kind === 'rejected') {
      heldCredential = null;
      forgetStoredToken(token);
      sessionExpired = true;
      expiredUserId = token === null ? null : readTokenUserId(token);
    }
  } else if (authenticates) {
    // Keep an expiry raised while auth status was unavailable so recovery can explain it.
    ({ expiredUserId, sessionExpired } = store.getSnapshot());
  }

  let accountEpoch: number;

  if (!status.multiuser_enabled) {
    accountEpoch = getAccountLifecycle().activate('single-user').epoch;
    setActiveUserScope(null);
  } else if (user !== null && token !== null) {
    accountEpoch = activateAccount(user, token);
  } else {
    accountEpoch = endLifetime();
    setActiveUserScope(null);
  }

  store.patchSnapshot({
    accountEpoch,
    multiuserEnabled: status.multiuser_enabled,
    expiredUserId: user === null ? expiredUserId : null,
    hasSingleUserWork: status.has_single_user_work,
    phase: 'ready',
    sessionExpired,
    setupRequired: status.setup_required,
    strictPasswordChecking: status.strict_password_checking,
    user,
  });
  syncStoredCredential();

  return store.getSnapshot();
};

let pendingResolve: Promise<AuthSession> | null = null;

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

/** The old lifetime ends first, so nothing more is sent or accepted for it. */
const followStoredPrincipal = async (token: string): Promise<void> => {
  const transition = beginTransition();
  const invalidatedEpoch = endLifetime();

  setActiveUserScope(null);
  bindHeldToken(token);
  store.patchSnapshot({
    accountEpoch: invalidatedEpoch,
    expiredUserId: null,
    sessionExpired: false,
    switchingAccount: true,
    user: null,
  });

  const principal = await resolvePrincipal(transition.controller.signal);

  if (!isCurrentTransition(transition)) {
    return;
  }

  const announced = principal.kind === 'rejected' ? readAnnouncedRotation(readTokenUserId(token)) : null;

  if (announced !== null) {
    // A stored replacement supersedes this transition; only a withdrawal without one rejects the token.
    await untilAnnouncedRotationSettles(announced);

    if (!isCurrentTransition(transition)) {
      return;
    }
  }

  endTransition(transition);

  if (principal.kind === 'user') {
    const accountEpoch = activateAccount(principal.user, token);
    store.patchSnapshot({
      accountEpoch,
      hasSingleUserWork: false,
      sessionExpired: false,
      setupRequired: false,
      switchingAccount: false,
      user: principal.user,
    });
  } else if (principal.kind === 'rejected') {
    signOut(true, forgetStoredToken);
  } else {
    // Unknown principal: stay signed out with the token stored and unobserved, so the next sync retries.
    signOut(false, () => undefined);
    observedStoredToken = undefined;
  }
};

/** Other tabs follow the shared credential through a transition and never run one principal on another's token. */
const syncStoredCredential = (): void => {
  const session = store.getSnapshot();

  if (session.phase !== 'ready' || !session.multiuserEnabled) {
    return;
  }

  const stored = tokenStore.read();

  if (stored === undefined || stored === observedStoredToken) {
    return;
  }

  const change = classifyStoredCredential(stored, getHeldToken(), session.user?.user_id ?? null);
  const pendingRotation = getPendingRotation();

  // A rotation supersedes renewals of the revoked token; a sign-out or account change applies at once.
  if (change === 'renewed' && pendingRotation !== null) {
    pendingRotation.deferredStorageSync = true;
    return;
  }

  observedStoredToken = stored;

  if (change === 'removed') {
    cancelTransition();
    signOut(false, () => undefined);
  } else if (change === 'renewed' && stored !== null) {
    replaceHeldToken(stored, false);
  } else if (change === 'principal-changed' && stored !== null) {
    void followStoredPrincipal(stored);
  }
};

/** Storage events miss back/forward cache restores and some returns to view. */
export const startIdentityCredentialSync = (): (() => void) => {
  const onPageShow = (event: PageTransitionEvent): void => {
    if (event.persisted) {
      syncStoredCredential();
    }
  };
  const onVisibilityChange = (): void => {
    if (document.visibilityState === 'visible') {
      syncStoredCredential();
    }
  };
  // A tab leaving mid-change cannot store its replacement, so other tabs should stop waiting for one at once.
  const onPageHide = (): void => {
    const pendingRotation = getPendingRotation();

    if (pendingRotation !== null) {
      withdrawAnnouncedRotation(pendingRotation);
    }
  };
  const unsubscribeStorage = tokenStore.subscribe(syncStoredCredential);

  window.addEventListener('pageshow', onPageShow);
  window.addEventListener('pagehide', onPageHide);
  document.addEventListener('visibilitychange', onVisibilityChange);

  return () => {
    unsubscribeStorage();
    window.removeEventListener('pageshow', onPageShow);
    window.removeEventListener('pagehide', onPageHide);
    document.removeEventListener('visibilitychange', onVisibilityChange);
  };
};

interface ProtectedMediaCookieRefresh {
  scope: LifetimeScope;
  promise: Promise<boolean>;
}

let pendingProtectedMediaCookieRefresh: ProtectedMediaCookieRefresh | null = null;

/** Request failures and completions never leak across account lifetimes. */
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

/** Unknown auth mode is an availability error, never an implicit single-user grant. */
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

  const attempt = beginTransition();

  // Invalidate the old epoch first; storage stays until success so a mistyped password signs out no other tab.
  const invalidatedEpoch = endLifetime();
  heldCredential = null;
  setActiveUserScope(null);
  store.patchSnapshot({ accountEpoch: invalidatedEpoch, sessionExpired: false, switchingAccount: false, user: null });

  try {
    const result = await login({ email, password, remember_me: rememberMe }, attempt.controller.signal);

    if (!isCurrentTransition(attempt)) {
      throw new LoginAttemptSupersededError();
    }

    persistToken(result.token);
    const accountEpoch = activateAccount(result.user, result.token);
    store.patchSnapshot({ accountEpoch, expiredUserId: null, sessionExpired: false, user: result.user });
  } catch (error) {
    if (!isCurrentTransition(attempt)) {
      throw new LoginAttemptSupersededError();
    }

    throw error;
  } finally {
    endTransition(attempt);
  }
};

export const logoutSession = (): Promise<void> => {
  cancelTransition();

  // apiFetch captures the token synchronously, so the best-effort request finishes in the old account.
  void logout().catch(() => {
    // Tokens are stateless on the backend; local sign-out always wins.
  });

  signOut(false, () => persistToken(null));

  return Promise.resolve();
};

export const completeAdminSetup = async (
  email: string,
  displayName: string | null,
  password: string
): Promise<void> => {
  await setupAdmin({ display_name: displayName, email, password });
  store.patchSnapshot({ hasSingleUserWork: false, setupRequired: false });
  await loginWithCredentials(email, password, false);
};

/** Reflect a profile edit only into the authenticated lifetime that started it. */
const setSessionUser = (user: UserDTO, expectedAccountEpoch: number): void => {
  const session = store.getSnapshot();

  if (session.accountEpoch !== expectedAccountEpoch || session.user?.user_id !== user.user_id) {
    return;
  }

  store.patchSnapshot({ user });
};

/** A password change keeps this tab signed in with the replacement token. */
export const updateOwnProfile = async (changes: ProfileUpdateRequest): Promise<UserDTO> => {
  const { accountEpoch } = store.getSnapshot();
  const { user } =
    changes.new_password === undefined
      ? await updateCurrentUser(changes)
      : await rotateOwnCredential((signal) => updateCurrentUser(changes, signal));

  setSessionUser(user, accountEpoch);

  return user;
};

/** Administrator edit of any user; resetting one's own password rotates this tab's credential like a profile change. */
export const updateManagedUser = async (userId: string, changes: UserUpdateRequest): Promise<UserDTO> => {
  const rotatesOwnCredential = changes.password !== undefined && store.getSnapshot().user?.user_id === userId;
  const { user } = rotatesOwnCredential
    ? await rotateOwnCredential((signal) => updateUser(userId, changes, signal))
    : await updateUser(userId, changes);

  return user;
};

// A 401 for the held credential ends its lifetime; known single-user mode stays inert.
const handleUnauthorizedResponse = (): void => {
  const canceledTransition = cancelTransition();
  const session = store.getSnapshot();

  if (!canceledTransition && !shouldExpireUnauthorizedSession(session)) {
    return;
  }

  signOut(true, forgetStoredToken);
};
