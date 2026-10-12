import { createUuid } from '@platform/browser/randomUuid';
import { createExternalStore } from '@platform/state/externalStore';
import {
  ApiError,
  type HttpAuthAdapter,
  type HttpCredential,
  HttpRequestIdentityExpiredError,
} from '@platform/transport/http';

import { shouldExpireUnauthorizedSession } from './core/sessionPolicy';
import { classifyStoredCredential, readTokenUserId } from './core/storedCredential';
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

/**
 * Keep auth surfaces dormant in single-user mode; unresolved phase must never interpret conservative defaults as a
 * grant.
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

type LifetimeScope = ReturnType<IdentityAccountLifecycle['capture']>;

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

/** Workbench imperative reads follow route-guard resolution; changing users remounts the route. */
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

/** Scope user-owned storage keys by account; single-user mode preserves existing unsuffixed keys. */
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

// ---------------------------------------------------------------------------------------------------------------
// Credential. The in-memory token is authoritative for this tab and is bound to the identity lifetime it was issued
// for: once that lifetime ends, no request can carry it. Browser storage persists it across reloads and carries other
// tabs' changes; it is read when a session resolves and when reconciling with other tabs, never per request.
// ---------------------------------------------------------------------------------------------------------------

const tokenStore = browserIdentityTokenAdapter;

let heldCredential: { readonly scope: LifetimeScope; readonly token: string } | null = null;

/**
 * The stored value this tab last wrote or adopted. Storage holding anything else was changed by another tab;
 * `undefined` while nothing is known, so the next reconciliation acts on whatever storage holds.
 */
let observedStoredToken: string | null | undefined;

const credentialListeners = new Set<() => void>();

const captureCredential = (): HttpCredential => {
  const scope = getAccountLifecycle().capture();

  return { identity: scope, token: heldCredential?.scope === scope ? heldCredential.token : null };
};

const getHeldToken = (): string | null => captureCredential().token;

/** True while `credential` is the token this tab currently holds for the current lifetime. */
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
  /** The marker this rotation wrote to shared storage, until it withdraws it. */
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

/**
 * How long another tab's announced rotation keeps a rejected token waiting for its replacement. A password change
 * hashes twice and round-trips once; a marker older than this was left by a tab that unloaded mid-change.
 */
const ANNOUNCED_ROTATION_WAIT_MS = 30_000;

/**
 * The rotation of `userId`'s credential announced most recently and still within its wait; another tab may announce
 * a concurrent one, which keeps its own marker. A marker dated in the future (a clock step, or not written by this
 * app) counts from when it was first read, now unless `firstSeen` (marker id to time) recorded an earlier read, so it
 * can never hold longer than the wait.
 */
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

/**
 * Announce this tab's rotation, first removing markers past their wait: they are ignored anyway, and only a tab that
 * unloaded mid-change leaves one behind. A marker dated in the future (a clock step back during a change that never
 * withdrew) is left until real time passes it and it ages out; such orphans are rare, bounded by the size of the
 * clock step, and never hold a 401 longer than the wait.
 */
const announceRotation = (marker: CredentialRotationMarker): void => {
  for (const stale of tokenStore.readRotations() ?? []) {
    if (marker.at - stale.at >= ANNOUNCED_ROTATION_WAIT_MS) {
      tokenStore.clearRotation(stale.id);
    }
  }

  tokenStore.writeRotation(marker);
};

/**
 * Resolves once no rotation of `marker`'s principal is announced any longer, or when the latest live announcement's
 * wait ends. Waiting for every announcement, not just `marker`, keeps one concurrent change that fails from releasing
 * a 401 that another, still in flight, caused. Each storage change re-anchors the wait to the latest live
 * announcement, so a later one extends it and withdrawing the latest shortens it to the next.
 */
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

/** The rejected credential already waiting on an announced rotation; a second 401 for it adds nothing. */
let awaitingRotation: HttpCredential | null = null;

/**
 * A 401 for the current token while another tab has announced it is rotating that principal's credential is most
 * likely the revocation that change caused. Wait for the replacement to be stored (adopted as a renewal) before
 * treating the rejection as the end of the session.
 */
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

  // The rejection may race another tab's replacement of the shared token.
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

/** Selected by the App composition root as the HTTP transport's only source of credentials. */
export const identityTransportAuthAdapter: HttpAuthAdapter = {
  capture: captureCredential,
  onRefreshedToken: acceptRefreshedToken,
  onUnauthorized: handleTransportUnauthorized,
  subscribe: subscribeCredential,
};

/**
 * Run a request that rotates this tab's own credential: an own password change revokes every earlier token and
 * returns the only replacement that survives. Until it settles, other replacements are ignored, and a 401 for the
 * current token or another tab's renewal waits for its outcome; another tab's sign-out applies at once and discards
 * the replacement. The rotation is announced in storage so other tabs hold the 401 the revocation causes until the
 * replacement reaches them. Rotations run one at a time so each sends the credential its predecessor delivered. Each
 * belongs to the lifetime that queued it: one still queued when that lifetime ends is rejected without being sent,
 * since dispatching it later would carry the next account's credential; one in flight is aborted with its lifetime.
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
          // The response that delivered the replacement also set the media cookie for it.
          noteMediaCookieIssued();
        }
      }

      return result;
    } finally {
      rotation = null;
      // Withdraw before handling a deferred 401, so this tab does not wait on its own, settled announcement. The
      // withdrawal removes only this rotation's marker: when the 401 came from another tab's concurrent change
      // revoking this tab's token, that tab's marker is still announced and the 401 waits for its replacement.
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

// ---------------------------------------------------------------------------------------------------------------
// Identity transitions. Every transition invalidates the old lifetime before touching its credential, and at most
// one asynchronous transition (a login or following another tab) is current at a time.
// ---------------------------------------------------------------------------------------------------------------

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
  // Login and session resolution set the media cookie for this token.
  noteMediaCookieIssued();

  return epoch;
};

/**
 * Withdrawn once the replacement is stored, so a waiting tab adopts the replacement before its wait ends. Only this
 * rotation's own marker is cleared; another tab of the same user may have announced its own meanwhile.
 */
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

/** End the authenticated lifetime locally; `forget` decides what happens to the stored token. */
const signOut = (sessionExpired: boolean, forget: (token: string | null) => void): void => {
  const token = getHeldToken();
  const accountEpoch = endLifetime();

  heldCredential = null;
  forget(token);
  setActiveUserScope(null);
  store.patchSnapshot({ accountEpoch, sessionExpired, user: null });
};

const publishUnavailableSession = (): AuthSession => {
  cancelTransition();
  setActiveUserScope(null);
  const accountEpoch = endLifetime();
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

type PrincipalResolution = { kind: 'user'; user: UserDTO } | { kind: 'rejected' } | { kind: 'unavailable' };

/** Resolve who the held token belongs to, with the media cookie its media elements need. */
const resolvePrincipal = async (signal?: AbortSignal): Promise<PrincipalResolution> => {
  try {
    const user = await getCurrentUser(signal);
    // Await the media cookie before rendering. Refresh failure breaks media but must not invalidate the session.
    await refreshMediaCookie(signal).catch(() => undefined);

    return { kind: 'user', user };
  } catch (error) {
    return error instanceof ApiError && (error.status === 401 || error.status === 403)
      ? { kind: 'rejected' }
      : { kind: 'unavailable' };
  }
};

/**
 * Resolve who a stored token belongs to. A rejection while another tab has announced a rotation of that principal's
 * credential is most likely the revocation the change caused, so the restore waits for the announcement to settle
 * and continues from whatever storage holds then: the replacement, the same token (the change failed), or nothing
 * (another tab signed out meanwhile).
 */
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
    // Network failure cannot establish auth mode; keep routes and storage owners unmounted until a successful
    // retry.
    return publishUnavailableSession();
  }

  const authenticates = status.multiuser_enabled && !status.setup_required;
  let user: UserDTO | null = null;
  let token: string | null = null;
  let sessionExpired = false;

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
    }
  } else if (authenticates) {
    // Preserve an expiry raised while auth status was unavailable so recovery
    // can explain why the remembered session returned to the login screen.
    sessionExpired = store.getSnapshot().sessionExpired;
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
    phase: 'ready',
    sessionExpired,
    setupRequired: status.setup_required,
    strictPasswordChecking: status.strict_password_checking,
    user,
  });
  // Another tab may have changed the stored token while this one resolved.
  syncStoredCredential();

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

/**
 * Another tab stored a token for a different (or unreadable) principal. The old lifetime ends first, so nothing more
 * is sent or accepted for it, then the new principal resolves and activates like a restored session.
 */
const followStoredPrincipal = async (token: string): Promise<void> => {
  const transition = beginTransition();
  const invalidatedEpoch = endLifetime();

  setActiveUserScope(null);
  bindHeldToken(token);
  store.patchSnapshot({ accountEpoch: invalidatedEpoch, sessionExpired: false, user: null });

  const principal = await resolvePrincipal(transition.controller.signal);

  if (!isCurrentTransition(transition)) {
    return;
  }

  const announced = principal.kind === 'rejected' ? readAnnouncedRotation(readTokenUserId(token)) : null;

  if (announced !== null) {
    // The rejection is most likely the revocation an announced change caused. Its replacement, once stored, starts
    // a newer transition that supersedes this one; only an announcement withdrawn without one rejects the token.
    await untilAnnouncedRotationSettles(announced);

    if (!isCurrentTransition(transition)) {
      return;
    }
  }

  endTransition(transition);

  if (principal.kind === 'user') {
    const accountEpoch = activateAccount(principal.user, token);
    store.patchSnapshot({ accountEpoch, sessionExpired: false, setupRequired: false, user: principal.user });
  } else if (principal.kind === 'rejected') {
    signOut(true, forgetStoredToken);
  } else {
    // The backend could not say who the token belongs to. Stay signed out with the token stored, and leave it
    // unobserved so the next storage event, return to view or page restore tries again.
    signOut(false, () => undefined);
    observedStoredToken = undefined;
  }
};

/**
 * Reconcile this tab with the token another tab stored. Other tabs follow the shared credential through an explicit
 * transition and never keep running one principal on another's token.
 */
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

  // A rotation's replacement supersedes another tab's renewal of the revoked token. A sign-out or another account
  // applies at once: storage cannot say whether a sign-out was deliberate, and a deliberate one must end every tab.
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

/**
 * App starts this once. Storage events cover other tabs' changes; a page restored from the back/forward cache or a
 * tab returning to view may have missed them.
 */
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

  const attempt = beginTransition();

  // Invalidate the old epoch before touching its token. This also protects a same-user reauthentication from
  // completions started by the prior login. Storage is left alone until the attempt succeeds: a mistyped password
  // must not sign out the tabs that share it.
  const invalidatedEpoch = endLifetime();
  heldCredential = null;
  setActiveUserScope(null);
  store.patchSnapshot({ accountEpoch: invalidatedEpoch, sessionExpired: false, user: null });

  try {
    const result = await login({ email, password, remember_me: rememberMe }, attempt.controller.signal);

    if (!isCurrentTransition(attempt)) {
      throw new LoginAttemptSupersededError();
    }

    persistToken(result.token);
    const accountEpoch = activateAccount(result.user, result.token);
    store.patchSnapshot({ accountEpoch, sessionExpired: false, user: result.user });
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

  // apiFetch captures the current token synchronously. Let the best-effort
  // request finish in the old account while local sign-out proceeds at once.
  void logout().catch(() => {
    // Tokens are stateless on the backend; local sign-out always wins.
  });

  signOut(false, () => persistToken(null));

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
const setSessionUser = (user: UserDTO, expectedAccountEpoch: number): void => {
  const session = store.getSnapshot();

  if (session.accountEpoch !== expectedAccountEpoch || session.user?.user_id !== user.user_id) {
    return;
  }

  store.patchSnapshot({ user });
};

/** Update the signed-in user's profile; a password change keeps this tab signed in with the replacement token. */
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

// A 401 for the held credential invalidates its account lifetime. Login
// requests carry no held token, and known single-user mode stays inert.
const handleUnauthorizedResponse = (): void => {
  const canceledTransition = cancelTransition();
  const session = store.getSnapshot();

  if (!canceledTransition && !shouldExpireUnauthorizedSession(session)) {
    return;
  }

  signOut(true, forgetStoredToken);
};
