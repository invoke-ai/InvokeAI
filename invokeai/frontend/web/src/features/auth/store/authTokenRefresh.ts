import { tokensBelongToSameUser } from 'features/auth/store/authSlice';

const AUTH_GENERATION_KEY = 'auth_generation';
const MEDIA_AUTH_LOCK = 'invokeai-media-auth';
const FALLBACK_LOCK_PREFIX = `${MEDIA_AUTH_LOCK}:`;
const FALLBACK_LOCK_LEASE_MS = 30_000;
const FALLBACK_LOCK_POLL_MS = 10;

// Bound on the media-cookie sync fetch made while holding the media-auth lock.
export const MEDIA_COOKIE_SYNC_TIMEOUT_MS = 10_000;

// The middleware mints a refreshed token on every successful mutating request, but
// token lifetimes are measured in days — re-committing (and re-syncing the media
// cookie over the network) once a minute is plenty, and it keeps bulk operations from
// paying one serialized cookie round trip per request.
const TOKEN_REFRESH_THROTTLE_MS = 60_000;
let lastTokenRefreshAcceptedAt = 0;

export const isTokenRefreshThrottled = () => Date.now() - lastTokenRefreshAcceptedAt < TOKEN_REFRESH_THROTTLE_MS;

export const markTokenRefreshAccepted = () => {
  lastTokenRefreshAcceptedAt = Date.now();
};

type FallbackLockTicket = {
  choosing: boolean;
  expiresAt: number;
  owner: string;
  ticket: number;
};

const getAuthGeneration = () => {
  const value = Number(localStorage.getItem(AUTH_GENERATION_KEY) ?? 0);
  return Number.isSafeInteger(value) && value >= 0 ? value : 0;
};

export const captureAuthGeneration = () => getAuthGeneration();

export const beginAuthTransition = () => {
  const next = getAuthGeneration() + 1;
  localStorage.setItem(AUTH_GENERATION_KEY, String(next));
  return next;
};

export const shouldAcceptRefreshedToken = (requestToken: string, requestGeneration: number) =>
  getAuthGeneration() === requestGeneration && localStorage.getItem('auth_token') === requestToken;

/**
 * True when a 401 for a request that carried `requestToken` should end the live session.
 *
 * A 401 is evidence about the credential that was *sent*, and about nothing else. By the time
 * one lands, the tab may already belong to someone else: `setCredentials` (a login here) and
 * `externalTokenAdopted` (a login in another tab — localStorage is shared) both swap the token
 * synchronously while earlier requests are still in flight. Ending the session on such a 401
 * destroys the session that just replaced it, and the user who never issued the request is the
 * one logged out. So the token that was sent must still be the live one.
 *
 * Byte equality, deliberately — and note this is the opposite of what `isSameAuthContext` wants.
 * A sliding-window refresh must NOT pass here: it mints a new token for the same login, and a
 * 401 for the token it replaced says nothing about whether the replacement is still good. That
 * costs nothing, because the session ending is not a one-shot event to be caught — if it really
 * has ended, the next request carries the live token and its 401 ends it here.
 *
 * A null `requestToken` never qualifies: unauthenticated requests (client_state probes during
 * page load, the setup-status query) 401 routinely and must not log anyone out.
 */
export const shouldEndSessionForUnauthorized = (requestToken: string | null): boolean =>
  requestToken !== null && localStorage.getItem('auth_token') === requestToken;

/** The session an operation started under. See `isSameAuthContext`. */
export type AuthContext = {
  token: string | null;
};

export const captureAuthContext = (): AuthContext => ({
  token: localStorage.getItem('auth_token'),
});

/**
 * True while the session an operation began under is still the live one.
 *
 * Requests read the bearer token out of localStorage at send time (`dynamicBaseQuery`), so an
 * operation that issues several of them does not carry one identity: log out and back in as
 * someone else midway and the remaining requests go out as the new user. Nothing stops that on
 * its own — RTK Query's `resetApiState` clears the store, not a `queryFn` that is already
 * running. Callers that fan a single user action out into a sequence of requests must therefore
 * capture the context up front and check it before each one.
 *
 * The comparison is on the identity the token carries, not on the token itself and not on the
 * auth generation counter:
 * - Not the bytes, because the sliding-window refresh mints a fresh token for the same login
 *   mid-operation, and aborting on that would abandon a batch for a routine event.
 * - Not the generation counter, because `beginAuthTransition` bumps it when a login or logout
 *   *request is sent*, before anything has changed and regardless of whether it succeeds. A
 *   login in a second tab — even one that fails, or one that signs the same user back in —
 *   would abort an unrelated batch in this one. Identity answers the question the counter was
 *   standing in for, and answers it correctly: whoever the next chunk would be sent as is read
 *   fresh from localStorage every time.
 *
 * `sessionExpiredLogout` removes the token with no request at all, so the token comparison is
 * what catches it. Two nulls compare equal, which keeps deployments that never issue a token —
 * the single-user default — out of this entirely.
 */
export const isSameAuthContext = (context: AuthContext): boolean => {
  const token = localStorage.getItem('auth_token');
  return token === context.token || tokensBelongToSameUser(context.token, token);
};

const getFallbackLockTickets = (): FallbackLockTicket[] => {
  const tickets: FallbackLockTicket[] = [];
  const now = Date.now();
  for (let index = 0; index < localStorage.length; index++) {
    const key = localStorage.key(index);
    if (!key?.startsWith(FALLBACK_LOCK_PREFIX)) {
      continue;
    }
    try {
      const ticket = JSON.parse(localStorage.getItem(key) ?? '') as FallbackLockTicket;
      if (
        ticket.owner &&
        Number.isSafeInteger(ticket.ticket) &&
        ticket.ticket >= 0 &&
        Number.isFinite(ticket.expiresAt) &&
        ticket.expiresAt > now
      ) {
        tickets.push(ticket);
      }
    } catch {
      // Ignore malformed or stale lock records.
    }
  }
  return tickets;
};

const delay = (milliseconds: number) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, milliseconds);
  });

export const createMediaAuthLock = (owner: string) => {
  const key = `${FALLBACK_LOCK_PREFIX}${owner}`;
  let localQueue = Promise.resolve();

  const run = async <T>(callback: () => T | PromiseLike<T>): Promise<T> => {
    const writeTicket = (ticket: number, choosing: boolean) => {
      localStorage.setItem(
        key,
        JSON.stringify({ choosing, expiresAt: Date.now() + FALLBACK_LOCK_LEASE_MS, owner, ticket })
      );
    };

    writeTicket(0, true);
    const ticket = Math.max(0, ...getFallbackLockTickets().map((entry) => entry.ticket)) + 1;
    writeTicket(ticket, false);

    let lastRenewedAt = Date.now();
    while (
      getFallbackLockTickets().some(
        (entry) =>
          entry.owner !== owner &&
          (entry.choosing || entry.ticket < ticket || (entry.ticket === ticket && entry.owner < owner))
      )
    ) {
      await delay(FALLBACK_LOCK_POLL_MS);
      // Renew the waiter's own lease while queueing: tickets are stamped with a 30s
      // expiry at acquisition, and a waiter that outlives it vanishes from other tabs'
      // views — a later-ticket tab would then enter the critical section alongside it.
      if (Date.now() - lastRenewedAt >= FALLBACK_LOCK_LEASE_MS / 3) {
        writeTicket(ticket, false);
        lastRenewedAt = Date.now();
      }
    }

    const heartbeat = setInterval(() => writeTicket(ticket, false), FALLBACK_LOCK_LEASE_MS / 3);
    try {
      return await callback();
    } finally {
      clearInterval(heartbeat);
      localStorage.removeItem(key);
    }
  };

  return <T>(callback: () => T | PromiseLike<T>): Promise<T> => {
    const result = localQueue.then(
      () => run(callback),
      () => run(callback)
    );
    localQueue = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  };
};

const fallbackMediaAuthLock = createMediaAuthLock(
  globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
);

export const runWithMediaAuthLock = <T>(callback: () => T | PromiseLike<T>): Promise<T> => {
  if (typeof navigator !== 'undefined' && navigator.locks) {
    return navigator.locks.request(MEDIA_AUTH_LOCK, callback) as Promise<T>;
  }
  return fallbackMediaAuthLock(callback);
};
