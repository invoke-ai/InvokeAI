const AUTH_GENERATION_KEY = 'auth_generation';
const MEDIA_AUTH_LOCK = 'invokeai-media-auth';
const FALLBACK_LOCK_PREFIX = `${MEDIA_AUTH_LOCK}:`;
const FALLBACK_LOCK_LEASE_MS = 30_000;
const FALLBACK_LOCK_POLL_MS = 10;

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

    while (
      getFallbackLockTickets().some(
        (entry) =>
          entry.owner !== owner &&
          (entry.choosing || entry.ticket < ticket || (entry.ticket === ticket && entry.owner < owner))
      )
    ) {
      await delay(FALLBACK_LOCK_POLL_MS);
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
