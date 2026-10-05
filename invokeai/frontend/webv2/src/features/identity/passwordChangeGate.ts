import { createUuid } from '@platform/browser/randomUuid';

const MARKER_PREFIX = 'invokeai-webv2-password-change:';
const MAX_WAIT_MS = 30_000;
const POLL_MS = 100;

type TokenClaims = { epoch: number; userId: string };
type Marker = { expiresAt: number; sessionKey: string };

const localMarkers = new Map<string, Marker>();

const getClaims = (token: string | null): TokenClaims | null => {
  const encoded = token?.split('.')[1];

  if (!encoded) {
    return null;
  }

  try {
    const normalized = encoded.replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(atob(normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '='))) as Record<
      string,
      unknown
    >;

    if (typeof payload.user_id !== 'string') {
      return null;
    }

    return {
      epoch: typeof payload.token_epoch === 'number' ? payload.token_epoch : 0,
      userId: payload.user_id,
    };
  } catch {
    return null;
  }
};

export const getTokenUserId = (token: string | null): string | null => getClaims(token)?.userId ?? null;

const getSessionKey = (token: string): string => {
  const claims = getClaims(token);

  return claims ? `${claims.userId}:${claims.epoch}` : token;
};

/** Accept an epoch replacement even when a routine refresh changed the stored token first. */
export const isNewEpochForCurrentSession = (request: string, current: string | null, replacement: string): boolean => {
  const requestClaims = getClaims(request);
  const currentClaims = getClaims(current);
  const replacementClaims = getClaims(replacement);

  return (
    requestClaims !== null &&
    currentClaims !== null &&
    replacementClaims !== null &&
    requestClaims.userId === currentClaims.userId &&
    currentClaims.userId === replacementClaims.userId &&
    requestClaims.epoch === currentClaims.epoch &&
    replacementClaims.epoch > requestClaims.epoch
  );
};

const getStorage = (): Storage | null => {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
};

/** Store intent before sending the password change; never persist bearer bytes. */
export const beginPasswordChange = (token: string | null): (() => void) => {
  if (!token) {
    return () => undefined;
  }

  const sessionKey = getSessionKey(token);
  const id = `${MARKER_PREFIX}${createUuid()}`;
  const marker: Marker = { expiresAt: Date.now() + MAX_WAIT_MS, sessionKey };
  localMarkers.set(id, marker);

  // An unreadable token can still coordinate this tab, but must not be copied into storage.
  if (sessionKey !== token) {
    try {
      getStorage()?.setItem(id, JSON.stringify(marker));
    } catch {
      // Storage can be unavailable while this tab still has an in-memory credential.
    }
  }

  return () => {
    localMarkers.delete(id);
    try {
      getStorage()?.removeItem(id);
    } catch {
      // An expired marker is ignored even if storage becomes unavailable at cleanup.
    }
  };
};

const hasPendingPasswordChange = (token: string): boolean => {
  const sessionKey = getSessionKey(token);
  const now = Date.now();

  for (const [id, marker] of localMarkers) {
    if (marker.expiresAt <= now) {
      localMarkers.delete(id);
      continue;
    }

    if (marker.sessionKey === sessionKey && marker.expiresAt > now) {
      return true;
    }
  }

  // Cross-tab markers are only used for readable JWTs. A malformed token is never persisted.
  if (sessionKey === token) {
    return false;
  }

  try {
    const storage = getStorage();

    if (!storage) {
      return false;
    }

    for (let index = storage.length - 1; index >= 0; index--) {
      const key = storage.key(index);

      if (!key?.startsWith(MARKER_PREFIX)) {
        continue;
      }

      try {
        const marker = JSON.parse(storage.getItem(key) ?? '') as Marker;

        if (marker.expiresAt <= now) {
          storage.removeItem(key);
        } else if (marker.sessionKey === sessionKey) {
          return true;
        }
      } catch {
        // Malformed marker is not evidence of an in-flight password change.
      }
    }
  } catch {
    // Fall back to same-tab markers when storage is blocked.
  }

  return false;
};

/** Bound delayed expiry when an old-token 401 races the response that replaces it. */
export const waitForPasswordChange = async (token: string, isCurrent: () => boolean): Promise<void> => {
  const deadline = Date.now() + MAX_WAIT_MS;

  while (isCurrent() && hasPendingPasswordChange(token) && Date.now() < deadline) {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, Math.min(POLL_MS, deadline - Date.now()));
    });
  }
};

export const isPasswordChangePending = hasPendingPasswordChange;
