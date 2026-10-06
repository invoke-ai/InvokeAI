const AUTH_TOKEN_STORAGE_KEY = 'auth_token';

const ROTATION_STORAGE_KEY = 'auth_token_rotation';

/**
 * Announces to other tabs that this tab is rotating the shared credential (an own password change), so that a 401 the
 * revocation causes elsewhere waits for the replacement instead of ending the session. Stale markers are ignored by
 * age, so a tab that unloads mid-change cannot keep other tabs waiting forever.
 */
export interface CredentialRotationMarker {
  /** `Date.now()` when the rotation started. */
  readonly at: number;
  /** Whose credential is being rotated; a 401 for another principal's token is not the rotation's doing. */
  readonly userId: string;
}

/**
 * Persists the bearer token across reloads and carries other tabs' changes to it. Requests never read it: the tab's
 * in-memory credential is authoritative.
 */
export interface IdentityTokenAdapter {
  /** `undefined` when storage cannot be read. */
  read(): string | null | undefined;
  write(token: string): void;
  clear(): void;
  /** `undefined` when storage cannot be read; `null` when no rotation is announced. */
  readRotation(): CredentialRotationMarker | null | undefined;
  writeRotation(marker: CredentialRotationMarker): void;
  clearRotation(): void;
  /** Notifies when another document changes the stored token or the rotation marker. */
  subscribe(onChange: () => void): () => void;
}

const parseRotationMarker = (value: string | null): CredentialRotationMarker | null => {
  if (value === null) {
    return null;
  }

  try {
    const parsed: unknown = JSON.parse(value);

    return typeof parsed === 'object' &&
      parsed !== null &&
      'at' in parsed &&
      typeof parsed.at === 'number' &&
      'userId' in parsed &&
      typeof parsed.userId === 'string'
      ? { at: parsed.at, userId: parsed.userId }
      : null;
  } catch {
    return null;
  }
};

export const browserIdentityTokenAdapter: IdentityTokenAdapter = {
  clear: () => {
    try {
      window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    } catch {
      // Nothing to clear if storage is unavailable.
    }
  },
  clearRotation: () => {
    try {
      window.localStorage.removeItem(ROTATION_STORAGE_KEY);
    } catch {
      // Nothing to clear if storage is unavailable.
    }
  },
  read: () => {
    try {
      return window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
    } catch {
      return undefined;
    }
  },
  readRotation: () => {
    try {
      return parseRotationMarker(window.localStorage.getItem(ROTATION_STORAGE_KEY));
    } catch {
      return undefined;
    }
  },
  subscribe: (onChange) => {
    const onStorage = (event: StorageEvent): void => {
      // A null key means another document cleared all of storage.
      if (event.key === AUTH_TOKEN_STORAGE_KEY || event.key === ROTATION_STORAGE_KEY || event.key === null) {
        onChange();
      }
    };

    window.addEventListener('storage', onStorage);

    return () => window.removeEventListener('storage', onStorage);
  },
  write: (token) => {
    try {
      window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
    } catch {
      // Storage unavailable: the in-memory credential lasts until reload.
    }
  },
  writeRotation: (marker) => {
    try {
      window.localStorage.setItem(ROTATION_STORAGE_KEY, JSON.stringify(marker));
    } catch {
      // Storage unavailable: other tabs cannot see this tab's credential either, so there is nothing to announce.
    }
  },
};
