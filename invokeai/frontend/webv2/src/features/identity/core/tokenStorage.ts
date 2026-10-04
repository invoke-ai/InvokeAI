const AUTH_TOKEN_STORAGE_KEY = 'auth_token';

/**
 * Persists the bearer token across reloads and carries other tabs' changes to it. Requests never read it: the tab's
 * in-memory credential is authoritative.
 */
export interface IdentityTokenAdapter {
  /** `undefined` when storage cannot be read. */
  read(): string | null | undefined;
  write(token: string): void;
  clear(): void;
  /** Notifies when another document changes the stored token. */
  subscribe(onChange: () => void): () => void;
}

export const browserIdentityTokenAdapter: IdentityTokenAdapter = {
  clear: () => {
    try {
      window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
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
  subscribe: (onChange) => {
    const onStorage = (event: StorageEvent): void => {
      // A null key means another document cleared all of storage.
      if (event.key === AUTH_TOKEN_STORAGE_KEY || event.key === null) {
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
};
