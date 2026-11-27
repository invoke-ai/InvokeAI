const AUTH_TOKEN_STORAGE_KEY = 'auth_token';

/** Each announced rotation is stored under its own key, so concurrent rotations never overwrite each other. */
const ROTATION_STORAGE_KEY_PREFIX = 'auth_token_rotation:';

/** Earlier builds' single marker; removed on announcing so an orphan from a mid-change unload does not persist. */
const LEGACY_ROTATION_STORAGE_KEY = 'auth_token_rotation';

/** Lets a 401 the revocation causes in other tabs wait for the replacement; stale markers are ignored by age. */
export interface CredentialRotationMarker {
  /** Unique per rotation; the announcing tab withdraws only the marker with its own id. */
  readonly id: string;
  readonly at: number;
  /** Whose credential is being rotated; a 401 for another principal's token is not the rotation's doing. */
  readonly userId: string;
}

/** Requests never read it: the tab's in-memory credential is authoritative. */
export interface IdentityTokenAdapter {
  read(): string | null | undefined;
  write(token: string): void;
  clear(): void;
  readRotations(): readonly CredentialRotationMarker[] | undefined;
  writeRotation(marker: CredentialRotationMarker): void;
  clearRotation(id: string): void;
  subscribe(onChange: () => void): () => void;
}

const parseRotationMarker = (id: string, value: string | null): CredentialRotationMarker | null => {
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
      ? { at: parsed.at, id, userId: parsed.userId }
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
  clearRotation: (id) => {
    try {
      window.localStorage.removeItem(`${ROTATION_STORAGE_KEY_PREFIX}${id}`);
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
  readRotations: () => {
    try {
      const storage = window.localStorage;
      const markers: CredentialRotationMarker[] = [];

      for (let index = 0; index < storage.length; index += 1) {
        const key = storage.key(index);

        if (key?.startsWith(ROTATION_STORAGE_KEY_PREFIX)) {
          const marker = parseRotationMarker(key.slice(ROTATION_STORAGE_KEY_PREFIX.length), storage.getItem(key));

          if (marker !== null) {
            markers.push(marker);
          }
        }
      }

      return markers;
    } catch {
      return undefined;
    }
  },
  subscribe: (onChange) => {
    const onStorage = (event: StorageEvent): void => {
      // A null key means another document cleared all of storage.
      if (
        event.key === null ||
        event.key === AUTH_TOKEN_STORAGE_KEY ||
        event.key.startsWith(ROTATION_STORAGE_KEY_PREFIX)
      ) {
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
  writeRotation: ({ at, id, userId }) => {
    try {
      window.localStorage.removeItem(LEGACY_ROTATION_STORAGE_KEY);
      window.localStorage.setItem(`${ROTATION_STORAGE_KEY_PREFIX}${id}`, JSON.stringify({ at, userId }));
    } catch {
      // Storage unavailable: other tabs cannot see this tab's credential either, so there is nothing to announce.
    }
  },
};
