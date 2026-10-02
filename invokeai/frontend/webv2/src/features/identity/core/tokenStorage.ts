const AUTH_TOKEN_STORAGE_KEY = 'auth_token';

export interface IdentityTokenAdapter {
  clear(): void;
  get(): string | null;
  set(token: string): void;
}

export const getAuthToken = (): string | null => {
  try {
    return window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
};

export const setAuthToken = (token: string): void => {
  try {
    window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
  } catch {
    // Storage unavailable: the backend session lasts until reload.
  }
};

export const clearAuthToken = (): void => {
  try {
    window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
  } catch {
    // Nothing to clear if storage is unavailable.
  }
};

export const browserIdentityTokenAdapter: IdentityTokenAdapter = {
  clear: clearAuthToken,
  get: getAuthToken,
  set: setAuthToken,
};
