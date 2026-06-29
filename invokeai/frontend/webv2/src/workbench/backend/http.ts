/**
 * Shared HTTP client for the InvokeAI backend. Every REST call goes through
 * `apiFetch` so authentication matches the WebSocket connection: both read the
 * same `auth_token` and send it as a bearer token. The token is read per
 * request, so a login that lands mid-session applies without a reload.
 */

const API_BASE_URL = import.meta.env.VITE_INVOKEAI_API_BASE_URL ?? '';
const AUTH_TOKEN_STORAGE_KEY = 'auth_token';

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
    // Storage unavailable (private mode, quota): the session lasts until reload.
  }
};

export const clearAuthToken = (): void => {
  try {
    window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
  } catch {
    // Nothing to clear if storage is unavailable.
  }
};

/**
 * Called when an authenticated request comes back 401 — the stored token is no
 * longer valid. The auth session store registers itself here so the HTTP layer
 * stays unaware of session semantics.
 */
let unauthorizedHandler: (() => void) | null = null;

export const setUnauthorizedHandler = (handler: (() => void) | null): void => {
  unauthorizedHandler = handler;
};

export const getBackendSocketUrl = (): string => {
  if (!API_BASE_URL.trim()) {
    return window.location.origin;
  }

  return new URL(API_BASE_URL, window.location.origin).origin;
};

export const buildApiUrl = (path: string): string => `${API_BASE_URL}${path}`;

/** Resolve a backend-relative resource URL (e.g. image URLs in DTOs) against the API host. */
export const absolutizeApiUrl = (url: string): string => {
  if (!API_BASE_URL || url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }

  return new URL(url, API_BASE_URL).toString();
};

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

const humanizeFieldName = (value: string): string => value.replaceAll('_', ' ');

const getLastString = (values: unknown[]): string | null => {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index];

    if (typeof value === 'string') {
      return value;
    }
  }

  return null;
};

const getValidationIssueMessage = (issue: unknown): string | null => {
  if (!issue || typeof issue !== 'object') {
    return null;
  }

  const record = issue as { ctx?: unknown; input?: unknown; loc?: unknown; msg?: unknown; type?: unknown };
  const loc = Array.isArray(record.loc) ? record.loc : [];
  const lastLoc = getLastString(loc);
  const field = lastLoc ? humanizeFieldName(lastLoc) : null;

  if (record.type === 'multiple_of') {
    const ctx = record.ctx && typeof record.ctx === 'object' ? (record.ctx as { multiple_of?: unknown }) : null;
    const multipleOf = ctx?.multiple_of;

    if (field && typeof multipleOf === 'number') {
      const received =
        typeof record.input === 'number' || typeof record.input === 'string' ? ` (received ${record.input})` : '';

      return `${field} must be a multiple of ${multipleOf}${received}.`;
    }
  }

  if (typeof record.msg === 'string' && record.msg) {
    return field ? `${field}: ${record.msg}` : record.msg;
  }

  return null;
};

export const assertOk = async (response: Response): Promise<Response> => {
  if (response.ok) {
    return response;
  }

  const text = await response.text();
  throw new ApiError(text || `${response.status} ${response.statusText}`, response.status);
};

/** Authenticated fetch that leaves status handling to the caller. */
export const apiFetchRaw = (path: string, init?: RequestInit): Promise<Response> => {
  const token = getAuthToken();
  const headers = new Headers(init?.headers);

  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  return fetch(buildApiUrl(path), { ...init, headers });
};

export const apiFetch = async (path: string, init?: RequestInit): Promise<Response> => {
  const hadToken = getAuthToken() !== null;
  const response = await apiFetchRaw(path, init);

  if (response.status === 401 && hadToken) {
    unauthorizedHandler?.();
  }

  return assertOk(response);
};

/**
 * Backend errors arrive as FastAPI JSON (`{"detail": "..."}`); `ApiError`
 * carries the raw body. This unwraps `detail` into a human-readable message.
 */
export const getApiErrorMessage = (error: unknown, fallback: string): string => {
  if (error instanceof ApiError) {
    try {
      const parsed = JSON.parse(error.message) as { detail?: unknown };

      if (typeof parsed.detail === 'string' && parsed.detail) {
        return parsed.detail;
      }

      // Validation errors come as a list of issues; surface the first one.
      if (Array.isArray(parsed.detail)) {
        const message = getValidationIssueMessage(parsed.detail[0]);

        if (message) {
          return message;
        }
      }
    } catch {
      // Not JSON — fall through to the raw message.
    }

    return error.message || fallback;
  }

  return error instanceof Error && error.message ? error.message : fallback;
};

export const apiFetchJson = async <T>(path: string, init?: RequestInit): Promise<T> => {
  const headers = new Headers(init?.headers);

  if (init?.body !== undefined && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await apiFetch(path, { ...init, headers });

  return (await response.json()) as T;
};

export const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
