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

export const apiFetch = async (path: string, init?: RequestInit): Promise<Response> =>
  assertOk(await apiFetchRaw(path, init));

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
