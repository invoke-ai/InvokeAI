/**
 * Each request captures the Identity-owned credential once and reports its outcome (a replacement token, a 401)
 * against that capture, so a late response can never renew or expire a newer credential.
 */

import { recordLogEvent } from '@platform/logging/logger';
import { captureAccountScope } from '@platform/state/accountLifecycle';

import { getDeploymentBasePath, getDeploymentBaseUrl } from './deploymentBase';

const API_BASE_URL = (import.meta.env.VITE_INVOKEAI_API_BASE_URL ?? '').trim().replace(/\/$/, '');

/** Sliding-expiry renewals and own-password-change replacements arrive in this response header. */
export const REFRESHED_TOKEN_HEADER = 'X-Refreshed-Token';

export interface HttpCredential {
  /** Opaque identity lifetime; object identity rotates on reauthentication. */
  readonly identity: unknown;
  readonly token: string | null;
}

export interface HttpAuthAdapter {
  /** One synchronous read, so a request never pairs a token with a different identity lifetime. */
  capture(): HttpCredential;
  /** A response to a request sent with `credential` carried a replacement token. */
  onRefreshedToken(credential: HttpCredential, token: string): void;
  /** A request sent with `credential` was rejected with 401. */
  onUnauthorized(credential: HttpCredential): void;
  /** Notifies when the token of the current identity lifetime is replaced in place. */
  subscribe(listener: () => void): () => void;
}

/** Transport failures are breadcrumbs; the caller that handles the outcome owns the terminal report. */
const HTTP_LOG_SOURCE = { area: 'http', namespace: 'transport' } as const;

let authAdapter: HttpAuthAdapter = {
  capture: () => ({ identity: null, token: null }),
  onRefreshedToken: () => undefined,
  onUnauthorized: () => undefined,
  subscribe: () => () => undefined,
};

/** Selected once by the App composition root; Platform owns no Auth policy. */
export const configureHttpAuth = (adapter: HttpAuthAdapter): void => {
  authAdapter = adapter;
};

export const getHttpAuthToken = (): string | null => authAdapter.capture().token;

export const subscribeHttpCredential = (listener: () => void): (() => void) => authAdapter.subscribe(listener);

export const getBackendSocketUrl = (): string => {
  const baseUrl = API_BASE_URL || getDeploymentBaseUrl();

  return new URL(baseUrl, getDeploymentBaseUrl()).origin;
};

export const getBackendSocketPath = (): string => {
  if (!API_BASE_URL) {
    return `${getDeploymentBasePath()}/ws/socket.io`;
  }

  const pathname = new URL(API_BASE_URL, getDeploymentBaseUrl()).pathname.replace(/\/$/, '');

  return `${pathname === '/' ? '' : pathname}/ws/socket.io`;
};

export const buildApiUrl = (path: string): string => `${API_BASE_URL || getDeploymentBaseUrl()}${path}`;

/** Resolve a backend-relative resource URL (e.g. image URLs in DTOs) against the API host. */
export const absolutizeApiUrl = (url: string): string => {
  try {
    const parsed = new URL(url);

    if (parsed.protocol) {
      return url;
    }
  } catch {
    // Relative URL — resolve it against the backend deployment root below.
  }

  return url.startsWith('/') ? buildApiUrl(url) : new URL(url, `${API_BASE_URL || getDeploymentBaseUrl()}/`).toString();
};

export class ApiError extends Error {
  readonly headers: Headers;
  readonly status: number;

  constructor(message: string, status: number, headers?: HeadersInit) {
    super(message);
    this.name = 'ApiError';
    this.headers = new Headers(headers);
    this.status = status;
  }
}

export class HttpRequestIdentityExpiredError extends Error {
  constructor() {
    super('The identity lifetime that started this HTTP request is no longer active.');
    this.name = 'HttpRequestIdentityExpiredError';
  }
}

const assertHttpIdentityCurrent = (identity: unknown): void => {
  if (authAdapter.capture().identity !== identity) {
    throw new HttpRequestIdentityExpiredError();
  }
};

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
  throw new ApiError(text || `${response.status} ${response.statusText}`, response.status, response.headers);
};

interface AuthenticatedResponse {
  /** The identity lifetime the request was captured in. */
  identity: unknown;
  response: Response;
  /** The captured credential when it authenticated the request; null for anonymous or caller-authorized requests. */
  sent: HttpCredential | null;
}

/** Throws if the identity lifetime ended before the response arrived; reports a replacement token otherwise. */
const fetchAuthenticated = async (path: string, init: RequestInit | undefined): Promise<AuthenticatedResponse> => {
  const credential = authAdapter.capture();
  const headers = new Headers(init?.headers);
  const method = (init?.method ?? 'GET').toUpperCase();
  // Breadcrumbs carry the path only; query strings can hold tokens and user input. They are fenced to the account
  // that started the request so late settlements never land in the next account's history.
  const safePath = path.split(/[?#]/, 1)[0] ?? path;
  const owner = captureAccountScope();
  // A caller-supplied Authorization header (a lease released for an earlier account) is not this tab's credential,
  // so its response reports nothing about it.
  const sent = credential.token !== null && !headers.has('Authorization') ? credential : null;

  if (sent) {
    headers.set('Authorization', `Bearer ${sent.token}`);
  }

  // Use same-origin credentials so login stores the media cookie. Cross-origin API callers must opt into include
  // and configure server credential support.
  let response: Response;

  try {
    response = await fetch(buildApiUrl(path), { credentials: 'same-origin', ...init, headers });
  } catch (error) {
    // Aborts are expected cancellations (account cleanup, superseded queries), not failures.
    if (!(error instanceof DOMException && error.name === 'AbortError')) {
      recordLogEvent(
        'debug',
        HTTP_LOG_SOURCE,
        {
          context: { method, path: safePath },
          error,
          message: `${method} ${safePath} failed`,
          name: 'http.request-failed',
        },
        owner
      );
    }
    throw error;
  }

  if (!response.ok) {
    recordLogEvent(
      'debug',
      HTTP_LOG_SOURCE,
      {
        context: { method, path: safePath, status: response.status },
        message: `${method} ${safePath} responded ${response.status}`,
        name: 'http.response-error',
      },
      owner
    );
  }

  assertHttpIdentityCurrent(credential.identity);

  const refreshedToken = sent ? response.headers.get(REFRESHED_TOKEN_HEADER) : null;

  if (sent && refreshedToken) {
    authAdapter.onRefreshedToken(sent, refreshedToken);
  }

  return { identity: credential.identity, response, sent };
};

/** Authenticated fetch that leaves status handling, including 401s, to the caller. */
export const apiFetchRaw = async (path: string, init?: RequestInit): Promise<Response> =>
  (await fetchAuthenticated(path, init)).response;

export const apiFetch = async (path: string, init?: RequestInit): Promise<Response> => {
  const { identity, response, sent } = await fetchAuthenticated(path, init);
  const reportedUnauthorized = response.status === 401 && sent !== null;

  if (reportedUnauthorized) {
    authAdapter.onUnauthorized(sent);
  }

  try {
    const asserted = await assertOk(response);

    assertHttpIdentityCurrent(identity);

    return asserted;
  } catch (error) {
    // Preserve the original ApiError even when 401 handling rotates identity.
    if (!reportedUnauthorized) {
      assertHttpIdentityCurrent(identity);
    }

    throw error;
  }
};

/** Unwrap FastAPI's detail field from ApiError's raw body for display. */
export const getApiErrorMessage = (error: unknown, fallback: string): string => {
  if (error instanceof ApiError) {
    try {
      const parsed = JSON.parse(error.message) as { detail?: unknown };

      if (typeof parsed.detail === 'string' && parsed.detail) {
        return parsed.detail;
      }

      if (parsed.detail && typeof parsed.detail === 'object') {
        const message = (parsed.detail as { message?: unknown }).message;

        if (typeof message === 'string' && message) {
          return message;
        }
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
  const requestIdentity = authAdapter.capture().identity;
  const headers = new Headers(init?.headers);

  if (init?.body !== undefined && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await apiFetch(path, { ...init, headers });
  const body = (await response.json()) as T;

  assertHttpIdentityCurrent(requestIdentity);

  return body;
};

export const sleep = (ms: number, signal?: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }

    const onAbort = (): void => {
      globalThis.clearTimeout(timeoutId);
      reject(signal?.reason);
    };
    const timeoutId = globalThis.setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);

    signal?.addEventListener('abort', onAbort, { once: true });
  });
