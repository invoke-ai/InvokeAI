import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./deploymentBase', () => ({
  getDeploymentBasePath: () => '/invoke',
  getDeploymentBaseUrl: () => 'https://app.example/invoke',
}));

import {
  ApiError,
  absolutizeApiUrl,
  apiFetch,
  apiFetchJson,
  apiFetchRaw,
  buildApiUrl,
  configureHttpAuth,
  getBackendSocketPath,
  getBackendSocketUrl,
  type HttpAuthAdapter,
} from './http';

/** A transport-only adapter: Identity's acceptance policy is covered by its own tests. */
const configureTestAuth = (
  read: () => { identity: unknown; token: string | null },
  handlers: Partial<Pick<HttpAuthAdapter, 'onRefreshedToken' | 'onUnauthorized'>> = {}
): void => {
  configureHttpAuth({
    capture: read,
    onRefreshedToken: handlers.onRefreshedToken ?? vi.fn(),
    onUnauthorized: handlers.onUnauthorized ?? vi.fn(),
    subscribe: () => () => undefined,
  });
};

describe('deployment-aware backend URLs', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { location: { origin: 'https://app.example' } });
  });

  it('prefixes API requests with the deployment root', () => {
    expect(buildApiUrl('/api/v1/app/version')).toBe('https://app.example/invoke/api/v1/app/version');
  });

  it('prefixes backend-relative resource URLs and preserves absolute URLs', () => {
    expect(absolutizeApiUrl('/api/v1/images/i/image.png/full')).toBe(
      'https://app.example/invoke/api/v1/images/i/image.png/full'
    );
    expect(absolutizeApiUrl('https://cdn.example/image.png')).toBe('https://cdn.example/image.png');
  });

  it('uses the page origin with a deployment-prefixed Socket.IO path', () => {
    expect(getBackendSocketUrl()).toBe('https://app.example');
    expect(getBackendSocketPath()).toBe('/invoke/ws/socket.io');
  });
});

describe('request identity ownership', () => {
  it('preserves response headers on API errors for Retry-After handling', async () => {
    const identity = {};
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('{"detail":{"code":"project_write_busy"}}', {
          headers: { 'Retry-After': '1' },
          status: 429,
        })
      )
    );
    configureTestAuth(() => ({ identity, token: null }));

    const error = await apiFetch('/api/v1/projects/').catch((error: unknown) => error);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).headers.get('Retry-After')).toBe('1');
  });

  it('does not expire a newer account when an older request returns 401', async () => {
    let token: string | null = 'token-a';
    let identity = {};
    const onUnauthorized = vi.fn();
    let resolveFetch: ((response: Response) => void) | undefined;
    let sentInit: RequestInit | undefined;
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      sentInit = init;

      return new Promise<Response>((resolve) => {
        resolveFetch = resolve;
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    configureTestAuth(() => ({ identity, token }), { onUnauthorized });

    const oldRequest = apiFetch('/api/v1/projects/');
    const sentHeaders = sentInit?.headers as Headers;

    expect(sentHeaders.get('Authorization')).toBe('Bearer token-a');

    token = 'token-b';
    identity = {};
    resolveFetch?.(new Response('', { status: 401 }));

    await expect(oldRequest).rejects.toMatchObject({ name: 'HttpRequestIdentityExpiredError' });
    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it('expires the account whose token was rejected', async () => {
    const onUnauthorized = vi.fn();
    const identity = {};
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 401 })));
    configureTestAuth(() => ({ identity, token: 'current-token' }), { onUnauthorized });

    await expect(apiFetch('/api/v1/projects/')).rejects.toMatchObject({ status: 401 });

    expect(onUnauthorized).toHaveBeenCalledWith({ identity, token: 'current-token' });
  });

  it('does not expire a new epoch when the backend reuses the same token string', async () => {
    const token = 'stable-token';
    let identity = {};
    const onUnauthorized = vi.fn();
    let resolveFetch: ((response: Response) => void) | undefined;
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            resolveFetch = resolve;
          })
      )
    );
    configureTestAuth(() => ({ identity, token }), { onUnauthorized });

    const oldRequest = apiFetch('/api/v1/projects/');

    identity = {};
    resolveFetch?.(new Response('', { status: 401 }));

    await expect(oldRequest).rejects.toMatchObject({ name: 'HttpRequestIdentityExpiredError' });
    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it('rejects a tokenless error body that finishes under a newer identity lifetime', async () => {
    let identity = {};
    let resolveBody: ((value: string) => void) | undefined;
    const response = {
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      text: () =>
        new Promise<string>((resolve) => {
          resolveBody = resolve;
        }),
    } as Response;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
    configureTestAuth(() => ({ identity, token: null }));

    const oldRequest = apiFetch('/api/v1/auth/status');
    await vi.waitFor(() => {
      expect(resolveBody).toBeDefined();
    });

    identity = {};
    resolveBody?.('Unauthorized');

    await expect(oldRequest).rejects.toMatchObject({ name: 'HttpRequestIdentityExpiredError' });
  });

  it('rejects a response body that finishes under a newer identity lifetime', async () => {
    let identity = {};
    let resolveBody: ((value: unknown) => void) | undefined;
    const response = {
      headers: new Headers(),
      json: () =>
        new Promise((resolve) => {
          resolveBody = resolve;
        }),
      ok: true,
      status: 200,
    } as Response;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
    configureTestAuth(() => ({ identity, token: 'stable-token' }));

    const oldRequest = apiFetchJson<{ owner: string }>('/api/v1/projects/');
    await vi.waitFor(() => {
      expect(resolveBody).toBeDefined();
    });

    identity = {};
    resolveBody?.({ owner: 'user-a' });

    await expect(oldRequest).rejects.toMatchObject({ name: 'HttpRequestIdentityExpiredError' });
  });
});

describe('credential outcome reporting', () => {
  it('reports a replacement token with the credential that sent the request, from checked and raw fetches', async () => {
    const identity = {};
    const onRefreshedToken = vi.fn();
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(new Response('{}', { headers: { 'X-Refreshed-Token': 'token-b' }, status: 200 }))
    );
    vi.stubGlobal('fetch', fetchMock);
    configureTestAuth(() => ({ identity, token: 'token-a' }), { onRefreshedToken });

    await apiFetch('/api/v1/boards/', { method: 'POST' });
    await apiFetchRaw('/api/v1/images/upload', { method: 'POST' });

    expect(onRefreshedToken.mock.calls).toEqual([
      [{ identity, token: 'token-a' }, 'token-b'],
      [{ identity, token: 'token-a' }, 'token-b'],
    ]);
    expect(new Headers(fetchMock.mock.calls[0]?.[1]?.headers).get('Authorization')).toBe('Bearer token-a');
  });

  it('attributes nothing to the held credential when a request did not carry it', async () => {
    const identity = {};
    let token: string | null = null;
    const onRefreshedToken = vi.fn();
    const onUnauthorized = vi.fn();
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(new Response('', { headers: { 'X-Refreshed-Token': 'replacement' }, status: 401 }))
    );
    vi.stubGlobal('fetch', fetchMock);
    configureTestAuth(() => ({ identity, token }), { onRefreshedToken, onUnauthorized });

    await expect(apiFetch('/api/v1/auth/status')).rejects.toMatchObject({ status: 401 });
    token = 'held-token';
    // A lease released for an earlier account authenticates with that account's token, not the held one.
    await expect(
      apiFetch('/api/v1/intermediates/holds/lease', {
        headers: { Authorization: 'Bearer lease-token' },
        method: 'DELETE',
      })
    ).rejects.toMatchObject({ status: 401 });

    expect(new Headers(fetchMock.mock.calls[1]?.[1]?.headers).get('Authorization')).toBe('Bearer lease-token');
    expect(onRefreshedToken).not.toHaveBeenCalled();
    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it('never reports a replacement that arrives after its identity lifetime ended', async () => {
    let identity = {};
    const onRefreshedToken = vi.fn();
    let resolveFetch: ((response: Response) => void) | undefined;
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            resolveFetch = resolve;
          })
      )
    );
    configureTestAuth(() => ({ identity, token: 'token-a' }), { onRefreshedToken });

    const lateRequest = apiFetchRaw('/api/v1/boards/', { method: 'POST' });

    identity = {};
    resolveFetch?.(new Response('{}', { headers: { 'X-Refreshed-Token': 'token-a-renewed' }, status: 200 }));

    await expect(lateRequest).rejects.toMatchObject({ name: 'HttpRequestIdentityExpiredError' });
    expect(onRefreshedToken).not.toHaveBeenCalled();
  });
});

describe('transport diagnostics', () => {
  beforeEach(() => {
    const identity = {};
    configureTestAuth(() => ({ identity, token: null }));
    resetLogging();
    configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'debug' });
  });

  it('records failed responses and network failures as debug breadcrumbs without query strings', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response('{"detail":"nope"}', { status: 503 }))
        .mockRejectedValueOnce(new TypeError('Failed to fetch'))
    );

    await expect(apiFetch('/api/v1/images?token=abc', { method: 'POST' })).rejects.toBeInstanceOf(ApiError);
    await expect(apiFetch('/api/v1/boards')).rejects.toBeInstanceOf(TypeError);

    expect(getLogSnapshot().entries).toMatchObject([
      {
        error: { message: 'Failed to fetch', name: 'TypeError' },
        level: 'debug',
        name: 'http.request-failed',
        source: { area: 'http', namespace: 'transport' },
      },
      {
        context: { method: 'POST', path: '/api/v1/images', status: 503 },
        level: 'debug',
        message: 'POST /api/v1/images responded 503',
        name: 'http.response-error',
      },
    ]);
  });

  it('drops breadcrumbs from requests whose account lifetime ended and ignores aborts', async () => {
    accountLifecycle.activate('user-a');
    let settle: (response: Response) => void = () => undefined;

    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockImplementationOnce(
          () =>
            new Promise<Response>((resolve) => {
              settle = resolve;
            })
        )
        .mockRejectedValueOnce(new DOMException('aborted', 'AbortError'))
    );

    const late = apiFetch('/api/v1/boards');

    accountLifecycle.invalidate();
    configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'debug' });
    settle(new Response('', { status: 500 }));
    await expect(late).rejects.toBeInstanceOf(ApiError);
    await expect(apiFetch('/api/v1/images')).rejects.toBeInstanceOf(DOMException);

    expect(getLogSnapshot().entries).toEqual([]);
  });

  it('records nothing for successful requests', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 200 })));

    await apiFetch('/api/v1/app/version');

    expect(getLogSnapshot().entries).toEqual([]);
  });
});

describe('media cookie credentials', () => {
  beforeEach(() => {
    // Keep the identity object stable; a fresh object would simulate rotation before the assertion.
    const identity = {};
    configureTestAuth(() => ({ identity, token: null }));
  });

  it('sends credentials so login can set the media cookie that authenticates <img> requests', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await apiFetch('/api/v1/auth/login', { method: 'POST' });

    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ credentials: 'same-origin' });
  });

  it('lets a caller override credentials for a cross-origin API base', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await apiFetch('/api/v1/auth/media-cookie', { credentials: 'include', method: 'POST' });

    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ credentials: 'include' });
  });
});
