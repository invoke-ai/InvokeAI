import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./deploymentBase', () => ({
  getDeploymentBasePath: () => '/invoke',
  getDeploymentBaseUrl: () => 'https://app.example/invoke',
}));

import {
  ApiError,
  RequestDeadlineError,
  withRequestDeadline,
  apiFetch,
  apiFetchJson,
  apiFetchRaw,
  buildApiUrl,
  configureHttpAuth,
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
});

describe('request identity ownership', () => {
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
});

describe('request deadline cancellation', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    configureTestAuth(() => ({ identity: null, token: null }));
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it.each(['headers', 'body', 'error-body'] as const)('aborts a hung %s and disposes the deadline', async (phase) => {
    let signal!: AbortSignal;
    vi.stubGlobal(
      'fetch',
      vi.fn((_url, init) => {
        signal = init.signal;
        if (phase === 'headers') {
          return new Promise(() => {
            /* Simulate an unresponsive request. */
          });
        }
        return Promise.resolve(new Response(new ReadableStream(), { status: phase === 'body' ? 200 : 500 }));
      })
    );
    const result = expect(
      withRequestDeadline((signal) => apiFetchJson('/hung', { signal }), 100)
    ).rejects.toBeInstanceOf(RequestDeadlineError);
    await vi.advanceTimersByTimeAsync(100);
    await result;
    expect(signal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each(['caller', 'account'] as const)('cancels underlying work immediately on %s abort', async (source) => {
    const caller = new AbortController();
    let requestSignal!: AbortSignal;
    const result = withRequestDeadline(
      (signal) => {
        requestSignal = signal;
        return new Promise(() => {
          /* Simulate an unresponsive request. */
        });
      },
      100,
      caller.signal
    );
    const rejected = expect(result).rejects.toMatchObject({ name: 'AbortError' });
    if (source === 'caller') {
      caller.abort();
    } else {
      accountLifecycle.invalidate();
    }
    await rejected;
    expect(requestSignal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });
});
