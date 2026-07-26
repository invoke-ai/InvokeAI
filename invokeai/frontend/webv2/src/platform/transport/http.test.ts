import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./deploymentBase', () => ({
  getDeploymentBasePath: () => '/invoke',
  getDeploymentBaseUrl: () => 'https://app.example/invoke',
}));

import {
  absolutizeApiUrl,
  apiFetch,
  apiFetchJson,
  buildApiUrl,
  configureHttpAuth,
  getBackendSocketPath,
  getBackendSocketUrl,
} from './http';

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
    configureHttpAuth({ getIdentity: () => identity, getToken: () => token, onUnauthorized });

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
    configureHttpAuth({ getIdentity: () => identity, getToken: () => 'current-token', onUnauthorized });

    await expect(apiFetch('/api/v1/projects/')).rejects.toMatchObject({ status: 401 });

    expect(onUnauthorized).toHaveBeenCalledWith('current-token', identity);
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
    configureHttpAuth({ getIdentity: () => identity, getToken: () => token, onUnauthorized });

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
    configureHttpAuth({ getIdentity: () => identity, getToken: () => null, onUnauthorized: vi.fn() });

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
      json: () =>
        new Promise((resolve) => {
          resolveBody = resolve;
        }),
      ok: true,
      status: 200,
    } as Response;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
    configureHttpAuth({ getIdentity: () => identity, getToken: () => 'stable-token', onUnauthorized: vi.fn() });

    const oldRequest = apiFetchJson<{ owner: string }>('/api/v1/projects/');
    await vi.waitFor(() => {
      expect(resolveBody).toBeDefined();
    });

    identity = {};
    resolveBody?.({ owner: 'user-a' });

    await expect(oldRequest).rejects.toMatchObject({ name: 'HttpRequestIdentityExpiredError' });
  });
});
