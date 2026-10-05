import type { AccountScope } from '@platform/state/accountLifecycle';

import { apiFetch, configureHttpAuth } from '@platform/transport/http';
import { describe, expect, it, vi } from 'vitest';

import type { IdentityTokenAdapter } from './core/tokenStorage';

import { updateCurrentUser } from './data/api';
import { createIdentityTransportAuthAdapter } from './transportAdapter';

describe('Identity transport auth adapter', () => {
  it('keeps the password replacement when an old-token 401 arrives first', async () => {
    let token: string | null = 'old-token';
    let identity: AccountScope = {
      accountId: 'user',
      epoch: 1,
      signal: new AbortController().signal,
      storageSuffix: ':user:user',
    };
    const onUnauthorized = vi.fn(() => {
      token = null;
      identity = { ...identity, epoch: 2 };
    });
    const adapter = createIdentityTransportAuthAdapter(
      {
        clear: () => {
          token = null;
        },
        get: () => token,
        set: (value) => {
          token = value;
        },
      },
      onUnauthorized,
      { capture: () => identity, isCurrent: (candidate) => candidate === identity }
    );
    configureHttpAuth(adapter);
    let finishPasswordChange!: (response: Response) => void;
    const passwordResponse = new Promise<Response>((resolve) => {
      finishPasswordChange = resolve;
    });
    let finishStaleRequest!: (response: Response) => void;
    const staleResponse = new Promise<Response>((resolve) => {
      finishStaleRequest = resolve;
    });
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = input.toString();

      if (url.endsWith('/auth/me')) {
        return passwordResponse;
      }
      if (url.endsWith('/images/i/example.png')) {
        return staleResponse;
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    try {
      const change = updateCurrentUser({ current_password: 'old', new_password: 'new' });
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
      const stale = apiFetch('/api/v1/images/i/example.png');
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

      finishStaleRequest(new Response('', { status: 401 }));
      await expect(stale).rejects.toMatchObject({ status: 401 });
      expect(onUnauthorized).not.toHaveBeenCalled();

      finishPasswordChange(
        new Response('{"user_id":"user"}', {
          headers: { 'X-Refreshed-Token': 'new-token' },
        })
      );
      await change;

      expect(token).toBe('new-token');
      expect(onUnauthorized).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('accepts a replacement only for the token and account lifetime that sent the request', () => {
    let token = 'request-token';
    let identity: AccountScope = {
      accountId: 'a',
      epoch: 1,
      signal: new AbortController().signal,
      storageSuffix: ':user:a',
    };
    const adapter = createIdentityTransportAuthAdapter(
      {
        clear: () => {
          token = '';
        },
        get: () => token,
        set: (value) => {
          token = value;
        },
      },
      vi.fn(),
      { capture: () => identity, isCurrent: (candidate) => candidate === identity }
    );

    adapter.onRefreshedToken('replacement', 'request-token', identity);
    expect(token).toBe('replacement');

    adapter.onRefreshedToken('stale', 'request-token', identity);
    expect(token).toBe('replacement');

    const oldIdentity = identity;
    identity = { ...identity, epoch: 2 };
    token = 'request-token';
    adapter.onRefreshedToken('old-account', 'request-token', oldIdentity);
    expect(token).toBe('request-token');
  });

  it('accepts a newer epoch after a same-epoch refresh, but never restores an older epoch', () => {
    const makeToken = (epoch: number, nonce: string): string =>
      `header.${btoa(JSON.stringify({ token_epoch: epoch, user_id: 'user', nonce }))}.signature`;
    const requestToken = makeToken(1, 'request');
    let token = makeToken(1, 'routine-refresh');
    const identity: AccountScope = {
      accountId: 'user',
      epoch: 1,
      signal: new AbortController().signal,
      storageSuffix: ':user:user',
    };
    const adapter = createIdentityTransportAuthAdapter(
      {
        clear: () => undefined,
        get: () => token,
        set: (value) => {
          token = value;
        },
      },
      vi.fn(),
      { capture: () => identity, isCurrent: (candidate) => candidate === identity }
    );
    const replacement = makeToken(2, 'password-change');

    adapter.onRefreshedToken(replacement, requestToken, identity);
    expect(token).toBe(replacement);

    adapter.onRefreshedToken(makeToken(1, 'late-routine-refresh'), requestToken, identity);
    expect(token).toBe(replacement);
  });

  it('expires a still-current token when a password change fails after a deferred 401', async () => {
    let token: string | null = 'old-token';
    const identity: AccountScope = {
      accountId: 'user',
      epoch: 1,
      signal: new AbortController().signal,
      storageSuffix: ':user:user',
    };
    const onUnauthorized = vi.fn();
    const adapter = createIdentityTransportAuthAdapter(
      {
        clear: () => {
          token = null;
        },
        get: () => token,
        set: (value) => {
          token = value;
        },
      },
      onUnauthorized,
      { capture: () => identity, isCurrent: (candidate) => candidate === identity }
    );
    configureHttpAuth(adapter);
    let finishPasswordChange!: (response: Response) => void;
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            finishPasswordChange = resolve;
          })
      )
    );

    try {
      const change = updateCurrentUser({ current_password: 'old', new_password: 'new' });
      adapter.onUnauthorized('old-token', identity);
      expect(onUnauthorized).not.toHaveBeenCalled();

      finishPasswordChange(new Response('bad request', { status: 400 }));
      await expect(change).rejects.toMatchObject({ status: 400 });
      await vi.waitFor(() => expect(onUnauthorized).toHaveBeenCalledOnce());
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('gives browser and test token stores the same transport contract', () => {
    let token: string | null = 'test-token';
    const testTokenAdapter: IdentityTokenAdapter = {
      clear: () => {
        token = null;
      },
      get: () => token,
      set: (value) => {
        token = value;
      },
    };
    const onUnauthorized = vi.fn();
    let identity: AccountScope = {
      accountId: 'a',
      epoch: 1,
      signal: new AbortController().signal,
      storageSuffix: ':user:a',
    };
    const adapter = createIdentityTransportAuthAdapter(testTokenAdapter, onUnauthorized, {
      capture: () => identity,
      isCurrent: (candidate) => candidate === identity,
    });

    expect(adapter.getToken()).toBe('test-token');
    expect(adapter.getIdentity()).toBe(identity);
    testTokenAdapter.set('next-token');
    expect(adapter.getToken()).toBe('next-token');
    adapter.onUnauthorized('next-token', identity);
    expect(onUnauthorized).toHaveBeenCalledOnce();

    testTokenAdapter.set('newer-token');
    adapter.onUnauthorized('next-token', identity);
    expect(onUnauthorized).toHaveBeenCalledOnce();

    testTokenAdapter.set('next-token');
    const oldIdentity = identity;
    identity = {
      accountId: 'a',
      epoch: 2,
      signal: new AbortController().signal,
      storageSuffix: ':user:a',
    };
    adapter.onUnauthorized('next-token', oldIdentity);
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });
});
