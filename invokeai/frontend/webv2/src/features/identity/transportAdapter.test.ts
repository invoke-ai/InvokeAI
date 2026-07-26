import type { AccountScope } from '@platform/state/accountLifecycle';

import { describe, expect, it, vi } from 'vitest';

import type { IdentityTokenAdapter } from './core/tokenStorage';

import { createIdentityTransportAuthAdapter } from './transportAdapter';

describe('Identity transport auth adapter', () => {
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
