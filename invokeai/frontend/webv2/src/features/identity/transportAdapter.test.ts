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
    const adapter = createIdentityTransportAuthAdapter(testTokenAdapter, onUnauthorized);

    expect(adapter.getToken()).toBe('test-token');
    testTokenAdapter.set('next-token');
    expect(adapter.getToken()).toBe('next-token');
    adapter.onUnauthorized();
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });
});
