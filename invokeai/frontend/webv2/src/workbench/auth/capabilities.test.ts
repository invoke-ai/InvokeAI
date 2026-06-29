import { describe, expect, it } from 'vitest';

import type { AuthSession } from './session';

import { getCapabilities } from './capabilities';

const createSession = (overrides: Partial<AuthSession>): AuthSession => ({
  multiuserEnabled: false,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  user: null,
  ...overrides,
});

describe('getCapabilities', () => {
  it('allows clearing intermediates in single-user mode', () => {
    expect(getCapabilities(createSession({ multiuserEnabled: false })).canClearIntermediates).toBe(true);
  });

  it('allows clearing intermediates for multi-user admins only', () => {
    expect(
      getCapabilities(
        createSession({
          multiuserEnabled: true,
          user: { is_admin: true } as AuthSession['user'],
        })
      ).canClearIntermediates
    ).toBe(true);

    expect(
      getCapabilities(
        createSession({
          multiuserEnabled: true,
          user: { is_admin: false } as AuthSession['user'],
        })
      ).canClearIntermediates
    ).toBe(false);
  });
});
