import { describe, expect, it } from 'vitest';

import type { AuthSession } from './session';

import { getCapabilities } from './capabilities';

const session = (overrides: Partial<AuthSession>): AuthSession => ({
  multiuserEnabled: true,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  user: null,
  ...overrides,
});

describe('Identity route capabilities', () => {
  it('allows all local management in single-user mode', () => {
    expect(getCapabilities(session({ multiuserEnabled: false }))).toEqual({
      canManageModels: true,
      canManageNodes: true,
      canManageUsers: false,
    });
  });

  it('limits multi-user administration to admins', () => {
    const baseUser = {
      created_at: '',
      display_name: null,
      email: 'user@example.com',
      is_active: true,
      last_login_at: null,
      updated_at: '',
      user_id: 'user',
    };

    expect(getCapabilities(session({ user: { ...baseUser, is_admin: false } }))).toEqual({
      canManageModels: false,
      canManageNodes: false,
      canManageUsers: false,
    });
    expect(getCapabilities(session({ user: { ...baseUser, is_admin: true } }))).toEqual({
      canManageModels: true,
      canManageNodes: true,
      canManageUsers: true,
    });
  });
});
