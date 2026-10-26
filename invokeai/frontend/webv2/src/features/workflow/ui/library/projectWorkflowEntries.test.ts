import type { AuthSession } from '@features/identity';
import type { ProjectWorkflowSource } from '@features/workflow/core/types';

import { describe, expect, it } from 'vitest';

import { isUpdatableSource } from './projectWorkflowEntries';

const session = (overrides: Partial<AuthSession> = {}): AuthSession => ({
  accountEpoch: 1,
  multiuserEnabled: true,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  user: {
    created_at: '',
    display_name: null,
    email: '',
    is_active: true,
    is_admin: false,
    last_login_at: null,
    updated_at: '',
    user_id: 'user-a',
  },
  ...overrides,
});

const admin = session({
  user: {
    created_at: '',
    display_name: null,
    email: '',
    is_active: true,
    is_admin: true,
    last_login_at: null,
    updated_at: '',
    user_id: 'admin',
  },
});

const source = (overrides: Partial<ProjectWorkflowSource> = {}): ProjectWorkflowSource => ({
  libraryWorkflowId: 'library-1',
  ownerUserId: 'user-a',
  revision: 1,
  ...overrides,
});

describe('isUpdatableSource', () => {
  it('allows admins to update foreign saved records while keeping bundled, ordinary foreign, and unresolved sources blocked', () => {
    const foreignSource = source({ ownerUserId: 'user-b' });

    expect(isUpdatableSource(foreignSource, admin)).toBe(true);
    expect(isUpdatableSource(foreignSource, session())).toBe(false);
    expect(isUpdatableSource(source({ libraryWorkflowId: 'default_template' }), admin)).toBe(false);
    expect(isUpdatableSource(foreignSource, session({ ...admin, phase: 'unknown' }))).toBe(false);
  });
});
