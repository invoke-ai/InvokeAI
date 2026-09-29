import { describe, expect, it } from 'vitest';

import { getIsAdmin } from './useIsAdmin';

describe('getIsAdmin', () => {
  it('treats single-user mode as admin regardless of user', () => {
    expect(getIsAdmin({ multiuser_enabled: false }, undefined)).toBe(true);
    expect(getIsAdmin({ multiuser_enabled: false }, null)).toBe(true);
    expect(getIsAdmin({ multiuser_enabled: false }, { is_admin: false })).toBe(true);
  });

  it('treats the multiuser admin and the single-user operator alike', () => {
    expect(getIsAdmin({ multiuser_enabled: true }, { is_admin: true })).toBe(true);
    expect(getIsAdmin({ multiuser_enabled: false }, { is_admin: false })).toBe(true);
  });

  it('denies multiuser non-admin or missing user', () => {
    expect(getIsAdmin({ multiuser_enabled: true }, { is_admin: false })).toBe(false);
    expect(getIsAdmin({ multiuser_enabled: true }, null)).toBe(false);
    expect(getIsAdmin({ multiuser_enabled: true }, undefined)).toBe(false);
  });

  it('denies while setup status is still loading, unless the user is a known admin', () => {
    // Conservative during the query window: admin surfaces stay hidden rather than flashing in.
    expect(getIsAdmin(undefined, undefined)).toBe(false);
    expect(getIsAdmin(undefined, null)).toBe(false);
    expect(getIsAdmin(undefined, { is_admin: false })).toBe(false);
    expect(getIsAdmin(undefined, { is_admin: true })).toBe(true);
  });
});
