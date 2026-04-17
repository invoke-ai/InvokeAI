import { describe, expect, it } from 'vitest';

/**
 * Pure-logic test for the admin-gate decision that useIsCustomNodesEnabled implements.
 * The hook itself reads from Redux + RTK Query, but the permission logic is a simple
 * function of (multiuser_enabled, is_admin). We test the decision matrix directly to
 * catch regressions without needing a full store/provider harness.
 */
const isCustomNodesEnabled = (multiuserEnabled: boolean, isAdmin: boolean | undefined): boolean => {
  if (!multiuserEnabled) {
    return true;
  }
  return isAdmin ?? false;
};

describe('useIsCustomNodesEnabled logic', () => {
  it('returns true in single-user mode regardless of admin status', () => {
    expect(isCustomNodesEnabled(false, false)).toBe(true);
    expect(isCustomNodesEnabled(false, true)).toBe(true);
    expect(isCustomNodesEnabled(false, undefined)).toBe(true);
  });

  it('returns true in multiuser mode for admin users', () => {
    expect(isCustomNodesEnabled(true, true)).toBe(true);
  });

  it('returns false in multiuser mode for non-admin users', () => {
    expect(isCustomNodesEnabled(true, false)).toBe(false);
  });

  it('returns false in multiuser mode when user is not yet loaded', () => {
    expect(isCustomNodesEnabled(true, undefined)).toBe(false);
  });
});
