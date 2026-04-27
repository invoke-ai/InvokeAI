import { describe, expect, it } from 'vitest';

import { deriveCustomNodesPermission, getIsCustomNodesEnabled } from './useIsCustomNodesEnabled';

describe('getIsCustomNodesEnabled', () => {
  it('returns true in single-user mode regardless of admin status', () => {
    expect(getIsCustomNodesEnabled(false, false)).toBe(true);
    expect(getIsCustomNodesEnabled(false, true)).toBe(true);
    expect(getIsCustomNodesEnabled(false, undefined)).toBe(true);
  });

  it('returns true in multiuser mode for admin users', () => {
    expect(getIsCustomNodesEnabled(true, true)).toBe(true);
  });

  it('returns false in multiuser mode for non-admin users', () => {
    expect(getIsCustomNodesEnabled(true, false)).toBe(false);
  });

  it('returns false in multiuser mode when user is not yet loaded', () => {
    expect(getIsCustomNodesEnabled(true, undefined)).toBe(false);
  });
});

/**
 * Permission-state tests.
 *
 * These call deriveCustomNodesPermission directly — the same function the hook
 * uses internally — so the test and the hook can never drift. The contract is:
 *   loading (setupStatus undefined) -> { isKnown: false, isAllowed: false }
 *   resolved                        -> { isKnown: true,  isAllowed: getIsCustomNodesEnabled(...) }
 *
 * Consumers read this state:
 *   VerticalNavBar   shows tab only when isAllowed
 *   AppContent       redirects only when isKnown && !isAllowed
 */
describe('deriveCustomNodesPermission', () => {
  it('returns unknown/denied while setupStatus is still loading', () => {
    expect(deriveCustomNodesPermission(undefined, undefined)).toEqual({ isKnown: false, isAllowed: false });
    expect(deriveCustomNodesPermission(undefined, null)).toEqual({ isKnown: false, isAllowed: false });
    expect(deriveCustomNodesPermission(undefined, { is_admin: false })).toEqual({ isKnown: false, isAllowed: false });
    expect(deriveCustomNodesPermission(undefined, { is_admin: true })).toEqual({ isKnown: false, isAllowed: false });
  });

  it('resolves to known/allowed in single-user mode regardless of user', () => {
    expect(deriveCustomNodesPermission({ multiuser_enabled: false }, undefined)).toEqual({
      isKnown: true,
      isAllowed: true,
    });
    expect(deriveCustomNodesPermission({ multiuser_enabled: false }, null)).toEqual({
      isKnown: true,
      isAllowed: true,
    });
    expect(deriveCustomNodesPermission({ multiuser_enabled: false }, { is_admin: false })).toEqual({
      isKnown: true,
      isAllowed: true,
    });
  });

  it('resolves to known/allowed for multiuser admin', () => {
    expect(deriveCustomNodesPermission({ multiuser_enabled: true }, { is_admin: true })).toEqual({
      isKnown: true,
      isAllowed: true,
    });
  });

  it('resolves to known/denied for multiuser non-admin or missing user', () => {
    expect(deriveCustomNodesPermission({ multiuser_enabled: true }, { is_admin: false })).toEqual({
      isKnown: true,
      isAllowed: false,
    });
    expect(deriveCustomNodesPermission({ multiuser_enabled: true }, null)).toEqual({
      isKnown: true,
      isAllowed: false,
    });
    expect(deriveCustomNodesPermission({ multiuser_enabled: true }, undefined)).toEqual({
      isKnown: true,
      isAllowed: false,
    });
  });

  it('non-admin multiuser user never sees isAllowed=true in any state', () => {
    // Regression: during loading AND after resolution, a non-admin in multiuser
    // mode must never get isAllowed=true, so the tab never renders content.
    expect(deriveCustomNodesPermission(undefined, { is_admin: false }).isAllowed).toBe(false);
    expect(deriveCustomNodesPermission({ multiuser_enabled: true }, { is_admin: false }).isAllowed).toBe(false);
  });
});
