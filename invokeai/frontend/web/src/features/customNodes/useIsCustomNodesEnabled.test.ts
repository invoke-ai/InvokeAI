import { describe, expect, it } from 'vitest';

import type { CustomNodesPermission } from './useIsCustomNodesEnabled';
import { getIsCustomNodesEnabled } from './useIsCustomNodesEnabled';

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
 * Hook-level contract tests.
 *
 * The hook (useIsCustomNodesEnabled) returns { isKnown, isAllowed } so that
 * consumers can distinguish three states:
 *   loading   — isKnown=false, isAllowed=false → hide tab, do NOT redirect
 *   allowed   — isKnown=true,  isAllowed=true  → show tab, render content
 *   denied    — isKnown=true,  isAllowed=false  → hide tab, redirect away
 *
 * We simulate the hook's decision path here to verify the contract without
 * needing a full Redux/RTK Query harness.
 */
describe('useIsCustomNodesEnabled hook contract', () => {
  const simulateHook = (
    setupStatus: { multiuser_enabled: boolean } | undefined,
    user: { is_admin: boolean } | undefined
  ): CustomNodesPermission => {
    if (!setupStatus) {
      return { isKnown: false, isAllowed: false };
    }
    const isAllowed = getIsCustomNodesEnabled(setupStatus.multiuser_enabled, user?.is_admin);
    return { isKnown: true, isAllowed };
  };

  it('returns unknown/denied while setupStatus is still loading', () => {
    // Tab hidden, no redirect, no admin-only requests fired
    expect(simulateHook(undefined, undefined)).toEqual({ isKnown: false, isAllowed: false });
    expect(simulateHook(undefined, { is_admin: false })).toEqual({ isKnown: false, isAllowed: false });
    expect(simulateHook(undefined, { is_admin: true })).toEqual({ isKnown: false, isAllowed: false });
  });

  it('resolves to known/allowed in single-user mode', () => {
    expect(simulateHook({ multiuser_enabled: false }, undefined)).toEqual({ isKnown: true, isAllowed: true });
    expect(simulateHook({ multiuser_enabled: false }, { is_admin: false })).toEqual({
      isKnown: true,
      isAllowed: true,
    });
  });

  it('resolves to known/allowed for multiuser admin', () => {
    expect(simulateHook({ multiuser_enabled: true }, { is_admin: true })).toEqual({
      isKnown: true,
      isAllowed: true,
    });
  });

  it('resolves to known/denied for multiuser non-admin', () => {
    expect(simulateHook({ multiuser_enabled: true }, { is_admin: false })).toEqual({
      isKnown: true,
      isAllowed: false,
    });
    expect(simulateHook({ multiuser_enabled: true }, undefined)).toEqual({ isKnown: true, isAllowed: false });
  });

  it('non-admin multiuser user never sees isAllowed=true in any state', () => {
    // This is the regression test: during loading AND after resolution,
    // a non-admin in multiuser mode must never get isAllowed=true.
    const loading = simulateHook(undefined, { is_admin: false });
    const resolved = simulateHook({ multiuser_enabled: true }, { is_admin: false });
    expect(loading.isAllowed).toBe(false);
    expect(resolved.isAllowed).toBe(false);
  });
});
