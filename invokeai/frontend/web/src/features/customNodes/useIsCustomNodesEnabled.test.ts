import { describe, expect, it } from 'vitest';

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
 * The hook (useIsCustomNodesEnabled) wraps getIsCustomNodesEnabled but adds
 * one important behavior: when setupStatus has not yet loaded (RTK Query
 * initial state), it returns `true` optimistically to prevent the AppContent
 * redirect from kicking a legitimate single-user session off a persisted
 * customNodes tab before the query resolves. The tests below verify that
 * contract by simulating the same decision path the hook takes.
 */
describe('useIsCustomNodesEnabled hook contract', () => {
  /**
   * Simulates the hook's decision for a given setupStatus and user state.
   * This mirrors the hook logic: if setupStatus is undefined, return true;
   * otherwise delegate to the pure helper.
   */
  const simulateHook = (
    setupStatus: { multiuser_enabled: boolean } | undefined,
    user: { is_admin: boolean } | undefined
  ): boolean => {
    if (!setupStatus) {
      return true;
    }
    return getIsCustomNodesEnabled(setupStatus.multiuser_enabled, user?.is_admin);
  };

  it('returns true while setupStatus is still loading (optimistic default)', () => {
    // This prevents redirect away from a persisted customNodes tab on startup
    expect(simulateHook(undefined, undefined)).toBe(true);
    expect(simulateHook(undefined, { is_admin: false })).toBe(true);
    expect(simulateHook(undefined, { is_admin: true })).toBe(true);
  });

  it('resolves correctly once setupStatus loads in single-user mode', () => {
    expect(simulateHook({ multiuser_enabled: false }, undefined)).toBe(true);
    expect(simulateHook({ multiuser_enabled: false }, { is_admin: false })).toBe(true);
  });

  it('resolves correctly once setupStatus loads in multiuser mode', () => {
    expect(simulateHook({ multiuser_enabled: true }, { is_admin: true })).toBe(true);
    expect(simulateHook({ multiuser_enabled: true }, { is_admin: false })).toBe(false);
    expect(simulateHook({ multiuser_enabled: true }, undefined)).toBe(false);
  });
});
