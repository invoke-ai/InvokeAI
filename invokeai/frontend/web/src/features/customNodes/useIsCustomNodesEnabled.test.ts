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
