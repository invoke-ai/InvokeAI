import { getIsAdmin } from 'features/auth/hooks/useIsAdmin';
import { describe, expect, it } from 'vitest';

import {
  getIsClearInvocationCacheDisabled,
  getIsDisableInvocationCacheDisabled,
  getIsEnableInvocationCacheDisabled,
} from './invocationCacheControls';

const ENABLED_CACHE = { enabled: true, size: 4, hits: 1, misses: 1, max_size: 20 };
const DISABLED_CACHE = { enabled: false, size: 0, hits: 0, misses: 0, max_size: 20 };

describe('invocation cache controls', () => {
  describe('non-admin in multiuser mode', () => {
    // The status route is admin-only, so a non-admin's cacheStatus is always undefined. Without the
    // admin check that reads as "cache is disabled" and offers an Enable button that always 403s.
    const isAdmin = getIsAdmin({ multiuser_enabled: true }, { is_admin: false });

    it('is not admin', () => {
      expect(isAdmin).toBe(false);
    });

    it('offers no cache mutation while connected with no status data', () => {
      expect(getIsEnableInvocationCacheDisabled(isAdmin, true, undefined)).toBe(true);
      expect(getIsDisableInvocationCacheDisabled(isAdmin, true, undefined)).toBe(true);
      expect(getIsClearInvocationCacheDisabled(isAdmin, true, undefined)).toBe(true);
    });

    it('offers no cache mutation even if status data is somehow present', () => {
      for (const status of [ENABLED_CACHE, DISABLED_CACHE]) {
        expect(getIsEnableInvocationCacheDisabled(isAdmin, true, status)).toBe(true);
        expect(getIsDisableInvocationCacheDisabled(isAdmin, true, status)).toBe(true);
        expect(getIsClearInvocationCacheDisabled(isAdmin, true, status)).toBe(true);
      }
    });
  });

  describe('admin', () => {
    it('treats the multiuser admin and the single-user operator alike', () => {
      expect(getIsAdmin({ multiuser_enabled: true }, { is_admin: true })).toBe(true);
      expect(getIsAdmin({ multiuser_enabled: false }, { is_admin: false })).toBe(true);
    });

    it('preserves the pre-existing enable/disable/clear semantics', () => {
      // Enable is offered only when the cache is off and has capacity.
      expect(getIsEnableInvocationCacheDisabled(true, true, DISABLED_CACHE)).toBe(false);
      expect(getIsEnableInvocationCacheDisabled(true, true, ENABLED_CACHE)).toBe(true);
      expect(getIsEnableInvocationCacheDisabled(true, true, { ...DISABLED_CACHE, max_size: 0 })).toBe(true);

      // Disable is offered only when the cache is on.
      expect(getIsDisableInvocationCacheDisabled(true, true, ENABLED_CACHE)).toBe(false);
      expect(getIsDisableInvocationCacheDisabled(true, true, DISABLED_CACHE)).toBe(true);

      // Clear is offered only when there is something cached.
      expect(getIsClearInvocationCacheDisabled(true, true, ENABLED_CACHE)).toBe(false);
      expect(getIsClearInvocationCacheDisabled(true, true, DISABLED_CACHE)).toBe(true);
    });

    it('offers nothing while disconnected', () => {
      expect(getIsEnableInvocationCacheDisabled(true, false, DISABLED_CACHE)).toBe(true);
      expect(getIsDisableInvocationCacheDisabled(true, false, ENABLED_CACHE)).toBe(true);
      expect(getIsClearInvocationCacheDisabled(true, false, ENABLED_CACHE)).toBe(true);
    });
  });
});
