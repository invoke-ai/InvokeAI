import type { components } from 'services/api/schema';

type InvocationCacheStatus = components['schemas']['InvocationCacheStatus'];

/**
 * Disabled-state predicates for the invocation cache controls.
 *
 * The invocation cache routes are administrator-only on the backend, so a non-admin must never be
 * offered any of these mutations: `cacheStatus` is undefined for them (the status route 403s), which
 * on its own reads as "cache disabled" and would present an active Enable button.
 *
 * These are pure so they can be unit-tested without rendering.
 */

export const getIsEnableInvocationCacheDisabled = (
  isAdmin: boolean,
  isConnected: boolean,
  cacheStatus: InvocationCacheStatus | undefined
): boolean => !isAdmin || !isConnected || !!cacheStatus?.enabled || cacheStatus?.max_size === 0;

export const getIsDisableInvocationCacheDisabled = (
  isAdmin: boolean,
  isConnected: boolean,
  cacheStatus: InvocationCacheStatus | undefined
): boolean => !isAdmin || !isConnected || !cacheStatus?.enabled || cacheStatus?.max_size === 0;

export const getIsClearInvocationCacheDisabled = (
  isAdmin: boolean,
  isConnected: boolean,
  cacheStatus: InvocationCacheStatus | undefined
): boolean => !isAdmin || !isConnected || !cacheStatus?.size;
