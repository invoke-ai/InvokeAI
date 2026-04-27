import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useMemo } from 'react';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';

/**
 * Pure decision function: determines whether custom node management is enabled.
 *
 * Returns true if:
 * - Multiuser mode is disabled (single-user mode = always admin)
 * - Multiuser mode is enabled AND user is an admin
 *
 * Returns false if:
 * - Multiuser mode is enabled AND user is not an admin
 */
export const getIsCustomNodesEnabled = (multiuserEnabled: boolean, isAdmin: boolean | undefined): boolean => {
  if (!multiuserEnabled) {
    return true;
  }
  return isAdmin ?? false;
};

type CustomNodesPermission = {
  /** Whether setup status has loaded and a permission decision can be made. */
  isKnown: boolean;
  /** Whether the current user is allowed to access custom node management.
   *  Only meaningful when isKnown is true; defaults to false while loading. */
  isAllowed: boolean;
};

/** Minimal shapes the derivation needs — matches the runtime types from auth slice + RTK Query. */
type SetupStatusLike = { multiuser_enabled: boolean } | undefined;
type UserLike = { is_admin: boolean } | null | undefined;

/**
 * Pure derivation of the permission state from the raw inputs the hook reads.
 * Both the hook and the tests consume this directly so the two can never drift.
 *
 * - loading  (setupStatus undefined) -> { isKnown: false, isAllowed: false }
 * - resolved (setupStatus defined)   -> { isKnown: true,  isAllowed: getIsCustomNodesEnabled(...) }
 */
export const deriveCustomNodesPermission = (setupStatus: SetupStatusLike, user: UserLike): CustomNodesPermission => {
  if (!setupStatus) {
    return { isKnown: false, isAllowed: false };
  }
  return { isKnown: true, isAllowed: getIsCustomNodesEnabled(setupStatus.multiuser_enabled, user?.is_admin) };
};

/**
 * Hook that returns two-state permission info for custom node management.
 *
 * - `isKnown`:  false while setupStatus is still loading; true once resolved.
 * - `isAllowed`: the actual permission decision (only trustworthy when isKnown is true).
 *
 * Consumers use these separately:
 * - **VerticalNavBar**: show the tab only when `isAllowed` (conservative — hidden while loading).
 * - **AppContent redirect**: only redirect away once `isKnown && !isAllowed` (avoids kicking
 *   a legitimate single-user session off a persisted customNodes tab before the query resolves).
 */
export const useIsCustomNodesEnabled = (): CustomNodesPermission => {
  const user = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();

  return useMemo(() => deriveCustomNodesPermission(setupStatus, user), [setupStatus, user]);
};
