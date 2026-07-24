import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useMemo } from 'react';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';

/**
 * Returns true if:
 * - Multiuser mode is disabled (single-user mode = always admin)
 * - Multiuser mode is enabled AND the user is an admin
 *
 * Exported separately from the hook so the predicate can be unit-tested directly.
 */
export const getIsAdmin = (
  setupStatus: { multiuser_enabled: boolean } | undefined,
  user: { is_admin: boolean } | null | undefined
): boolean => {
  // If multiuser is disabled, treat as admin (single-user mode)
  if (setupStatus && !setupStatus.multiuser_enabled) {
    return true;
  }

  // If multiuser is enabled, check if user is admin
  return user?.is_admin ?? false;
};

/**
 * Hook to determine whether the current user holds administrator privileges.
 *
 * This mirrors the backend's `AdminUserOrDefault` dependency. Prefer it over reading `multiuser` from
 * `runtime_config`, which is itself an admin-only route.
 */
export const useIsAdmin = (): boolean => {
  const user = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();

  return useMemo(() => getIsAdmin(setupStatus, user), [setupStatus, user]);
};
