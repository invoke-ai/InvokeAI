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

/**
 * Hook wrapper around getIsCustomNodesEnabled that reads from Redux + RTK Query.
 *
 * While setupStatus is still loading we return `true` (optimistic) to prevent
 * the AppContent redirect from kicking a legitimate single-user session off a
 * persisted customNodes tab before the query resolves. The server-side auth
 * gate still protects the actual API calls.
 */
export const useIsCustomNodesEnabled = (): boolean => {
  const user = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();

  return useMemo(() => {
    if (!setupStatus) {
      return true;
    }
    return getIsCustomNodesEnabled(setupStatus.multiuser_enabled, user?.is_admin);
  }, [setupStatus, user]);
};
