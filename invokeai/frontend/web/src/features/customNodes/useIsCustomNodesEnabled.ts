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
 */
export const useIsCustomNodesEnabled = (): boolean => {
  const user = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();

  return useMemo(
    () => getIsCustomNodesEnabled(setupStatus ? setupStatus.multiuser_enabled : true, user?.is_admin),
    [setupStatus, user]
  );
};
