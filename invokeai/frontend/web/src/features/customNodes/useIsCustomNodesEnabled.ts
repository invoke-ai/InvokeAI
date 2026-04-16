import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useMemo } from 'react';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';

/**
 * Hook to determine if custom node management should be enabled for the current user.
 *
 * Returns true if:
 * - Multiuser mode is disabled (single-user mode = always admin)
 * - Multiuser mode is enabled AND user is an admin
 *
 * Returns false if:
 * - Multiuser mode is enabled AND user is not an admin
 */
export const useIsCustomNodesEnabled = (): boolean => {
  const user = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();

  return useMemo(() => {
    if (setupStatus && !setupStatus.multiuser_enabled) {
      return true;
    }

    return user?.is_admin ?? false;
  }, [setupStatus, user]);
};
