import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useMemo } from 'react';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';

/**
 * Hook to determine if model manager features should be enabled for the current user.
 *
 * Returns true if:
 * - Multiuser mode is disabled (single-user mode = always admin)
 * - Multiuser mode is enabled AND user is an admin
 *
 * Returns false if:
 * - Multiuser mode is enabled AND user is not an admin
 */
export const useIsModelManagerEnabled = (): boolean => {
  const user = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();

  return useMemo(() => {
    // If multiuser is disabled, treat as admin (single-user mode)
    if (setupStatus && !setupStatus.multiuser_enabled) {
      return true;
    }

    // If multiuser is enabled, check if user is admin
    return user?.is_admin ?? false;
  }, [setupStatus, user]);
};
