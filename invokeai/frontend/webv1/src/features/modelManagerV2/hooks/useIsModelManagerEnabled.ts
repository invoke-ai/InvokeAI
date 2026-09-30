import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';

/**
 * Hook to determine if model manager features should be enabled for the current user.
 *
 * Model management is administrator-only, so this is exactly the admin check.
 */
export const useIsModelManagerEnabled = (): boolean => useIsAdmin();
