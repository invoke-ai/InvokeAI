import { Center, Spinner } from '@invoke-ai/ui-library';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { logout, sessionExpiredLogout, setCredentials } from 'features/auth/store/authSlice';
import type { PropsWithChildren } from 'react';
import { memo, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGetCurrentUserQuery, useGetSetupStatusQuery } from 'services/api/endpoints/auth';

interface ProtectedRouteProps {
  requireAdmin?: boolean;
}

export const ProtectedRoute = memo(({ children, requireAdmin = false }: PropsWithChildren<ProtectedRouteProps>) => {
  const isAuthenticated = useAppSelector((state: RootState) => state.auth?.isAuthenticated || false);
  const token = useAppSelector((state: RootState) => state.auth?.token);
  const user = useAppSelector((state: RootState) => state.auth?.user);
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  // Check if multiuser mode is enabled
  const { data: setupStatus } = useGetSetupStatusQuery();
  const multiuserEnabled = setupStatus?.multiuser_enabled ?? true; // Default to true for safety

  // Only fetch user if we have a token but no user data, and multiuser mode is enabled
  const shouldFetchUser = multiuserEnabled && isAuthenticated && token && !user;
  const {
    data: currentUser,
    isLoading: isLoadingUser,
    error: userError,
  } = useGetCurrentUserQuery(undefined, {
    skip: !shouldFetchUser,
  });

  useEffect(() => {
    // Only treat 401 as session expiry. Other errors (500, network, etc.) are
    // transient and should not force logout — the 401 handler in dynamicBaseQuery
    // already covers the actual expiry case.
    if (userError && isAuthenticated && 'status' in userError && userError.status === 401) {
      dispatch(sessionExpiredLogout());
      navigate('/login', { replace: true });
    }
  }, [userError, isAuthenticated, dispatch, navigate]);

  // Detect when auth_token is removed from localStorage (e.g. by another tab,
  // browser devtools, or token expiry cleanup). The 'storage' event fires when
  // localStorage is modified by another context; we also poll periodically to
  // catch same-tab deletions (which don't trigger the storage event).
  useEffect(() => {
    if (!multiuserEnabled || !isAuthenticated) {
      return;
    }

    const checkToken = () => {
      if (!localStorage.getItem('auth_token') && isAuthenticated) {
        dispatch(sessionExpiredLogout());
        navigate('/login', { replace: true });
      }
    };

    // Listen for cross-tab localStorage changes
    window.addEventListener('storage', checkToken);
    // Poll for same-tab deletions (e.g. browser console)
    const interval = setInterval(checkToken, 5000);

    return () => {
      window.removeEventListener('storage', checkToken);
      clearInterval(interval);
    };
  }, [multiuserEnabled, isAuthenticated, dispatch, navigate]);

  useEffect(() => {
    // If we successfully fetched user data, update auth state
    if (currentUser && token && !user) {
      const userObj = {
        user_id: currentUser.user_id,
        email: currentUser.email,
        display_name: currentUser.display_name || null,
        is_admin: currentUser.is_admin || false,
        is_active: currentUser.is_active || true,
      };
      dispatch(setCredentials({ token, user: userObj }));
    }
  }, [currentUser, token, user, dispatch]);

  useEffect(() => {
    // If multiuser is disabled, allow access without authentication
    if (!multiuserEnabled) {
      // Clear any persisted auth state when switching to single-user mode
      if (isAuthenticated) {
        dispatch(logout());
      }
      return;
    }

    // In multiuser mode, check authentication
    if (!isLoadingUser && !isAuthenticated) {
      navigate('/login', { replace: true });
    } else if (!isLoadingUser && isAuthenticated && user && requireAdmin && !user.is_admin) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, isLoadingUser, requireAdmin, user, navigate, multiuserEnabled, dispatch]);

  // In single-user mode, always allow access
  if (!multiuserEnabled) {
    return <>{children}</>;
  }

  // Show loading while fetching user data
  if (isLoadingUser || (isAuthenticated && !user)) {
    return (
      <Center w="100dvw" h="100dvh">
        <Spinner size="xl" />
      </Center>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  if (requireAdmin && !user?.is_admin) {
    return null;
  }

  return <>{children}</>;
});

ProtectedRoute.displayName = 'ProtectedRoute';
