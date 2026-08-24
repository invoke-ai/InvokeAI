import { Center, Spinner } from '@invoke-ai/ui-library';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  externalTokenAdopted,
  sessionExpiredLogout,
  setCredentials,
  staleCredentialsDiscarded,
} from 'features/auth/store/authSlice';
import { shouldEndSessionForUnauthorized } from 'features/auth/store/authTokenRefresh';
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
    //
    // And only while this tab's token is still the live one. `sessionExpiredLogout` removes
    // `auth_token` from localStorage, which is SHARED across tabs, so acting on a 401 that
    // belongs to a superseded session deletes the credential of the session that replaced it:
    // this query goes out during page load carrying an expired token, another tab logs in
    // meanwhile, and this 401 lands before the adoption poll has run. `dynamicBaseQuery`
    // already declines to end the session in that case; this is the same decision, and both
    // ask `shouldEndSessionForUnauthorized`. Reading the token out of the store rather than
    // out of the request is sound because the two can only diverge via `externalTokenAdopted`,
    // whose foreign-token branch resets the API state and takes this error with it.
    if (userError && isAuthenticated && 'status' in userError && userError.status === 401) {
      if (!shouldEndSessionForUnauthorized(token ?? null)) {
        return;
      }
      dispatch(sessionExpiredLogout());
      navigate('/login', { replace: true });
    }
  }, [userError, isAuthenticated, token, dispatch, navigate]);

  // Detect when auth_token is removed from localStorage (e.g. by another tab,
  // browser devtools, or token expiry cleanup). The 'storage' event fires when
  // localStorage is modified by another context; we also poll periodically to
  // catch same-tab deletions (which don't trigger the storage event).
  useEffect(() => {
    if (!multiuserEnabled || !isAuthenticated) {
      return;
    }

    const checkToken = () => {
      const storedToken = localStorage.getItem('auth_token');
      if (!storedToken) {
        dispatch(sessionExpiredLogout());
        navigate('/login', { replace: true });
        return;
      }
      if (storedToken !== token) {
        dispatch(externalTokenAdopted(storedToken));
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
  }, [multiuserEnabled, isAuthenticated, token, dispatch, navigate]);

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
      // Discard the leftover auth state when switching to single-user mode. Deliberately not
      // `logout()`: that is the account-change action, and the workspace slices reset on it —
      // a mode switch keeps the same human at the machine, and in single-user mode the wipe
      // would persist over their stored canvas and workflows. See staleCredentialsDiscarded.
      if (isAuthenticated) {
        dispatch(staleCredentialsDiscarded());
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
