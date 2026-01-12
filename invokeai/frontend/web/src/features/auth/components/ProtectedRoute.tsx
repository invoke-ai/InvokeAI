import { Center, Spinner } from '@invoke-ai/ui-library';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { logout, setCredentials } from 'features/auth/store/authSlice';
import type { PropsWithChildren } from 'react';
import { memo, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGetCurrentUserQuery } from 'services/api/endpoints/auth';

interface ProtectedRouteProps {
  requireAdmin?: boolean;
}

export const ProtectedRoute = memo(({ children, requireAdmin = false }: PropsWithChildren<ProtectedRouteProps>) => {
  const isAuthenticated = useAppSelector((state: RootState) => state.auth?.isAuthenticated || false);
  const token = useAppSelector((state: RootState) => state.auth?.token);
  const user = useAppSelector((state: RootState) => state.auth?.user);
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  // Only fetch user if we have a token but no user data
  const shouldFetchUser = isAuthenticated && token && !user;
  const {
    data: currentUser,
    isLoading: isLoadingUser,
    error: userError,
  } = useGetCurrentUserQuery(undefined, {
    skip: !shouldFetchUser,
  });

  useEffect(() => {
    // If we have a token but fetching user failed, token is invalid - logout
    if (userError && isAuthenticated) {
      dispatch(logout());
      navigate('/login', { replace: true });
    }
  }, [userError, isAuthenticated, dispatch, navigate]);

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
    if (!isLoadingUser && !isAuthenticated) {
      navigate('/login', { replace: true });
    } else if (!isLoadingUser && isAuthenticated && user && requireAdmin && !user.is_admin) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, isLoadingUser, requireAdmin, user, navigate]);

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
