import { Box, Center, Spinner } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { GlobalHookIsolator } from 'app/components/GlobalHookIsolator';
import { GlobalModalIsolator } from 'app/components/GlobalModalIsolator';
import { clearStorage } from 'app/store/enhancers/reduxRemember/driver';
import Loading from 'common/components/Loading/Loading';
import { AdministratorSetup } from 'features/auth/components/AdministratorSetup';
import { LoginPage } from 'features/auth/components/LoginPage';
import { ProtectedRoute } from 'features/auth/components/ProtectedRoute';
import { UserManagement } from 'features/auth/components/UserManagement';
import { UserProfile } from 'features/auth/components/UserProfile';
import { AppContent } from 'features/ui/components/AppContent';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import type { ReactNode } from 'react';
import { memo, useEffect } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { Route, Routes, useNavigate } from 'react-router-dom';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';

import AppErrorBoundaryFallback from './AppErrorBoundaryFallback';
import ThemeLocaleProvider from './ThemeLocaleProvider';

const errorBoundaryOnReset = () => {
  clearStorage();
  location.reload();
  return false;
};

const MainApp = () => {
  const isNavigationAPIConnected = useStore(navigationApi.$isConnected);
  return (
    <Box id="invoke-app-wrapper" w="100dvw" h="100dvh" position="relative" overflow="hidden">
      {isNavigationAPIConnected ? <AppContent /> : <Loading />}
    </Box>
  );
};

const SetupChecker = () => {
  const { data, isLoading } = useGetSetupStatusQuery();
  const navigate = useNavigate();

  // Check if user is already authenticated
  const token = localStorage.getItem('auth_token');
  const isAuthenticated = !!token;

  useEffect(() => {
    if (!isLoading && data) {
      // If multiuser mode is disabled, go directly to the app
      if (!data.multiuser_enabled) {
        navigate('/app', { replace: true });
      } else if (isAuthenticated) {
        // In multiuser mode, check authentication
        navigate('/app', { replace: true });
      } else if (data.setup_required) {
        navigate('/setup', { replace: true });
      } else {
        navigate('/login', { replace: true });
      }
    }
  }, [data, isLoading, navigate, isAuthenticated]);

  if (isLoading) {
    return (
      <Center w="100dvw" h="100dvh">
        <Spinner size="xl" />
      </Center>
    );
  }

  return null;
};

/** Full-page wrapper for user management / profile pages rendered inside the protected area */
const FullPageWrapper = ({ children }: { children: ReactNode }) => (
  <Box w="100dvw" h="100dvh" overflowY="auto" bg="base.900">
    {children}
  </Box>
);

const App = () => {
  return (
    <ThemeLocaleProvider>
      <ErrorBoundary onReset={errorBoundaryOnReset} FallbackComponent={AppErrorBoundaryFallback}>
        <Routes>
          <Route path="/" element={<SetupChecker />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/setup" element={<AdministratorSetup />} />
          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <FullPageWrapper>
                  <UserProfile />
                </FullPageWrapper>
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/users"
            element={
              <ProtectedRoute requireAdmin>
                <FullPageWrapper>
                  <UserManagement />
                </FullPageWrapper>
              </ProtectedRoute>
            }
          />
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <MainApp />
              </ProtectedRoute>
            }
          />
        </Routes>
        <GlobalHookIsolator />
        <GlobalModalIsolator />
      </ErrorBoundary>
    </ThemeLocaleProvider>
  );
};

export default memo(App);
