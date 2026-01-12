import { Box, Center, Spinner } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { GlobalHookIsolator } from 'app/components/GlobalHookIsolator';
import { GlobalModalIsolator } from 'app/components/GlobalModalIsolator';
import { clearStorage } from 'app/store/enhancers/reduxRemember/driver';
import Loading from 'common/components/Loading/Loading';
import { AdministratorSetup } from 'features/auth/components/AdministratorSetup';
import { LoginPage } from 'features/auth/components/LoginPage';
import { ProtectedRoute } from 'features/auth/components/ProtectedRoute';
import { AppContent } from 'features/ui/components/AppContent';
import { navigationApi } from 'features/ui/layouts/navigation-api';
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
      // If user is already authenticated, redirect to main app
      if (isAuthenticated) {
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

const App = () => {
  return (
    <ThemeLocaleProvider>
      <ErrorBoundary onReset={errorBoundaryOnReset} FallbackComponent={AppErrorBoundaryFallback}>
        <Routes>
          <Route path="/" element={<SetupChecker />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/setup" element={<AdministratorSetup />} />
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
