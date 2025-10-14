import { Box } from '@invoke-ai/ui-library';
import { GlobalHookIsolator } from 'app/components/GlobalHookIsolator';
import { GlobalModalIsolator } from 'app/components/GlobalModalIsolator';
import { clearStorage } from 'app/store/enhancers/reduxRemember/driver';
import { AppContent } from 'features/ui/components/AppContent';
import { memo } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

import AppErrorBoundaryFallback from './AppErrorBoundaryFallback';
import ThemeLocaleProvider from './ThemeLocaleProvider';

const errorBoundaryOnReset = () => {
  clearStorage();
  location.reload();
  return false;
};

const App = () => {
  return (
    <ThemeLocaleProvider>
      <ErrorBoundary onReset={errorBoundaryOnReset} FallbackComponent={AppErrorBoundaryFallback}>
        <Box id="invoke-app-wrapper" w="100dvw" h="100dvh" position="relative" overflow="hidden">
          <AppContent />
        </Box>
        <GlobalHookIsolator />
        <GlobalModalIsolator />
      </ErrorBoundary>
    </ThemeLocaleProvider>
  );
};

export default memo(App);
