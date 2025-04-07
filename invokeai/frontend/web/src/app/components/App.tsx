import { Box } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { GlobalHookIsolator } from 'app/components/GlobalHookIsolator';
import { GlobalModalIsolator } from 'app/components/GlobalModalIsolator';
import type { StudioInitAction } from 'app/hooks/useStudioInitAction';
import { $didStudioInit } from 'app/hooks/useStudioInitAction';
import type { PartialAppConfig } from 'app/types/invokeai';
import Loading from 'common/components/Loading/Loading';
import { useClearStorage } from 'common/hooks/useClearStorage';
import { AppContent } from 'features/ui/components/AppContent';
import { memo, useCallback } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

import AppErrorBoundaryFallback from './AppErrorBoundaryFallback';
const DEFAULT_CONFIG = {};

interface Props {
  config?: PartialAppConfig;
  studioInitAction?: StudioInitAction;
}

const App = ({ config = DEFAULT_CONFIG, studioInitAction }: Props) => {
  const didStudioInit = useStore($didStudioInit);
  const clearStorage = useClearStorage();

  const handleReset = useCallback(() => {
    clearStorage();
    location.reload();
    return false;
  }, [clearStorage]);

  return (
    <ErrorBoundary onReset={handleReset} FallbackComponent={AppErrorBoundaryFallback}>
      <Box id="invoke-app-wrapper" w="100dvw" h="100dvh" position="relative" overflow="hidden">
        <AppContent />
        {!didStudioInit && <Loading />}
      </Box>
      <GlobalHookIsolator config={config} studioInitAction={studioInitAction} />
      <GlobalModalIsolator />
    </ErrorBoundary>
  );
};

export default memo(App);
