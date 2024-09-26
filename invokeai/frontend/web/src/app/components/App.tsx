import { Box, useGlobalModifiersInit } from '@invoke-ai/ui-library';
import type { StudioInitAction } from 'app/hooks/useStudioInitAction';
import { useStudioInitAction } from 'app/hooks/useStudioInitAction';
import { useSyncQueueStatus } from 'app/hooks/useSyncQueueStatus';
import { useLogger } from 'app/logging/useLogger';
import { appStarted } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { PartialAppConfig } from 'app/types/invokeai';
import ImageUploadOverlay from 'common/components/ImageUploadOverlay';
import { useFocusRegionWatcher } from 'common/hooks/interactionScopes';
import { useClearStorage } from 'common/hooks/useClearStorage';
import { useFullscreenDropzone } from 'common/hooks/useFullscreenDropzone';
import { useGlobalHotkeys } from 'common/hooks/useGlobalHotkeys';
import ChangeBoardModal from 'features/changeBoardModal/components/ChangeBoardModal';
import DeleteImageModal from 'features/deleteImageModal/components/DeleteImageModal';
import { DynamicPromptsModal } from 'features/dynamicPrompts/components/DynamicPromptsPreviewModal';
import DeleteBoardModal from 'features/gallery/components/Boards/DeleteBoardModal';
import { useStarterModelsToast } from 'features/modelManagerV2/hooks/useStarterModelsToast';
import { ClearQueueConfirmationsAlertDialog } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { StylePresetModal } from 'features/stylePresets/components/StylePresetForm/StylePresetModal';
import RefreshAfterResetModal from 'features/system/components/SettingsModal/RefreshAfterResetModal';
import { configChanged } from 'features/system/store/configSlice';
import { selectLanguage } from 'features/system/store/systemSelectors';
import { AppContent } from 'features/ui/components/AppContent';
import { AnimatePresence } from 'framer-motion';
import i18n from 'i18n';
import { size } from 'lodash-es';
import { memo, useCallback, useEffect } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';
import { useSocketIO } from 'services/events/useSocketIO';

import AppErrorBoundaryFallback from './AppErrorBoundaryFallback';

const DEFAULT_CONFIG = {};

interface Props {
  config?: PartialAppConfig;
  studioInitAction?: StudioInitAction;
}

const App = ({ config = DEFAULT_CONFIG, studioInitAction }: Props) => {
  const language = useAppSelector(selectLanguage);
  const logger = useLogger('system');
  const dispatch = useAppDispatch();
  const clearStorage = useClearStorage();

  // singleton!
  useSocketIO();
  useGlobalModifiersInit();
  useGlobalHotkeys();
  useGetOpenAPISchemaQuery();

  const { dropzone, isHandlingUpload, setIsHandlingUpload } = useFullscreenDropzone();

  const handleReset = useCallback(() => {
    clearStorage();
    location.reload();
    return false;
  }, [clearStorage]);

  useEffect(() => {
    i18n.changeLanguage(language);
  }, [language]);

  useEffect(() => {
    if (size(config)) {
      logger.info({ config }, 'Received config');
      dispatch(configChanged(config));
    }
  }, [dispatch, config, logger]);

  useEffect(() => {
    dispatch(appStarted());
  }, [dispatch]);

  useStudioInitAction(studioInitAction);
  useStarterModelsToast();
  useSyncQueueStatus();
  useFocusRegionWatcher();

  return (
    <ErrorBoundary onReset={handleReset} FallbackComponent={AppErrorBoundaryFallback}>
      <Box
        id="invoke-app-wrapper"
        w="100vw"
        h="100vh"
        position="relative"
        overflow="hidden"
        {...dropzone.getRootProps()}
      >
        <input {...dropzone.getInputProps()} />
        <AppContent />
        <AnimatePresence>
          {dropzone.isDragActive && isHandlingUpload && (
            <ImageUploadOverlay dropzone={dropzone} setIsHandlingUpload={setIsHandlingUpload} />
          )}
        </AnimatePresence>
      </Box>
      <DeleteImageModal />
      <ChangeBoardModal />
      <DynamicPromptsModal />
      <StylePresetModal />
      <ClearQueueConfirmationsAlertDialog />
      <RefreshAfterResetModal />
      <DeleteBoardModal />
    </ErrorBoundary>
  );
};

export default memo(App);
