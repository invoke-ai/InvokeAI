import { Box, useGlobalModifiersInit } from '@invoke-ai/ui-library';
import { useSocketIO } from 'app/hooks/useSocketIO';
import { useLogger } from 'app/logging/useLogger';
import { appStarted } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { PartialAppConfig } from 'app/types/invokeai';
import ImageUploadOverlay from 'common/components/ImageUploadOverlay';
import { useClearStorage } from 'common/hooks/useClearStorage';
import { useFullscreenDropzone } from 'common/hooks/useFullscreenDropzone';
import { useGlobalHotkeys } from 'common/hooks/useGlobalHotkeys';
import ChangeBoardModal from 'features/changeBoardModal/components/ChangeBoardModal';
import DeleteImageModal from 'features/deleteImageModal/components/DeleteImageModal';
import { DynamicPromptsModal } from 'features/dynamicPrompts/components/DynamicPromptsPreviewModal';
import { useHFLoginToast } from 'features/modelManagerV2/hooks/useHFLoginToast';
import { useStarterModelsToast } from 'features/modelManagerV2/hooks/useStarterModelsToast';
import { configChanged } from 'features/system/store/configSlice';
import { languageSelector } from 'features/system/store/systemSelectors';
import InvokeTabs from 'features/ui/components/InvokeTabs';
import { AnimatePresence } from 'framer-motion';
import i18n from 'i18n';
import { size } from 'lodash-es';
import { memo, useCallback, useEffect } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

import AppErrorBoundaryFallback from './AppErrorBoundaryFallback';
import PreselectedImage from './PreselectedImage';
import Toaster from './Toaster';

const DEFAULT_CONFIG = {};

interface Props {
  config?: PartialAppConfig;
  selectedImage?: {
    imageName: string;
    action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
  };
}

const App = ({ config = DEFAULT_CONFIG, selectedImage }: Props) => {
  const language = useAppSelector(languageSelector);
  const logger = useLogger('system');
  const dispatch = useAppDispatch();
  const clearStorage = useClearStorage();

  // singleton!
  useSocketIO();
  useGlobalModifiersInit();
  useGlobalHotkeys();

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

  useStarterModelsToast();
  useHFLoginToast()

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
        <InvokeTabs />
        <AnimatePresence>
          {dropzone.isDragActive && isHandlingUpload && (
            <ImageUploadOverlay dropzone={dropzone} setIsHandlingUpload={setIsHandlingUpload} />
          )}
        </AnimatePresence>
      </Box>
      <DeleteImageModal />
      <ChangeBoardModal />
      <DynamicPromptsModal />
      <Toaster />
      <PreselectedImage selectedImage={selectedImage} />
    </ErrorBoundary>
  );
};

export default memo(App);
