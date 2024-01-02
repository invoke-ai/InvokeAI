import { Flex, Grid } from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { useSocketIO } from 'app/hooks/useSocketIO';
import { useLogger } from 'app/logging/useLogger';
import { appStarted } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { $headerComponent } from 'app/store/nanostores/headerComponent';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { PartialAppConfig } from 'app/types/invokeai';
import ImageUploader from 'common/components/ImageUploader';
import { useClearStorage } from 'common/hooks/useClearStorage';
import { useGlobalHotkeys } from 'common/hooks/useGlobalHotkeys';
import { useGlobalModifiersInit } from 'common/hooks/useGlobalModifiers';
import ChangeBoardModal from 'features/changeBoardModal/components/ChangeBoardModal';
import DeleteImageModal from 'features/deleteImageModal/components/DeleteImageModal';
import { DynamicPromptsModal } from 'features/dynamicPrompts/components/DynamicPromptsPreviewModal';
import SiteHeader from 'features/system/components/SiteHeader';
import { configChanged } from 'features/system/store/configSlice';
import { languageSelector } from 'features/system/store/systemSelectors';
import InvokeTabs from 'features/ui/components/InvokeTabs';
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

  const headerComponent = useStore($headerComponent);

  return (
    <ErrorBoundary
      onReset={handleReset}
      FallbackComponent={AppErrorBoundaryFallback}
    >
      <Grid w="100vw" h="100vh" position="relative" overflow="hidden">
        <ImageUploader>
          <Grid p={4} gridAutoRows="min-content auto" w="full" h="full">
            {headerComponent || <SiteHeader />}
            <Flex gap={4} w="full" h="full">
              <InvokeTabs />
            </Flex>
          </Grid>
        </ImageUploader>
      </Grid>
      <DeleteImageModal />
      <ChangeBoardModal />
      <DynamicPromptsModal />
      <Toaster />
      <PreselectedImage selectedImage={selectedImage} />
    </ErrorBoundary>
  );
};

export default memo(App);
