import { Flex, Grid } from '@chakra-ui/react';
import { useLogger } from 'app/logging/useLogger';
import { appStarted } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { PartialAppConfig } from 'app/types/invokeai';
import ImageUploader from 'common/components/ImageUploader';
import ChangeBoardModal from 'features/changeBoardModal/components/ChangeBoardModal';
import DeleteImageModal from 'features/deleteImageModal/components/DeleteImageModal';
import SiteHeader from 'features/system/components/SiteHeader';
import { configChanged } from 'features/system/store/configSlice';
import { languageSelector } from 'features/system/store/systemSelectors';
import InvokeTabs from 'features/ui/components/InvokeTabs';
import i18n from 'i18n';
import { size } from 'lodash-es';
import { ReactNode, memo, useCallback, useEffect } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { usePreselectedImage } from '../../features/parameters/hooks/usePreselectedImage';
import AppErrorBoundaryFallback from './AppErrorBoundaryFallback';
import GlobalHotkeys from './GlobalHotkeys';
import Toaster from './Toaster';

const DEFAULT_CONFIG = {};

interface Props {
  config?: PartialAppConfig;
  headerComponent?: ReactNode;
  selectedImage?: {
    imageName: string;
    action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
  };
}

const App = ({
  config = DEFAULT_CONFIG,
  headerComponent,
  selectedImage,
}: Props) => {
  const language = useAppSelector(languageSelector);

  const logger = useLogger('system');
  const dispatch = useAppDispatch();
  const { handlePreselectedImage } = usePreselectedImage();
  const handleReset = useCallback(() => {
    localStorage.clear();
    location.reload();
    return false;
  }, []);

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

  useEffect(() => {
    handlePreselectedImage(selectedImage);
  }, [handlePreselectedImage, selectedImage]);

  return (
    <ErrorBoundary
      onReset={handleReset}
      FallbackComponent={AppErrorBoundaryFallback}
    >
      <Grid w="100vw" h="100vh" position="relative" overflow="hidden">
        <ImageUploader>
          <Grid
            sx={{
              gap: 4,
              p: 4,
              gridAutoRows: 'min-content auto',
              w: 'full',
              h: 'full',
            }}
          >
            {headerComponent || <SiteHeader />}
            <Flex
              sx={{
                gap: 4,
                w: 'full',
                h: 'full',
              }}
            >
              <InvokeTabs />
            </Flex>
          </Grid>
        </ImageUploader>
      </Grid>
      <DeleteImageModal />
      <ChangeBoardModal />
      <Toaster />
      <GlobalHotkeys />
    </ErrorBoundary>
  );
};

export default memo(App);
