import { Flex, Grid, Portal } from '@chakra-ui/react';
import { useLogger } from 'app/logging/useLogger';
import { appStarted } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { PartialAppConfig } from 'app/types/invokeai';
import ImageUploader from 'common/components/ImageUploader';
import GalleryDrawer from 'features/gallery/components/GalleryPanel';
import DeleteImageModal from 'features/imageDeletion/components/DeleteImageModal';
import Lightbox from 'features/lightbox/components/Lightbox';
import SiteHeader from 'features/system/components/SiteHeader';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { configChanged } from 'features/system/store/configSlice';
import { languageSelector } from 'features/system/store/systemSelectors';
import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import InvokeTabs from 'features/ui/components/InvokeTabs';
import ParametersDrawer from 'features/ui/components/ParametersDrawer';
import i18n from 'i18n';
import { ReactNode, memo, useEffect } from 'react';
import DeleteBoardImagesModal from '../../features/gallery/components/Boards/DeleteBoardImagesModal';
import UpdateImageBoardModal from '../../features/gallery/components/Boards/UpdateImageBoardModal';
import GlobalHotkeys from './GlobalHotkeys';
import Toaster from './Toaster';

const DEFAULT_CONFIG = {};

interface Props {
  config?: PartialAppConfig;
  headerComponent?: ReactNode;
}

const App = ({ config = DEFAULT_CONFIG, headerComponent }: Props) => {
  const language = useAppSelector(languageSelector);

  const log = useLogger();

  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;

  const dispatch = useAppDispatch();

  useEffect(() => {
    i18n.changeLanguage(language);
  }, [language]);

  useEffect(() => {
    log.info({ namespace: 'App', data: config }, 'Received config');
    dispatch(configChanged(config));
  }, [dispatch, config, log]);

  useEffect(() => {
    dispatch(appStarted());
  }, [dispatch]);

  return (
    <>
      <Grid w="100vw" h="100vh" position="relative" overflow="hidden">
        {isLightboxEnabled && <Lightbox />}
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

        <GalleryDrawer />
        <ParametersDrawer />
        <Portal>
          <FloatingParametersPanelButtons />
        </Portal>
        <Portal>
          <FloatingGalleryButton />
        </Portal>
      </Grid>
      <DeleteImageModal />
      <UpdateImageBoardModal />
      <DeleteBoardImagesModal />
      <Toaster />
      <GlobalHotkeys />
    </>
  );
};

export default memo(App);
