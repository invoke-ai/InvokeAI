import { Box, Flex, Grid, Portal } from '@chakra-ui/react';
import { useLogger } from 'app/logging/useLogger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { PartialAppConfig } from 'app/types/invokeai';
import ImageUploader from 'common/components/ImageUploader';
import Loading from 'common/components/Loading/Loading';
import GalleryDrawer from 'features/gallery/components/GalleryPanel';
import Lightbox from 'features/lightbox/components/Lightbox';
import SiteHeader from 'features/system/components/SiteHeader';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useIsApplicationReady } from 'features/system/hooks/useIsApplicationReady';
import { configChanged } from 'features/system/store/configSlice';
import { languageSelector } from 'features/system/store/systemSelectors';
import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import InvokeTabs from 'features/ui/components/InvokeTabs';
import ParametersDrawer from 'features/ui/components/ParametersDrawer';
import { AnimatePresence, motion } from 'framer-motion';
import i18n from 'i18n';
import { ReactNode, memo, useCallback, useEffect, useState } from 'react';
import { APP_HEIGHT, APP_WIDTH } from 'theme/util/constants';
import GlobalHotkeys from './GlobalHotkeys';
import Toaster from './Toaster';

const DEFAULT_CONFIG = {};

interface Props {
  config?: PartialAppConfig;
  headerComponent?: ReactNode;
  setIsReady?: (isReady: boolean) => void;
}

const App = ({
  config = DEFAULT_CONFIG,
  headerComponent,
  setIsReady,
}: Props) => {
  const language = useAppSelector(languageSelector);

  const log = useLogger();

  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;

  const isApplicationReady = useIsApplicationReady();

  const [loadingOverridden, setLoadingOverridden] = useState(false);

  const dispatch = useAppDispatch();

  useEffect(() => {
    i18n.changeLanguage(language);
  }, [language]);

  useEffect(() => {
    log.info({ namespace: 'App', data: config }, 'Received config');
    dispatch(configChanged(config));
  }, [dispatch, config, log]);

  const handleOverrideClicked = useCallback(() => {
    setLoadingOverridden(true);
  }, []);

  useEffect(() => {
    if (isApplicationReady && setIsReady) {
      setIsReady(true);
    }

    return () => {
      setIsReady && setIsReady(false);
    };
  }, [isApplicationReady, setIsReady]);

  return (
    <>
      <Grid w="100vw" h="100vh" position="relative" overflow="hidden">
        {isLightboxEnabled && <Lightbox />}
        <ImageUploader>
          <Grid
            gap={4}
            p={4}
            gridAutoRows="min-content auto"
            w={APP_WIDTH}
            h={APP_HEIGHT}
          >
            {headerComponent || <SiteHeader />}
            <Flex
              gap={4}
              w={{ base: '100vw', xl: 'full' }}
              h="full"
              flexDir={{ base: 'column', xl: 'row' }}
            >
              <InvokeTabs />
            </Flex>
          </Grid>
        </ImageUploader>

        <GalleryDrawer />
        <ParametersDrawer />

        <AnimatePresence>
          {!isApplicationReady && !loadingOverridden && (
            <motion.div
              key="loading"
              initial={{ opacity: 1 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              style={{ zIndex: 3 }}
            >
              <Box position="absolute" top={0} left={0} w="100vw" h="100vh">
                <Loading />
              </Box>
              <Box
                onClick={handleOverrideClicked}
                position="absolute"
                top={0}
                right={0}
                cursor="pointer"
                w="2rem"
                h="2rem"
              />
            </motion.div>
          )}
        </AnimatePresence>

        <Portal>
          <FloatingParametersPanelButtons />
        </Portal>
        <Portal>
          <FloatingGalleryButton />
        </Portal>
      </Grid>
      <Toaster />
      <GlobalHotkeys />
    </>
  );
};

export default memo(App);
