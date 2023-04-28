import ImageUploader from 'common/components/ImageUploader';
import Console from 'features/system/components/Console';
import ProgressBar from 'features/system/components/ProgressBar';
import SiteHeader from 'features/system/components/SiteHeader';
import InvokeTabs from 'features/ui/components/InvokeTabs';
import { keepGUIAlive } from './utils';

import useToastWatcher from 'features/system/hooks/useToastWatcher';

import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import { Box, Flex, Grid, Portal, useColorMode } from '@chakra-ui/react';
import { APP_HEIGHT, APP_WIDTH } from 'theme/util/constants';
import ImageGalleryPanel from 'features/gallery/components/ImageGalleryPanel';
import Lightbox from 'features/lightbox/components/Lightbox';
import { useAppDispatch, useAppSelector } from './storeHooks';
import { PropsWithChildren, useCallback, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Loading from 'common/components/Loading/Loading';
import { useIsApplicationReady } from 'features/system/hooks/useIsApplicationReady';
import { PartialAppConfig } from './invokeai';
import { useGlobalHotkeys } from 'common/hooks/useGlobalHotkeys';
import { configChanged } from 'features/system/store/configSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';

keepGUIAlive();

interface Props extends PropsWithChildren {
  config?: PartialAppConfig;
}

const App = ({ config = {}, children }: Props) => {
  useToastWatcher();
  useGlobalHotkeys();

  const currentTheme = useAppSelector((state) => state.ui.currentTheme);

  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;

  const isApplicationReady = useIsApplicationReady();

  const [loadingOverridden, setLoadingOverridden] = useState(false);

  const { setColorMode } = useColorMode();
  const dispatch = useAppDispatch();

  useEffect(() => {
    console.log('Received config: ', config);
    dispatch(configChanged(config));
  }, [dispatch, config]);

  useEffect(() => {
    setColorMode(['light'].includes(currentTheme) ? 'light' : 'dark');
  }, [setColorMode, currentTheme]);

  const handleOverrideClicked = useCallback(() => {
    setLoadingOverridden(true);
  }, []);

  return (
    <Grid w="100vw" h="100vh" position="relative">
      {isLightboxEnabled && <Lightbox />}
      <ImageUploader>
        <ProgressBar />
        <Grid
          gap={4}
          p={4}
          gridAutoRows="min-content auto"
          w={APP_WIDTH}
          h={APP_HEIGHT}
        >
          {children || <SiteHeader />}
          <Flex
            gap={4}
            w={{ base: '100vw', xl: 'full' }}
            h="full"
            flexDir={{ base: 'column', xl: 'row' }}
          >
            <InvokeTabs />
            <ImageGalleryPanel />
          </Flex>
        </Grid>
      </ImageUploader>

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
      <Portal>
        <Console />
      </Portal>
    </Grid>
  );
};

export default App;
