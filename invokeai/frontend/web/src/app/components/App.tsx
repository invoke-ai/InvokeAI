import ImageUploader from 'common/components/ImageUploader';
import ProgressBar from 'features/system/components/ProgressBar';
import SiteHeader from 'features/system/components/SiteHeader';
import InvokeTabs from 'features/ui/components/InvokeTabs';

import useToastWatcher from 'features/system/hooks/useToastWatcher';

import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import { Box, Flex, Grid, Portal, useColorMode } from '@chakra-ui/react';
import { APP_HEIGHT, APP_WIDTH } from 'theme/util/constants';
import ImageGalleryPanel from 'features/gallery/components/ImageGalleryPanel';
import Lightbox from 'features/lightbox/components/Lightbox';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  memo,
  PropsWithChildren,
  useCallback,
  useEffect,
  useState,
} from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Loading from 'common/components/Loading/Loading';
import { useIsApplicationReady } from 'features/system/hooks/useIsApplicationReady';
import { PartialAppConfig } from 'app/types/invokeai';
import { useGlobalHotkeys } from 'common/hooks/useGlobalHotkeys';
import { configChanged } from 'features/system/store/configSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useLogger } from 'app/logging/useLogger';
import ProgressImagePreview from 'features/parameters/components/ProgressImagePreview';

const DEFAULT_CONFIG = {};

interface Props extends PropsWithChildren {
  config?: PartialAppConfig;
}

const App = ({ config = DEFAULT_CONFIG, children }: Props) => {
  useToastWatcher();
  useGlobalHotkeys();
  const log = useLogger();

  const currentTheme = useAppSelector((state) => state.ui.currentTheme);

  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;

  const isApplicationReady = useIsApplicationReady();

  const [loadingOverridden, setLoadingOverridden] = useState(false);

  const { setColorMode } = useColorMode();
  const dispatch = useAppDispatch();

  useEffect(() => {
    log.info({ namespace: 'App', data: config }, 'Received config');
    dispatch(configChanged(config));
  }, [dispatch, config, log]);

  useEffect(() => {
    setColorMode(['light'].includes(currentTheme) ? 'light' : 'dark');
  }, [setColorMode, currentTheme]);

  const handleOverrideClicked = useCallback(() => {
    setLoadingOverridden(true);
  }, []);

  return (
    <Grid w="100vw" h="100vh" position="relative" overflow="hidden">
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
      <ProgressImagePreview />
    </Grid>
  );
};

export default memo(App);
