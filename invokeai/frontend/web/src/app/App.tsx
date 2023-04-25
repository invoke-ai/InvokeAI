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
import { PropsWithChildren, useEffect } from 'react';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { shouldTransformUrlsChanged } from 'features/system/store/systemSlice';
import { setShouldFetchImages } from 'features/gallery/store/resultsSlice';
import { motion, AnimatePresence } from 'framer-motion';
import Loading from 'common/components/Loading/Loading';
import {
  ApplicationFeature,
  disabledFeaturesChanged,
  disabledTabsChanged,
} from 'features/system/store/systemSlice';
import { useIsApplicationReady } from 'features/system/hooks/useIsApplicationReady';

keepGUIAlive();

interface Props extends PropsWithChildren {
  options: {
    disabledTabs: InvokeTabName[];
    disabledFeatures: ApplicationFeature[];
    shouldTransformUrls?: boolean;
    shouldFetchImages: boolean;
  };
}

const App = (props: Props) => {
  useToastWatcher();

  const currentTheme = useAppSelector((state) => state.ui.currentTheme);
  const disabledFeatures = useAppSelector(
    (state) => state.system.disabledFeatures
  );

  const isApplicationReady = useIsApplicationReady();

  const { setColorMode } = useColorMode();
  const dispatch = useAppDispatch();

  useEffect(() => {
    dispatch(disabledFeaturesChanged(props.options.disabledFeatures));
  }, [dispatch, props.options.disabledFeatures]);

  useEffect(() => {
    dispatch(disabledTabsChanged(props.options.disabledTabs));
  }, [dispatch, props.options.disabledTabs]);

  useEffect(() => {
    dispatch(
      shouldTransformUrlsChanged(Boolean(props.options.shouldTransformUrls))
    );
  }, [dispatch, props.options.shouldTransformUrls]);

  useEffect(() => {
    dispatch(setShouldFetchImages(props.options.shouldFetchImages));
  }, [dispatch, props.options.shouldFetchImages]);

  useEffect(() => {
    setColorMode(['light'].includes(currentTheme) ? 'light' : 'dark');
  }, [setColorMode, currentTheme]);

  return (
    <Grid w="100vw" h="100vh" position="relative">
      {!disabledFeatures.includes('lightbox') && <Lightbox />}
      <ImageUploader>
        <ProgressBar />
        <Grid
          gap={4}
          p={4}
          gridAutoRows="min-content auto"
          w={APP_WIDTH}
          h={APP_HEIGHT}
        >
          {props.children || <SiteHeader />}
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
        {!isApplicationReady && (
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
