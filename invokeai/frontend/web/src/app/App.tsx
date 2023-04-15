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
import { useAppSelector } from './storeHooks';
import { PropsWithChildren, useEffect } from 'react';
import { isMobile } from 'theme/util/isMobile';
import MediaQuery from 'react-responsive';

keepGUIAlive();

const App = (props: PropsWithChildren) => {
  useToastWatcher();

  const currentTheme = useAppSelector((state) => state.ui.currentTheme);
  const { setColorMode } = useColorMode();

  useEffect(() => {
    setColorMode(['light'].includes(currentTheme) ? 'light' : 'dark');
  }, [setColorMode, currentTheme]);

  return (
    <Grid w="100vw" h="100vh">
      <Lightbox />
      <ImageUploader>
        <ProgressBar />
        <Grid
          gap={4}
          p={isMobile ? 1 : 4}
          gridAutoRows="max-content auto"
          w={APP_WIDTH}
          h={APP_HEIGHT}
        >
          {props.children || <SiteHeader />}
          <MediaQuery minDeviceWidth={768}>
            <Flex gap={4} w="full" h="full">
              <InvokeTabs />
              <ImageGalleryPanel />
            </Flex>
          </MediaQuery>
          <MediaQuery maxDeviceWidth={768}>
            <Box position="relative" overflowY="scroll">
              <InvokeTabs />
            </Box>
          </MediaQuery>
        </Grid>
        <Box>
          <Console />
        </Box>
      </ImageUploader>
      <Portal>
        <FloatingParametersPanelButtons />
      </Portal>
      <Portal>
        <FloatingGalleryButton />
      </Portal>
    </Grid>
  );
};

export default App;
