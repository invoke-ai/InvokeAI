import ImageUploader from 'common/components/ImageUploader';
import Console from 'features/system/components/Console';
import ProgressBar from 'features/system/components/ProgressBar';
import SiteHeader from 'features/system/components/SiteHeader';
import InvokeTabs from 'features/ui/components/InvokeTabs';
import { keepGUIAlive } from './utils';

import useToastWatcher from 'features/system/hooks/useToastWatcher';

import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import { Box, Grid } from '@chakra-ui/react';
import { APP_HEIGHT, APP_PADDING, APP_WIDTH } from 'theme/util/constants';

keepGUIAlive();

const App = () => {
  useToastWatcher();

  return (
    <Grid w="100vw" h="100vh">
      <ImageUploader>
        <ProgressBar />
        <Grid
          gap={4}
          p={APP_PADDING}
          gridAutoRows="min-content auto"
          w={APP_WIDTH}
          h={APP_HEIGHT}
        >
          <SiteHeader />
          <InvokeTabs />
        </Grid>
        <Box>
          <Console />
        </Box>
      </ImageUploader>
      <FloatingParametersPanelButtons />
      <FloatingGalleryButton />
    </Grid>
  );
};

export default App;
