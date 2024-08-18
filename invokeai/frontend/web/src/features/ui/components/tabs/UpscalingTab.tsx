import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

const UpscalingTab = () => {
  const activeTabName = useAppSelector(activeTabNameSelector);

  return (
    <Box
      display={activeTabName === 'upscaling' ? undefined : 'none'}
      hidden={activeTabName !== 'upscaling'}
      layerStyle="first"
      position="relative"
      w="full"
      h="full"
      p={2}
      borderRadius="base"
    >
      {/* <ImageViewer /> */}
    </Box>
  );
};

export default memo(UpscalingTab);
