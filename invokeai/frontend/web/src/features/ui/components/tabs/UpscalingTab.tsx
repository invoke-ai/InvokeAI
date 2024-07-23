import { Box } from '@invoke-ai/ui-library';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { memo } from 'react';

const UpscalingTab = () => {
  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <ImageViewer />
    </Box>
  );
};

export default memo(UpscalingTab);
