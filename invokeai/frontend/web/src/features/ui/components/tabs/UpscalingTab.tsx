import { Box } from '@invoke-ai/ui-library';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo } from 'react';

const UpscalingTab = () => {
  const imageViewer = useImageViewer();
  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      {imageViewer.isOpen && <ImageViewer />}
    </Box>
  );
};

export default memo(UpscalingTab);
