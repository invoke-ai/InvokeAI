import { Box } from '@invoke-ai/ui-library';
import { ControlLayersEditor } from 'features/controlLayers/components/ControlLayersEditor';
import { ImageViewer } from 'features/gallery/components/ImageViewer';
import { memo } from 'react';

const TextToImageTab = () => {
  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <ControlLayersEditor />
      <ImageViewer />
    </Box>
  );
};

export default memo(TextToImageTab);
