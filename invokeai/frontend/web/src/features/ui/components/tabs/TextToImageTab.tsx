import { Box } from '@invoke-ai/ui-library';
import { ControlLayersEditor } from 'features/controlLayers/components/ControlLayersEditor';
import { ImageComparisonDroppable } from 'features/gallery/components/ImageViewer/ImageComparisonDroppable';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo } from 'react';

const TextToImageTab = () => {
  const imageViewer = useImageViewer();
  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <ControlLayersEditor />
      {imageViewer.isOpen && (
        <>
          <ImageViewer />
          <ImageComparisonDroppable />
        </>
      )}
    </Box>
  );
};

export default memo(TextToImageTab);
