import { Box, Flex } from '@invoke-ai/ui-library';
import { CompareToolbar } from 'features/gallery/components/ImageViewer/CompareToolbar';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import { memo } from 'react';
import { useMeasure } from 'react-use';

import { useImageViewer } from './useImageViewer';

export const ImageViewer = memo(() => {
  const imageViewer = useImageViewer();
  const [containerRef, containerDims] = useMeasure<HTMLDivElement>();

  return (
    <Flex
      layerStyle="first"
      borderRadius="base"
      position="absolute"
      flexDirection="column"
      top={0}
      right={0}
      bottom={0}
      left={0}
      p={2}
      rowGap={4}
      alignItems="center"
      justifyContent="center"
    >
      {imageViewer.isComparing && <CompareToolbar />}
      {!imageViewer.isComparing && <ViewerToolbar />}
      <Box ref={containerRef} w="full" h="full">
        {!imageViewer.isComparing && <CurrentImagePreview />}
        {imageViewer.isComparing && <ImageComparison containerDims={containerDims} />}
      </Box>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';
