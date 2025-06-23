import { Divider, Flex } from '@invoke-ai/ui-library';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import { memo } from 'react';

import { ImageViewerContextProvider } from './context';

export const ImageViewerPanel = memo(() => {
  return (
    <ImageViewerContextProvider>
      <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2} gap={2}>
        <ViewerToolbar />
        <Divider />
        <ImageViewer />
      </Flex>
    </ImageViewerContextProvider>
  );
});
ImageViewerPanel.displayName = 'ImageViewerPanel';
