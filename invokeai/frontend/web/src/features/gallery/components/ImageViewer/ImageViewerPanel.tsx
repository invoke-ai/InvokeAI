import { Divider, Flex } from '@invoke-ai/ui-library';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import { memo } from 'react';

import { ImageViewerContextProvider } from './context';

export const ImageViewerPanel = memo(() => {
  return (
    <ImageViewerContextProvider>
      <FocusRegionWrapper region="viewer" as={Flex} flexDir="column" w="full" h="full" overflow="hidden" p={2} gap={2}>
        <ViewerToolbar />
        <Divider />
        <ImageViewer />
      </FocusRegionWrapper>
    </ImageViewerContextProvider>
  );
});
ImageViewerPanel.displayName = 'ImageViewerPanel';
