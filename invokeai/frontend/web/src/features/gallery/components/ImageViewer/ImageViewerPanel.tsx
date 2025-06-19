import { Divider, Flex } from '@invoke-ai/ui-library';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer2';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar2';
import { memo } from 'react';

export const ImageViewerPanel = memo(() => (
  <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2} gap={2}>
    <ViewerToolbar />
    <Divider />
    <ImageViewer />
  </Flex>
));
ImageViewerPanel.displayName = 'ImageViewerPanel';
