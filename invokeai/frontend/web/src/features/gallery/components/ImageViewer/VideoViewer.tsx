import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { VideoViewerContextProvider } from 'features/video/context/VideoViewerContext';
import { memo } from 'react';
import { useVideoDTO } from 'services/api/endpoints/videos';

import { CurrentVideoPreview } from './CurrentVideoPreview';
import { VideoViewerToolbar } from './VideoViewerToolbar';

export const VideoViewer = memo(() => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const videoDTO = useVideoDTO(lastSelectedItem?.type === 'video' ? lastSelectedItem.id : null);

  return (
    <VideoViewerContextProvider>
      <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
        <VideoViewerToolbar />
        <Divider />
        <Flex w="full" h="full" position="relative">
          <CurrentVideoPreview videoDTO={videoDTO ?? null} />
        </Flex>
      </Flex>
    </VideoViewerContextProvider>
  );
});

VideoViewer.displayName = 'VideoViewer';
