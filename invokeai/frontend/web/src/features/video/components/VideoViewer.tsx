import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { useRef } from 'react';
import { useVideoDTO } from 'services/api/endpoints/videos';

import { VideoPlayer } from './VideoPlayer';

export const VideoViewer = () => {
  const ref = useRef<HTMLDivElement>(null);
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const videoDTO = useVideoDTO(lastSelectedItem?.id);

  useFocusRegion('video', ref);

  if (!videoDTO) {
    return null;
  }

  return (
    <Flex ref={ref} w="full" h="full" flexDirection="column" gap={4} alignItems="center" justifyContent="center">
      <VideoPlayer videoDTO={videoDTO} />
    </Flex>
  );
};
