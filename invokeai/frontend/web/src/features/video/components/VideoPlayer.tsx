import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo, useRef } from 'react';
import ReactPlayer from 'react-player';
import { useVideoDTO } from 'services/api/endpoints/videos';

export const VideoPlayer = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const videoDTO = useVideoDTO(lastSelectedItem?.id);

  useFocusRegion('video', ref);

  return (
    <Flex ref={ref} w="full" h="full" flexDirection="column" gap={4} alignItems="center" justifyContent="center">
      {videoDTO?.video_url && (
        <ReactPlayer src={videoDTO.video_url} controls={true} width={videoDTO.width} height={videoDTO.height} />
      )}
    </Flex>
  );
});

VideoPlayer.displayName = 'VideoPlayer';
