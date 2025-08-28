import { Box, Flex, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { selectGeneratedVideo } from 'features/parameters/store/videoSlice';
import { memo, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import ReactPlayer from 'react-player';
import { useGetVideoDTOQuery } from 'services/api/endpoints/videos';

export const VideoPlayerPanel = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const generatedVideo = useAppSelector(selectGeneratedVideo);
  const lastSelectedVideoId = useAppSelector(selectLastSelectedImage);
  const { data: videoDTO } = useGetVideoDTOQuery(generatedVideo?.video_id ?? skipToken);

  useFocusRegion('video', ref);

  const videoUrl = useMemo(() => {
    // if (generatedVideo) {
    //   return generatedVideo.video_url;
    // }
    if (!videoDTO) {
      return null;
    }
    return videoDTO.video_url;
  }, [videoDTO]);

  return (
    <Flex ref={ref} w="full" h="full" flexDirection="column" gap={4}>
      {videoUrl && (
        <>
          <Box flex={0.75} position="relative">
            <ReactPlayer
              src={videoUrl}
              width="75%"
              height="75%"
              controls={true}
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                maxWidth: '900px',
              }}
            />
          </Box>
        </>
      )}
      {!videoUrl && <Text>No video generated</Text>}
    </Flex>
  );
});

VideoPlayerPanel.displayName = 'VideoPlayerPanel';
