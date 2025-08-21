import { Box, chakra, Flex, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import ReactPlayer from 'react-player';
import { useGetVideoDTOQuery } from 'services/api/endpoints/videos';

const ChakraReactPlayer = chakra(ReactPlayer);

export const VideoPlayer = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const { data: videoDTO } = useGetVideoDTOQuery(lastSelectedItem?.id ?? skipToken);

  useFocusRegion('video', ref);

  return (
    <Flex ref={ref} w="full" h="full" flexDirection="column" gap={4}>
      {videoDTO?.video_url && (
        <Box flex={0.75} position="relative">
          <ChakraReactPlayer
            src={videoDTO.video_url}
            width="75%"
            height="75%"
            controls={true}
            position="absolute"
            top="50%"
            left="50%"
            transform="translate(-50%, -50%)"
            maxWidth="900px"
          />
        </Box>
      )}
      {!videoDTO?.video_url && <Text>{t('gallery.noVideoSelected')}</Text>}
    </Flex>
  );
});

VideoPlayer.displayName = 'VideoPlayer';
