import { Box, chakra, Flex, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import ReactPlayer from 'react-player';
import { useGetVideoDTOQuery } from 'services/api/endpoints/videos';

export const VideoPlayer = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const { data: videoDTO } = useGetVideoDTOQuery(lastSelectedItem?.id ?? skipToken);

  useFocusRegion('video', ref);

  return (
    <Flex ref={ref} w="full" h="full" flexDirection="column" gap={4} alignItems="center" justifyContent="center">
      {videoDTO?.video_url && (
        <ReactPlayer src={videoDTO.video_url} controls={true} width={videoDTO.width} height={videoDTO.height} />
      )}
      {!videoDTO?.video_url && <Text>{t('gallery.noVideoSelected')}</Text>}
    </Flex>
  );
});

VideoPlayer.displayName = 'VideoPlayer';
