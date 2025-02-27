import { ExternalLink, Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { VideoData } from 'features/system/components/VideosModal/data';
import { videoModalLinkClicked } from 'features/system/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const formatTime = ({ minutes, seconds }: { minutes: number; seconds: number }) => {
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

export const VideoCard = memo(({ video }: { video: VideoData }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { tKey, link, length } = video;
  const handleLinkClick = useCallback(() => {
    dispatch(videoModalLinkClicked(t(`supportVideos.videos.${tKey}.title`)));
  }, [dispatch, t, tKey]);

  return (
    <Flex flexDir="column" gap={1}>
      <Flex alignItems="center" gap={2}>
        <Text fontSize="md" fontWeight="semibold">
          {t(`supportVideos.videos.${tKey}.title`)}
        </Text>
        <Spacer />
        <Text variant="subtext">{formatTime(length)}</Text>
        <ExternalLink fontSize="sm" href={link} label={t('supportVideos.watch')} onClick={handleLinkClick} />
      </Flex>
      <Text fontSize="md" variant="subtext">
        {t(`supportVideos.videos.${tKey}.description`)}
      </Text>
    </Flex>
  );
});

VideoCard.displayName = 'VideoCard';
