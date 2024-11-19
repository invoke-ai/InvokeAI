import { ExternalLink, Flex, Spacer, Text } from '@invoke-ai/ui-library';
import type { VideoData } from 'features/system/components/VideosModal/data';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const formatTime = ({ minutes, seconds }: { minutes: number; seconds: number }) => {
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

export const VideoCard = memo(({ video }: { video: VideoData }) => {
  const { t } = useTranslation();
  const { tKey, link, length } = video;
  return (
    <Flex flexDir="column" gap={1}>
      <Flex alignItems="center" gap={2}>
        <Text fontSize="md" fontWeight="semibold">
          {t(`supportVideos.videos.${tKey}.title`)}
        </Text>
        <Spacer />
        <Text variant="subtext">{formatTime(length)}</Text>
        <ExternalLink fontSize="sm" href={link} label={t('supportVideos.watch')} />
      </Flex>
      <Text fontSize="md" variant="subtext">
        {t(`supportVideos.videos.${tKey}.description`)}
      </Text>
    </Flex>
  );
});

VideoCard.displayName = 'VideoCard';
