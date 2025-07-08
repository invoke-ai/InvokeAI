import { ExternalLink, Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { VideoData } from 'features/system/components/VideosModal/data';
import { videoModalLinkClicked } from 'features/system/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const VideoCard = memo(({ video }: { video: VideoData }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { tKey, link } = video;
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
        <ExternalLink fontSize="sm" href={link} label={t('supportVideos.watch')} onClick={handleLinkClick} />
      </Flex>
      <Text fontSize="md" variant="subtext">
        {t(`supportVideos.videos.${tKey}.description`)}
      </Text>
    </Flex>
  );
});

VideoCard.displayName = 'VideoCard';
