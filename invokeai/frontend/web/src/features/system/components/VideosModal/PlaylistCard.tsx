import { ExternalLink, Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { PlaylistData } from 'features/system/components/VideosModal/data';
import { videoModalLinkClicked } from 'features/system/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const PlaylistCard = memo(({ playlist }: { playlist: PlaylistData }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { tKey, link, videoCount } = playlist;
  const handleLinkClick = useCallback(() => {
    dispatch(videoModalLinkClicked(t(`supportVideos.playlists.${tKey}.title`)));
  }, [dispatch, t, tKey]);

  return (
    <Flex flexDir="column" gap={1}>
      <Flex alignItems="center" gap={2}>
        <Text fontSize="md" fontWeight="semibold">
          {t(`supportVideos.playlists.${tKey}.title`)}
        </Text>
        <Spacer />
        <Text variant="subtext">{t('supportVideos.videoCount', { count: videoCount })}</Text>
        <ExternalLink fontSize="sm" href={link} label={t('supportVideos.watchPlaylist')} onClick={handleLinkClick} />
      </Flex>
      <Text fontSize="md" variant="subtext">
        {t(`supportVideos.playlists.${tKey}.description`)}
      </Text>
    </Flex>
  );
});

PlaylistCard.displayName = 'PlaylistCard';
