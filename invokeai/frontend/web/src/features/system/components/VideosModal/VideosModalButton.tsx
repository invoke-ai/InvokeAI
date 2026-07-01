import { IconButton } from '@invoke-ai/ui-library';
import { useVideosModal } from 'features/system/components/VideosModal/VideosModal';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiYoutubeLogoFill } from 'react-icons/pi';

export const VideosModalButton = memo(() => {
  const { t } = useTranslation();

  const videosModal = useVideosModal();

  const onClickOpen = useCallback(() => {
    videosModal.open();
  }, [videosModal]);

  return (
    <IconButton
      aria-label={t('supportVideos.supportVideos')}
      variant="link"
      icon={<PiYoutubeLogoFill fontSize={20} />}
      boxSize={8}
      onClick={onClickOpen}
    />
  );
});
VideosModalButton.displayName = 'VideosModalButton';
