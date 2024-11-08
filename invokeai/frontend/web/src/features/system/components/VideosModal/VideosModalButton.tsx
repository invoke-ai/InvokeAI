import { IconButton } from '@invoke-ai/ui-library';
import { useVideosModal } from 'features/system/components/VideosModal/VideosModal';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiYoutubeLogoFill } from 'react-icons/pi';

export const VideosModalButton = memo(() => {
  const { t } = useTranslation();
  const videosModal = useVideosModal();
  return (
    <IconButton
      aria-label={t('supportVideos.supportVideos')}
      variant="link"
      icon={<PiYoutubeLogoFill fontSize={20} />}
      boxSize={8}
      onClick={videosModal.open}
    />
  );
});
VideosModalButton.displayName = 'VideosModalButton';
