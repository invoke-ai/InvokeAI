import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useVideosModal } from 'features/system/components/VideosModal/VideosModal';
import { videoModalOpened } from 'features/system/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiYoutubeLogoFill } from 'react-icons/pi';

export const VideosModalButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const videosModal = useVideosModal();

  const onClickOpen = useCallback(() => {
    dispatch(videoModalOpened());
    videosModal.open();
  }, [videosModal, dispatch]);

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
