import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarVideosMutation, useUnstarVideosMutation } from 'services/api/endpoints/videos';
import type { VideoDTO } from 'services/api/types';

type Props = {
  videoDTO: VideoDTO;
};

export const GalleryItemVideoStarIconButton = memo(({ videoDTO }: Props) => {
  const { t } = useTranslation();
  const [starVideos] = useStarVideosMutation();
  const [unstarVideos] = useUnstarVideosMutation();

  const toggleStarredState = useCallback(() => {
    if (videoDTO.starred) {
      unstarVideos({ video_names: [videoDTO.video_name] });
    } else {
      starVideos({ video_names: [videoDTO.video_name] });
    }
  }, [starVideos, unstarVideos, videoDTO]);

  return (
    <DndImageIcon
      onClick={toggleStarredState}
      icon={videoDTO.starred ? <PiStarFill /> : <PiStarBold />}
      tooltip={videoDTO.starred ? t('gallery.unstarVideo', { count: 1 }) : t('gallery.starVideo', { count: 1 })}
      position="absolute"
      top={2}
      insetInlineEnd={2}
    />
  );
});

GalleryItemVideoStarIconButton.displayName = 'GalleryItemVideoStarIconButton';
