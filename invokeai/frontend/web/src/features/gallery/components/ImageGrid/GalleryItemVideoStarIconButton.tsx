import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { memo, useCallback } from 'react';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarVideosMutation, useUnstarVideosMutation } from 'services/api/endpoints/videos';
import type { VideoDTO } from 'services/api/types';

type Props = {
  videoDTO: VideoDTO;
};

export const GalleryItemVideoStarIconButton = memo(({ videoDTO }: Props) => {
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
      tooltip={videoDTO.starred ? 'Unstar' : 'Star'}
      position="absolute"
      top={2}
      insetInlineEnd={2}
    />
  );
});

GalleryItemVideoStarIconButton.displayName = 'GalleryItemVideoStarIconButton';
