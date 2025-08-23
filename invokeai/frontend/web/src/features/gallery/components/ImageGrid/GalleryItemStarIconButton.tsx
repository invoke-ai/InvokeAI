import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { memo, useCallback } from 'react';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
import { useStarVideosMutation, useUnstarVideosMutation } from 'services/api/endpoints/videos';
import { type ImageDTO, isImageDTO, type VideoDTO } from 'services/api/types';

type Props = {
  itemDTO: ImageDTO | VideoDTO;
};

export const GalleryItemStarIconButton = memo(({ itemDTO }: Props) => {
  const customStarUi = useStore($customStarUI);
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();
  const [starVideos] = useStarVideosMutation();
  const [unstarVideos] = useUnstarVideosMutation();

  const toggleStarredState = useCallback(() => {
    if (itemDTO.starred) {
      if (isImageDTO(itemDTO)) {
        unstarImages({ image_names: [itemDTO.image_name] });
      } else {
        unstarVideos({ video_ids: [itemDTO.video_id] });
      }
    } else {
      if (isImageDTO(itemDTO)) {
        starImages({ image_names: [itemDTO.image_name] });
      } else {
        starVideos({ video_ids: [itemDTO.video_id] });
      }
    }
  }, [starImages, unstarImages, starVideos, unstarVideos, itemDTO]);

  if (customStarUi) {
    return (
      <DndImageIcon
        onClick={toggleStarredState}
        icon={itemDTO.starred ? customStarUi.on.icon : customStarUi.off.icon}
        tooltip={itemDTO.starred ? customStarUi.on.text : customStarUi.off.text}
        position="absolute"
        top={2}
        insetInlineEnd={2}
      />
    );
  }

  return (
    <DndImageIcon
      onClick={toggleStarredState}
      icon={itemDTO.starred ? <PiStarFill /> : <PiStarBold />}
      tooltip={itemDTO.starred ? 'Unstar' : 'Star'}
      position="absolute"
      top={2}
      insetInlineEnd={2}
    />
  );
});

GalleryItemStarIconButton.displayName = 'GalleryItemStarIconButton';
