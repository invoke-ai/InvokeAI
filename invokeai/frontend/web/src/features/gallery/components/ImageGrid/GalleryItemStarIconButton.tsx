import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { memo, useCallback } from 'react';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
};

export const GalleryItemStarIconButton = memo(({ imageDTO }: Props) => {
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const toggleStarredState = useCallback(() => {
    if (imageDTO.starred) {
      unstarImages({ image_names: [imageDTO.image_name] });
    } else {
      starImages({ image_names: [imageDTO.image_name] });
    }
  }, [starImages, unstarImages, imageDTO]);

  return (
    <DndImageIcon
      onClick={toggleStarredState}
      icon={imageDTO.starred ? <PiStarFill /> : <PiStarBold />}
      tooltip={imageDTO.starred ? 'Unstar' : 'Star'}
      position="absolute"
      top={2}
      insetInlineEnd={2}
    />
  );
});

GalleryItemStarIconButton.displayName = 'GalleryItemStarIconButton';
