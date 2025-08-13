import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { memo, useCallback } from 'react';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';
import { useStarResourcesMutation, useUnstarResourcesMutation } from 'services/api/endpoints/resources';

type Props = {
  imageDTO: ImageDTO;
};

export const GalleryImageStarIconButton = memo(({ imageDTO }: Props) => {
  const customStarUi = useStore($customStarUI);
  const [starResources] = useStarResourcesMutation();
  const [unstarResources] = useUnstarResourcesMutation();

  const toggleStarredState = useCallback(() => {
    if (imageDTO.starred) {
      unstarResources({ resources: [{ resource_id: imageDTO.image_name, resource_type: "image" }] });
    } else {
      starResources({ resources: [{ resource_id: imageDTO.image_name, resource_type: "image" }] });
    }
  }, [starResources, unstarResources, imageDTO]);

  if (customStarUi) {
    return (
      <DndImageIcon
        onClick={toggleStarredState}
        icon={imageDTO.starred ? customStarUi.on.icon : customStarUi.off.icon}
        tooltip={imageDTO.starred ? customStarUi.on.text : customStarUi.off.text}
        position="absolute"
        top={2}
        insetInlineEnd={2}
      />
    );
  }

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

GalleryImageStarIconButton.displayName = 'GalleryImageStarIconButton';
