import { useShiftModifier } from '@invoke-ai/ui-library';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import type { MouseEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';
import { useDeleteVideosMutation } from 'services/api/endpoints/videos';
import { isImageDTO, type ImageDTO, type VideoDTO } from 'services/api/types';

type Props = {
  itemDTO: ImageDTO | VideoDTO;
};

export const GalleryItemDeleteIconButton = memo(({ itemDTO }: Props) => {
  const shift = useShiftModifier();
  const { t } = useTranslation();
  const deleteImageModal = useDeleteImageModalApi();
  const [deleteVideos] = useDeleteVideosMutation();

  const onClick = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      if (!itemDTO) {
        return;
      }
      if (isImageDTO(itemDTO)) {
        deleteImageModal.delete([itemDTO.image_name]);
      } else {
        // TODO: Add confirm on delete and video usage functionality
        deleteVideos({ video_ids: [itemDTO.video_id] });
      }
    },
    [deleteImageModal, deleteVideos, itemDTO]
  );

  if (!shift) {
    return null;
  }

  return (
    <DndImageIcon
      onClick={onClick}
      icon={<PiTrashSimpleFill />}
      tooltip={t('gallery.deleteImage_one')}
      position="absolute"
      bottom={2}
      insetInlineEnd={2}
    />
  );
});

GalleryItemDeleteIconButton.displayName = 'GalleryItemDeleteIconButton';
