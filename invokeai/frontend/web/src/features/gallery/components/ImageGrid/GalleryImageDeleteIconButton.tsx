import { useShiftModifier } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import type { MouseEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
};

export const GalleryImageDeleteIconButton = memo(({ imageDTO }: Props) => {
  const shift = useShiftModifier();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      if (!imageDTO) {
        return;
      }
      dispatch(imagesToDeleteSelected([imageDTO]));
    },
    [dispatch, imageDTO]
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

GalleryImageDeleteIconButton.displayName = 'GalleryImageDeleteIconButton';
