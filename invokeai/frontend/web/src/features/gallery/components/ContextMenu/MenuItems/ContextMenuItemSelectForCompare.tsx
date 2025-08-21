import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { imageToCompareChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemSelectForCompare = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const itemDTO = useItemDTOContext();
  const selectMaySelectForCompare = useMemo(
    () =>
      createSelector(selectGallerySlice, (gallery) => {
        if (isImageDTO(itemDTO)) {
          return gallery.imageToCompare !== itemDTO.image_name;
        }
        return false;
      }),
    [itemDTO]
  );
  const maySelectForCompare = useAppSelector(selectMaySelectForCompare);

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      dispatch(imageToCompareChanged(itemDTO.image_name));
    } else {
      // TODO: Implement video select for compare
    }
  }, [dispatch, itemDTO]);

  return (
    <IconMenuItem
      icon={<PiImagesBold />}
      isDisabled={!maySelectForCompare}
      onClick={onClick}
      aria-label={t('gallery.selectForCompare')}
      tooltip={t('gallery.selectForCompare')}
    />
  );
});

ContextMenuItemSelectForCompare.displayName = 'ContextMenuItemSelectForCompare';
