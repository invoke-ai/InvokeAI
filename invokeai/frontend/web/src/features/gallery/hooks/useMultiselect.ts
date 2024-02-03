import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { galleryImageClicked, imageSelectionChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import type { MouseEvent } from 'react';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

export const useMultiselect = (imageDTO?: ImageDTO) => {
  const dispatch = useAppDispatch();
  const areMultiplesSelected = useAppSelector((s) => s.gallery.selection.length > 1);
  const selectIsSelected = useMemo(
    () =>
      createSelector(selectGallerySlice, (gallery) =>
        gallery.selection.some((i) => i.image_name === imageDTO?.image_name)
      ),
    [imageDTO?.image_name]
  );
  const isSelected = useAppSelector(selectIsSelected);
  const isMultiSelectEnabled = useFeatureStatus('multiselect').isFeatureEnabled;

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (!imageDTO) {
        return;
      }
      if (!isMultiSelectEnabled) {
        dispatch(imageSelectionChanged(imageDTO));
        return;
      }

      dispatch(
        galleryImageClicked({
          imageDTO,
          shiftKey: e.shiftKey,
          ctrlKey: e.ctrlKey,
          metaKey: e.metaKey,
        })
      );
    },
    [dispatch, imageDTO, isMultiSelectEnabled]
  );

  return {
    areMultiplesSelected,
    isSelected,
    handleClick,
  };
};
