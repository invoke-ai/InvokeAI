import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import { uniq } from 'lodash-es';
import { MouseEvent, useCallback, useMemo } from 'react';
import { useListImagesQuery } from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import { selectionChanged } from '../store/gallerySlice';
import { imagesSelectors } from 'services/api/util';
import { useFeatureStatus } from '../../system/hooks/useFeatureStatus';

const selector = createSelector(
  [stateSelector, selectListImagesBaseQueryArgs],
  ({ gallery }, queryArgs) => {
    const selection = gallery.selection;

    return {
      queryArgs,
      selection,
    };
  },
  defaultSelectorOptions
);

export const useMultiselect = (imageDTO?: ImageDTO) => {
  const dispatch = useAppDispatch();
  const { queryArgs, selection } = useAppSelector(selector);

  const { imageDTOs } = useListImagesQuery(queryArgs, {
    selectFromResult: (result) => ({
      imageDTOs: result.data ? imagesSelectors.selectAll(result.data) : [],
    }),
  });

  const isMultiSelectEnabled = useFeatureStatus('multiselect').isFeatureEnabled;

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (!imageDTO) {
        return;
      }
      if (!isMultiSelectEnabled) {
        dispatch(selectionChanged([imageDTO]));
        return;
      }

      if (e.shiftKey) {
        const rangeEndImageName = imageDTO.image_name;
        const lastSelectedImage = selection[selection.length - 1]?.image_name;
        const lastClickedIndex = imageDTOs.findIndex(
          (n) => n.image_name === lastSelectedImage
        );
        const currentClickedIndex = imageDTOs.findIndex(
          (n) => n.image_name === rangeEndImageName
        );
        if (lastClickedIndex > -1 && currentClickedIndex > -1) {
          // We have a valid range!
          const start = Math.min(lastClickedIndex, currentClickedIndex);
          const end = Math.max(lastClickedIndex, currentClickedIndex);
          const imagesToSelect = imageDTOs.slice(start, end + 1);
          dispatch(selectionChanged(uniq(selection.concat(imagesToSelect))));
        }
      } else if (e.ctrlKey || e.metaKey) {
        if (
          selection.some((i) => i.image_name === imageDTO.image_name) &&
          selection.length > 1
        ) {
          dispatch(
            selectionChanged(
              selection.filter((n) => n.image_name !== imageDTO.image_name)
            )
          );
        } else {
          dispatch(selectionChanged(uniq(selection.concat(imageDTO))));
        }
      } else {
        dispatch(selectionChanged([imageDTO]));
      }
    },
    [dispatch, imageDTO, imageDTOs, selection, isMultiSelectEnabled]
  );

  const isSelected = useMemo(
    () =>
      imageDTO
        ? selection.some((i) => i.image_name === imageDTO.image_name)
        : false,
    [imageDTO, selection]
  );

  const selectionCount = useMemo(() => selection.length, [selection.length]);

  return {
    selection,
    selectionCount,
    isSelected,
    handleClick,
  };
};
