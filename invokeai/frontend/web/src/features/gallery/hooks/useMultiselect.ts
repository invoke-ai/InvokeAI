import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import {
  selectGallerySlice,
  selectionChanged,
} from 'features/gallery/store/gallerySlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import type { MouseEvent } from 'react';
import { useCallback, useMemo } from 'react';
import { useListImagesQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { imagesSelectors } from 'services/api/util';

const selector = createMemoizedSelector(
  [selectGallerySlice, selectListImagesBaseQueryArgs],
  (gallery, queryArgs) => {
    return {
      queryArgs,
      selection: gallery.selection,
    };
  }
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
          dispatch(selectionChanged(selection.concat(imagesToSelect)));
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
          dispatch(selectionChanged(selection.concat(imageDTO)));
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
