import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useCallback } from 'react';
import { ImagesLoadedArg, imagesLoaded } from 'services/api/thunks/image';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { selectedBoardId, galleryView } = state.gallery;

    const imageNames =
      state.gallery.imageNamesByIdAndView[selectedBoardId]?.[galleryView]
        .imageNames ?? [];

    const total =
      state.gallery.imageNamesByIdAndView[selectedBoardId]?.[galleryView]
        .total ?? 0;

    const status =
      state.gallery.statusByIdAndView[selectedBoardId]?.[galleryView] ??
      undefined;

    return {
      imageNames,
      status,
      total,
      selectedBoardId,
      galleryView,
    };
  },
  defaultSelectorOptions
);

export const useLoadMoreImages = () => {
  const dispatch = useAppDispatch();
  const { selectedBoardId, imageNames, galleryView, total, status } =
    useAppSelector(selector);

  const loadMoreImages = useCallback(
    (arg: Partial<ImagesLoadedArg>) => {
      dispatch(
        imagesLoaded({
          board_id: selectedBoardId,
          offset: imageNames.length,
          view: galleryView,
          ...arg,
        })
      );
    },
    [dispatch, galleryView, imageNames.length, selectedBoardId]
  );

  return {
    loadMoreImages,
    selectedBoardId,
    imageNames,
    galleryView,
    areMoreAvailable: imageNames.length < total,
    total,
    status,
  };
};
