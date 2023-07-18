import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  IMAGE_LIMIT,
  imageSelected,
  selectImagesById,
} from 'features/gallery/store/gallerySlice';
import { clamp, isEqual } from 'lodash-es';
import { useCallback } from 'react';
import {
  ListImagesArgs,
  imagesAdapter,
  imagesApi,
  useLazyListImagesQuery,
} from 'services/api/endpoints/images';
import { selectListImagesBaseQueryArgs } from '../store/gallerySelectors';

export const nextPrevImageButtonsSelector = createSelector(
  [stateSelector, selectListImagesBaseQueryArgs],
  (state, baseQueryArgs) => {
    const { data, status } =
      imagesApi.endpoints.listImages.select(baseQueryArgs)(state);

    const lastSelectedImage =
      state.gallery.selection[state.gallery.selection.length - 1];

    const isFetching = status === 'pending';

    if (!data || !lastSelectedImage || data.total === 0) {
      return {
        isFetching,
        queryArgs: baseQueryArgs,
        isOnFirstImage: true,
        isOnLastImage: true,
      };
    }

    const queryArgs: ListImagesArgs = {
      ...baseQueryArgs,
      offset: data.ids.length,
      limit: IMAGE_LIMIT,
    };

    const selectors = imagesAdapter.getSelectors();

    const images = selectors.selectAll(data);

    const currentImageIndex = images.findIndex(
      (i) => i.image_name === lastSelectedImage
    );
    const nextImageIndex = clamp(currentImageIndex + 1, 0, images.length - 1);

    const prevImageIndex = clamp(currentImageIndex - 1, 0, images.length - 1);

    const nextImageId = images[nextImageIndex].image_name;
    const prevImageId = images[prevImageIndex].image_name;

    const nextImage = selectors.selectById(data, nextImageId);
    const prevImage = selectors.selectById(data, prevImageId);

    const imagesLength = images.length;

    return {
      isOnFirstImage: currentImageIndex === 0,
      isOnLastImage:
        !isNaN(currentImageIndex) && currentImageIndex === imagesLength - 1,
      areMoreImagesAvailable: data?.total ?? 0 > imagesLength,
      isFetching: status === 'pending',
      nextImage,
      prevImage,
      nextImageId,
      prevImageId,
      queryArgs,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const useNextPrevImage = () => {
  const dispatch = useAppDispatch();

  const {
    isOnFirstImage,
    isOnLastImage,
    nextImageId,
    prevImageId,
    areMoreImagesAvailable,
    isFetching,
    queryArgs,
  } = useAppSelector(nextPrevImageButtonsSelector);

  const handlePrevImage = useCallback(() => {
    prevImageId && dispatch(imageSelected(prevImageId));
  }, [dispatch, prevImageId]);

  const handleNextImage = useCallback(() => {
    nextImageId && dispatch(imageSelected(nextImageId));
  }, [dispatch, nextImageId]);

  const [listImages] = useLazyListImagesQuery();

  const handleLoadMoreImages = useCallback(() => {
    listImages(queryArgs);
  }, [listImages, queryArgs]);

  return {
    handlePrevImage,
    handleNextImage,
    isOnFirstImage,
    isOnLastImage,
    nextImageId,
    prevImageId,
    areMoreImagesAvailable,
    handleLoadMoreImages,
    isFetching,
  };
};
