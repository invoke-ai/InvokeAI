import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { IMAGE_LIMIT } from 'features/gallery/store/types';
import { getScrollToIndexAlign } from 'features/gallery/util/getScrollToIndexAlign';
import { clamp } from 'lodash-es';
import { map } from 'nanostores';
import type { RefObject } from 'react';
import { useCallback } from 'react';
import type { ListRange, VirtuosoGridHandle } from 'react-virtuoso';
import { boardsApi } from 'services/api/endpoints/boards';
import {
  imagesApi,
  useLazyListImagesQuery,
} from 'services/api/endpoints/images';
import type { ListImagesArgs } from 'services/api/types';
import { imagesSelectors } from 'services/api/util';

export type UseNextPrevImageState = {
  virtuosoRef: RefObject<VirtuosoGridHandle> | undefined;
  virtuosoRangeRef: RefObject<ListRange> | undefined;
};

export const $useNextPrevImageState = map<UseNextPrevImageState>({
  virtuosoRef: undefined,
  virtuosoRangeRef: undefined,
});

export const nextPrevImageButtonsSelector = createMemoizedSelector(
  [(state: RootState) => state, selectListImagesBaseQueryArgs],
  (state, baseQueryArgs) => {
    const { data, status } =
      imagesApi.endpoints.listImages.select(baseQueryArgs)(state);

    const { data: totalsData } =
      state.gallery.galleryView === 'images'
        ? boardsApi.endpoints.getBoardImagesTotal.select(
            baseQueryArgs.board_id ?? 'none'
          )(state)
        : boardsApi.endpoints.getBoardAssetsTotal.select(
            baseQueryArgs.board_id ?? 'none'
          )(state);

    const lastSelectedImage =
      state.gallery.selection[state.gallery.selection.length - 1];
    

    const isFetching = status === 'pending';

    if (!data || !lastSelectedImage || totalsData?.total === 0) {
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

    const images = imagesSelectors.selectAll(data);

    const currentImageIndex = images.findIndex(
      (i) => i.image_name === lastSelectedImage.image_name
    );
    const widthOfGalleryImage = Object.values(document.getElementsByClassName("gallerygrid-image"))[0]?.clientWidth
    const widthOfGalleryGrid = document.getElementById("gallery-grid")?.clientWidth

    
    const imagesPerRow = Math.floor((widthOfGalleryGrid ?? 0) / (widthOfGalleryImage ?? 1))
    

    const nextImageIndex = clamp(currentImageIndex + 1, 0, images.length - 1);
    const prevImageIndex = clamp(currentImageIndex - 1, 0, images.length - 1);
    const topImageIndex = clamp(currentImageIndex - imagesPerRow,0, images.length - 1)

    const nextImageId = images[nextImageIndex]?.image_name;
    const prevImageId = images[prevImageIndex]?.image_name;
    const topImageId = images[topImageIndex]?.image_name

    const nextImage = nextImageId
      ? imagesSelectors.selectById(data, nextImageId)
      : undefined;
    const prevImage = prevImageId
      ? imagesSelectors.selectById(data, prevImageId)
      : undefined;
    const topImage = topImageId  
      ? selectors.selectById(data, topImageId)
      : undefined;

    const imagesLength = images.length;

    return {
      loadedImagesCount: images.length,
      currentImageIndex,
      areMoreImagesAvailable: (totalsData?.total ?? 0) > imagesLength,
      isFetching: status === 'pending',
      nextImage,
      prevImage,
      nextImageIndex,
      prevImageIndex,
      queryArgs,
      topImageIndex,
      topImage
    };
  }
);

export const useNextPrevImage = () => {
  const dispatch = useAppDispatch();

  const {
    nextImage,
    nextImageIndex,
    prevImage,
    prevImageIndex,
    areMoreImagesAvailable,
    isFetching,
    queryArgs,
    loadedImagesCount,
    currentImageIndex,
    topImageIndex,
    topImage
  } = useAppSelector(nextPrevImageButtonsSelector);

  const handlePrevImage = useCallback(() => {
    prevImage && dispatch(imageSelected(prevImage));
    const range = $useNextPrevImageState.get().virtuosoRangeRef?.current;
    const virtuoso = $useNextPrevImageState.get().virtuosoRef?.current;
    if (!range || !virtuoso) {
      return;
    }

    if (
      prevImageIndex !== undefined &&
      (prevImageIndex < range.startIndex || prevImageIndex > range.endIndex)
    ) {
      virtuoso.scrollToIndex({
        index: prevImageIndex,
        behavior: 'smooth',
        align: getScrollToIndexAlign(prevImageIndex, range),
      });
    }
  }, [dispatch, prevImage, prevImageIndex]);

  const handleNextImage = useCallback(() => {
    nextImage && dispatch(imageSelected(nextImage));
    const range = $useNextPrevImageState.get().virtuosoRangeRef?.current;
    const virtuoso = $useNextPrevImageState.get().virtuosoRef?.current;
    if (!range || !virtuoso) {
      return;
    }

    if (
      nextImageIndex !== undefined &&
      (nextImageIndex < range.startIndex || nextImageIndex > range.endIndex)
    ) {
      virtuoso.scrollToIndex({
        index: nextImageIndex,
        behavior: 'smooth',
        align: getScrollToIndexAlign(nextImageIndex, range),
      });
    }
  }, [dispatch, nextImage, nextImageIndex]);

  const handleTopImage = useCallback(() => {
    topImage && dispatch(imageSelected(topImage));
    const range = $useNextPrevImageState.get().virtuosoRangeRef?.current;
    const virtuoso = $useNextPrevImageState.get().virtuosoRef?.current;

    if (!range || !virtuoso) {
      return;
    }

    if (
      topImageIndex !== undefined &&
      (topImageIndex < range.startIndex || topImageIndex > range.endIndex)
    ) {
      virtuoso.scrollToIndex({
        index: topImageIndex,
        behavior: 'smooth',
        align: getScrollToIndexAlign(topImageIndex, range),
      });
    }
  },[dispatch, topImage, topImageIndex])

  const [listImages] = useLazyListImagesQuery();

  const handleLoadMoreImages = useCallback(() => {
    listImages(queryArgs);
  }, [listImages, queryArgs]);

  return {
    handlePrevImage,
    handleNextImage,
    isOnFirstImage: currentImageIndex === 0,
    isOnLastImage:
      currentImageIndex !== undefined &&
      currentImageIndex === loadedImagesCount - 1,
    nextImage,
    prevImage,
    areMoreImagesAvailable,
    handleLoadMoreImages,
    isFetching,
    handleTopImage
  };
};
