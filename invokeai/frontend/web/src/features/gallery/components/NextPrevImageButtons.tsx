import { ChakraProps, Flex, Grid, IconButton, Spinner } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  imageSelected,
  selectFilteredImages,
  selectImagesById,
} from 'features/gallery/store/gallerySlice';
import { clamp, isEqual } from 'lodash-es';
import { memo, useCallback, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaAngleDoubleRight, FaAngleLeft, FaAngleRight } from 'react-icons/fa';
import { receivedPageOfImages } from 'services/api/thunks/image';

const nextPrevButtonTriggerAreaStyles: ChakraProps['sx'] = {
  height: '100%',
  width: '15%',
  alignItems: 'center',
  pointerEvents: 'auto',
};
const nextPrevButtonStyles: ChakraProps['sx'] = {
  color: 'base.100',
};

export const nextPrevImageButtonsSelector = createSelector(
  [stateSelector, selectFilteredImages],
  (state, filteredImages) => {
    const { total, isFetching } = state.gallery;
    const lastSelectedImage =
      state.gallery.selection[state.gallery.selection.length - 1];

    if (!lastSelectedImage || filteredImages.length === 0) {
      return {
        isOnFirstImage: true,
        isOnLastImage: true,
      };
    }

    const currentImageIndex = filteredImages.findIndex(
      (i) => i.image_name === lastSelectedImage
    );
    const nextImageIndex = clamp(
      currentImageIndex + 1,
      0,
      filteredImages.length - 1
    );

    const prevImageIndex = clamp(
      currentImageIndex - 1,
      0,
      filteredImages.length - 1
    );

    const nextImageId = filteredImages[nextImageIndex].image_name;
    const prevImageId = filteredImages[prevImageIndex].image_name;

    const nextImage = selectImagesById(state, nextImageId);
    const prevImage = selectImagesById(state, prevImageId);

    const imagesLength = filteredImages.length;

    return {
      isOnFirstImage: currentImageIndex === 0,
      isOnLastImage:
        !isNaN(currentImageIndex) && currentImageIndex === imagesLength - 1,
      areMoreImagesAvailable: total > imagesLength,
      isFetching,
      nextImage,
      prevImage,
      nextImageId,
      prevImageId,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const NextPrevImageButtons = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const {
    isOnFirstImage,
    isOnLastImage,
    nextImageId,
    prevImageId,
    areMoreImagesAvailable,
    isFetching,
  } = useAppSelector(nextPrevImageButtonsSelector);

  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] =
    useState<boolean>(false);

  const handleCurrentImagePreviewMouseOver = useCallback(() => {
    setShouldShowNextPrevButtons(true);
  }, []);

  const handleCurrentImagePreviewMouseOut = useCallback(() => {
    setShouldShowNextPrevButtons(false);
  }, []);

  const handlePrevImage = useCallback(() => {
    prevImageId && dispatch(imageSelected(prevImageId));
  }, [dispatch, prevImageId]);

  const handleNextImage = useCallback(() => {
    nextImageId && dispatch(imageSelected(nextImageId));
  }, [dispatch, nextImageId]);

  const handleLoadMoreImages = useCallback(() => {
    dispatch(
      receivedPageOfImages({
        is_intermediate: false,
      })
    );
  }, [dispatch]);

  useHotkeys(
    'left',
    () => {
      handlePrevImage();
    },
    [prevImageId]
  );

  useHotkeys(
    'right',
    () => {
      if (isOnLastImage && areMoreImagesAvailable && !isFetching) {
        handleLoadMoreImages();
        return;
      }
      if (!isOnLastImage) {
        handleNextImage();
      }
    },
    [
      nextImageId,
      isOnLastImage,
      areMoreImagesAvailable,
      handleLoadMoreImages,
      isFetching,
    ]
  );

  return (
    <Flex
      sx={{
        justifyContent: 'space-between',
        height: '100%',
        width: '100%',
        pointerEvents: 'none',
      }}
    >
      <Grid
        sx={{
          ...nextPrevButtonTriggerAreaStyles,
          justifyContent: 'flex-start',
        }}
        onMouseOver={handleCurrentImagePreviewMouseOver}
        onMouseOut={handleCurrentImagePreviewMouseOut}
      >
        {shouldShowNextPrevButtons && !isOnFirstImage && (
          <IconButton
            aria-label={t('accessibility.previousImage')}
            icon={<FaAngleLeft size={64} />}
            variant="unstyled"
            onClick={handlePrevImage}
            boxSize={16}
            sx={nextPrevButtonStyles}
          />
        )}
      </Grid>
      <Grid
        sx={{
          ...nextPrevButtonTriggerAreaStyles,
          justifyContent: 'flex-end',
        }}
        onMouseOver={handleCurrentImagePreviewMouseOver}
        onMouseOut={handleCurrentImagePreviewMouseOut}
      >
        {shouldShowNextPrevButtons && !isOnLastImage && (
          <IconButton
            aria-label={t('accessibility.nextImage')}
            icon={<FaAngleRight size={64} />}
            variant="unstyled"
            onClick={handleNextImage}
            boxSize={16}
            sx={nextPrevButtonStyles}
          />
        )}
        {shouldShowNextPrevButtons &&
          isOnLastImage &&
          areMoreImagesAvailable &&
          !isFetching && (
            <IconButton
              aria-label={t('accessibility.loadMore')}
              icon={<FaAngleDoubleRight size={64} />}
              variant="unstyled"
              onClick={handleLoadMoreImages}
              boxSize={16}
              sx={nextPrevButtonStyles}
            />
          )}
        {shouldShowNextPrevButtons &&
          isOnLastImage &&
          areMoreImagesAvailable &&
          isFetching && (
            <Flex
              sx={{
                w: 16,
                h: 16,
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Spinner opacity={0.5} size="xl" />
            </Flex>
          )}
      </Grid>
    </Flex>
  );
};

export default memo(NextPrevImageButtons);
