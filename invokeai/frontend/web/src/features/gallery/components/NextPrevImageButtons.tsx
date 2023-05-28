import { ChakraProps, Flex, Grid, IconButton } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clamp, isEqual } from 'lodash-es';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaAngleLeft, FaAngleRight } from 'react-icons/fa';
import { gallerySelector } from '../store/gallerySelectors';
import { RootState } from 'app/store/store';
import { imageSelected } from '../store/gallerySlice';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  selectFilteredImagesAsObject,
  selectFilteredImagesIds,
} from '../store/imagesSlice';

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
  [
    (state: RootState) => state,
    gallerySelector,
    selectFilteredImagesAsObject,
    selectFilteredImagesIds,
  ],
  (state, gallery, filteredImagesAsObject, filteredImageIds) => {
    const { selectedImage } = gallery;

    if (!selectedImage) {
      return {
        isOnFirstImage: true,
        isOnLastImage: true,
      };
    }

    const currentImageIndex = filteredImageIds.findIndex(
      (i) => i === selectedImage.image_name
    );

    const nextImageIndex = clamp(
      currentImageIndex + 1,
      0,
      filteredImageIds.length - 1
    );

    const prevImageIndex = clamp(
      currentImageIndex - 1,
      0,
      filteredImageIds.length - 1
    );

    const nextImageId = filteredImageIds[nextImageIndex];
    const prevImageId = filteredImageIds[prevImageIndex];

    const nextImage = filteredImagesAsObject[nextImageId];
    const prevImage = filteredImagesAsObject[prevImageId];

    const imagesLength = filteredImageIds.length;

    return {
      isOnFirstImage: currentImageIndex === 0,
      isOnLastImage:
        !isNaN(currentImageIndex) && currentImageIndex === imagesLength - 1,
      nextImage,
      prevImage,
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

  const { isOnFirstImage, isOnLastImage, nextImage, prevImage } =
    useAppSelector(nextPrevImageButtonsSelector);

  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] =
    useState<boolean>(false);

  const handleCurrentImagePreviewMouseOver = useCallback(() => {
    setShouldShowNextPrevButtons(true);
  }, []);

  const handleCurrentImagePreviewMouseOut = useCallback(() => {
    setShouldShowNextPrevButtons(false);
  }, []);

  const handlePrevImage = useCallback(() => {
    dispatch(imageSelected(prevImage));
  }, [dispatch, prevImage]);

  const handleNextImage = useCallback(() => {
    dispatch(imageSelected(nextImage));
  }, [dispatch, nextImage]);

  useHotkeys(
    'left',
    () => {
      handlePrevImage();
    },
    [prevImage]
  );

  useHotkeys(
    'right',
    () => {
      handleNextImage();
    },
    [nextImage]
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
      </Grid>
    </Flex>
  );
};

export default NextPrevImageButtons;
