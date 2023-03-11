import { ChakraProps, Flex, Grid, IconButton } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { isEqual } from 'lodash';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaAngleLeft, FaAngleRight } from 'react-icons/fa';
import { gallerySelector } from '../store/gallerySelectors';
import {
  GalleryCategory,
  selectNextImage,
  selectPrevImage,
} from '../store/gallerySlice';

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
  gallerySelector,
  (gallery) => {
    const { currentImage } = gallery;

    const tempImages =
      gallery.categories[
        currentImage ? (currentImage.category as GalleryCategory) : 'result'
      ].images;

    const currentImageIndex = tempImages.findIndex(
      (i) => i.uuid === gallery?.currentImage?.uuid
    );
    const imagesLength = tempImages.length;

    return {
      isOnFirstImage: currentImageIndex === 0,
      isOnLastImage:
        !isNaN(currentImageIndex) && currentImageIndex === imagesLength - 1,
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

  const { isOnFirstImage, isOnLastImage } = useAppSelector(
    nextPrevImageButtonsSelector
  );

  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] =
    useState<boolean>(false);

  const handleCurrentImagePreviewMouseOver = () => {
    setShouldShowNextPrevButtons(true);
  };

  const handleCurrentImagePreviewMouseOut = () => {
    setShouldShowNextPrevButtons(false);
  };

  const handleClickPrevButton = () => {
    dispatch(selectPrevImage());
  };

  const handleClickNextButton = () => {
    dispatch(selectNextImage());
  };

  return (
    <Flex
      sx={{
        justifyContent: 'space-between',
        zIndex: 1,
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
            onClick={handleClickPrevButton}
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
            onClick={handleClickNextButton}
            boxSize={16}
            sx={nextPrevButtonStyles}
          />
        )}
      </Grid>
    </Flex>
  );
};

export default NextPrevImageButtons;
