import { Box, Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'features/dnd/types';
import { useNextPrevImage } from 'features/gallery/hooks/useNextPrevImage';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { AnimatePresence, motion } from 'framer-motion';
import { isEqual } from 'lodash-es';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaImage } from 'react-icons/fa';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import ImageMetadataViewer from '../ImageMetadataViewer/ImageMetadataViewer';
import NextPrevImageButtons from '../NextPrevImageButtons';
import { useTranslation } from 'react-i18next';

export const imagesSelector = createSelector(
  [stateSelector, selectLastSelectedImage],
  ({ ui, system }, lastSelectedImage) => {
    const {
      shouldShowImageDetails,
      shouldHidePreview,
      shouldShowProgressInViewer,
    } = ui;
    const { denoiseProgress, shouldAntialiasProgressImage } = system;
    return {
      shouldShowImageDetails,
      shouldHidePreview,
      imageName: lastSelectedImage?.image_name,
      denoiseProgress,
      shouldShowProgressInViewer,
      shouldAntialiasProgressImage,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const CurrentImagePreview = () => {
  const {
    shouldShowImageDetails,
    imageName,
    denoiseProgress,
    shouldShowProgressInViewer,
    shouldAntialiasProgressImage,
  } = useAppSelector(imagesSelector);

  const {
    handlePrevImage,
    handleNextImage,
    isOnLastImage,
    handleLoadMoreImages,
    areMoreImagesAvailable,
    isFetching,
  } = useNextPrevImage();

  useHotkeys(
    'left',
    () => {
      handlePrevImage();
    },
    [handlePrevImage]
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
      isOnLastImage,
      areMoreImagesAvailable,
      handleLoadMoreImages,
      isFetching,
      handleNextImage,
    ]
  );

  const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (imageDTO) {
      return {
        id: 'current-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO]);

  const droppableData = useMemo<TypesafeDroppableData | undefined>(
    () => ({
      id: 'current-image',
      actionType: 'SET_CURRENT_IMAGE',
    }),
    []
  );

  // Show and hide the next/prev buttons on mouse move
  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] =
    useState<boolean>(false);

  const timeoutId = useRef(0);

  const { t } = useTranslation();

  const handleMouseOver = useCallback(() => {
    setShouldShowNextPrevButtons(true);
    window.clearTimeout(timeoutId.current);
  }, []);

  const handleMouseOut = useCallback(() => {
    timeoutId.current = window.setTimeout(() => {
      setShouldShowNextPrevButtons(false);
    }, 500);
  }, []);

  return (
    <Flex
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
      sx={{
        width: 'full',
        height: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
      }}
    >
      {denoiseProgress?.progress_image && shouldShowProgressInViewer ? (
        <Image
          src={denoiseProgress.progress_image.dataURL}
          width={denoiseProgress.progress_image.width}
          height={denoiseProgress.progress_image.height}
          draggable={false}
          data-testid="progress-image"
          sx={{
            objectFit: 'contain',
            maxWidth: 'full',
            maxHeight: 'full',
            height: 'auto',
            position: 'absolute',
            borderRadius: 'base',
            imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
          }}
        />
      ) : (
        <IAIDndImage
          imageDTO={imageDTO}
          droppableData={droppableData}
          draggableData={draggableData}
          isUploadDisabled={true}
          fitContainer
          useThumbailFallback
          dropLabel={t('gallery.setCurrentImage')}
          noContentFallback={
            <IAINoContentFallback
              icon={FaImage}
              label={t('gallery.noImageSelected')}
            />
          }
          dataTestId="image-preview"
        />
      )}
      {shouldShowImageDetails && imageDTO && (
        <Box
          sx={{
            position: 'absolute',
            top: '0',
            width: 'full',
            height: 'full',
            borderRadius: 'base',
          }}
        >
          <ImageMetadataViewer image={imageDTO} />
        </Box>
      )}
      <AnimatePresence>
        {!shouldShowImageDetails && imageDTO && shouldShowNextPrevButtons && (
          <motion.div
            key="nextPrevButtons"
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
              transition: { duration: 0.1 },
            }}
            exit={{
              opacity: 0,
              transition: { duration: 0.1 },
            }}
            style={{
              position: 'absolute',
              top: '0',
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
            }}
          >
            <NextPrevImageButtons />
          </motion.div>
        )}
      </AnimatePresence>
    </Flex>
  );
};

export default memo(CurrentImagePreview);
