import { Box, Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGetUrl } from 'common/util/getUrl';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';

import { gallerySelector } from '../store/gallerySelectors';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from './NextPrevImageButtons';
import { DragEvent, memo, useCallback, useEffect, useState } from 'react';
import { systemSelector } from 'features/system/store/systemSelectors';
import ImageFallbackSpinner from './ImageFallbackSpinner';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { configSelector } from '../../system/store/configSelectors';
import { useAppToaster } from 'app/components/Toaster';
import { imageSelected } from '../store/gallerySlice';
import { useDraggable } from '@dnd-kit/core';

export const imagesSelector = createSelector(
  [uiSelector, gallerySelector, systemSelector],
  (ui, gallery, system) => {
    const {
      shouldShowImageDetails,
      shouldHidePreview,
      shouldShowProgressInViewer,
    } = ui;
    const { selectedImage } = gallery;
    const { progressImage, shouldAntialiasProgressImage } = system;
    return {
      shouldShowImageDetails,
      shouldHidePreview,
      image: selectedImage,
      progressImage,
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
    image,
    progressImage,
    shouldShowProgressInViewer,
    shouldAntialiasProgressImage,
  } = useAppSelector(imagesSelector);
  const { shouldFetchImages } = useAppSelector(configSelector);
  const { getUrl } = useGetUrl();
  const toaster = useAppToaster();
  const dispatch = useAppDispatch();
  const [isLoaded, setIsLoaded] = useState(false);

  const { attributes, listeners, setNodeRef } = useDraggable({
    id: `currentImage_${image?.image_name}`,
    data: {
      image,
    },
  });

  const handleError = useCallback(() => {
    dispatch(imageSelected());
    if (shouldFetchImages) {
      toaster({
        title: 'Something went wrong, please refresh',
        status: 'error',
        isClosable: true,
      });
    }
  }, [dispatch, toaster, shouldFetchImages]);

  useEffect(() => {
    setIsLoaded(false);
  }, [image]);

  return (
    <Flex
      sx={{
        width: 'full',
        height: 'full',
        position: 'relative',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {progressImage && shouldShowProgressInViewer ? (
        <Image
          src={progressImage.dataURL}
          width={progressImage.width}
          height={progressImage.height}
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
        image && (
          <Flex
            sx={{
              alignItems: 'center',
              justifyContent: 'center',
              position: 'absolute',
            }}
          >
            <Image
              ref={setNodeRef}
              {...listeners}
              {...attributes}
              src={getUrl(image.image_url)}
              fallbackStrategy="beforeLoadOrError"
              fallback={<ImageFallbackSpinner />}
              sx={{
                objectFit: 'contain',
                maxWidth: '100%',
                maxHeight: '100%',
                height: 'auto',
                borderRadius: 'base',
                touchAction: 'none',
              }}
              onError={handleError}
              onLoad={() => {
                setIsLoaded(true);
              }}
            />
            {isLoaded && <ImageMetadataOverlay image={image} />}
          </Flex>
        )
      )}
      {shouldShowImageDetails && image && 'metadata' in image && (
        <Box
          sx={{
            position: 'absolute',
            top: '0',
            width: '100%',
            height: '100%',
            borderRadius: 'base',
            overflow: 'scroll',
          }}
        >
          <ImageMetadataViewer image={image} />
        </Box>
      )}
      {!shouldShowImageDetails && <NextPrevImageButtons />}
    </Flex>
  );
};

export default memo(CurrentImagePreview);
