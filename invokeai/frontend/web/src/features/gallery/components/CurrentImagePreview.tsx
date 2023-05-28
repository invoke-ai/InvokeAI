import { Box, Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGetUrl } from 'common/util/getUrl';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';

import { gallerySelector } from '../store/gallerySelectors';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from './NextPrevImageButtons';
import { DragEvent, memo, useCallback } from 'react';
import { systemSelector } from 'features/system/store/systemSelectors';
import ImageFallbackSpinner from './ImageFallbackSpinner';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { configSelector } from '../../system/store/configSelectors';
import { useAppToaster } from 'app/components/Toaster';
import { imageSelected } from '../store/gallerySlice';

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
    shouldHidePreview,
    progressImage,
    shouldShowProgressInViewer,
    shouldAntialiasProgressImage,
  } = useAppSelector(imagesSelector);
  const { shouldFetchImages } = useAppSelector(configSelector);
  const { getUrl } = useGetUrl();
  const toaster = useAppToaster();
  const dispatch = useAppDispatch();

  const handleDragStart = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      if (!image) {
        return;
      }
      e.dataTransfer.setData('invokeai/imageName', image.image_name);
      e.dataTransfer.effectAllowed = 'move';
    },
    [image]
  );

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

  return (
    <Flex
      sx={{
        width: '100%',
        height: '100%',
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
            maxWidth: '100%',
            maxHeight: '100%',
            height: 'auto',
            position: 'absolute',
            borderRadius: 'base',
            imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
          }}
        />
      ) : (
        image && (
          <>
            <Image
              src={getUrl(image.image_url)}
              fallbackStrategy="beforeLoadOrError"
              fallback={<ImageFallbackSpinner />}
              onDragStart={handleDragStart}
              sx={{
                objectFit: 'contain',
                maxWidth: '100%',
                maxHeight: '100%',
                height: 'auto',
                position: 'absolute',
                borderRadius: 'base',
              }}
              onError={handleError}
            />
            <ImageMetadataOverlay image={image} />
          </>
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
