import { Box, Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';

import { gallerySelector } from '../store/gallerySelectors';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from './NextPrevImageButtons';
import { memo, useCallback } from 'react';
import { systemSelector } from 'features/system/store/systemSelectors';
import { configSelector } from '../../system/store/configSelectors';
import { useAppToaster } from 'app/components/Toaster';
import { imageSelected } from '../store/gallerySlice';
import IAIDndImage from 'common/components/IAIDndImage';
import { ImageDTO } from 'services/api';
import { IAIImageFallback } from 'common/components/IAIImageFallback';

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
  const dispatch = useAppDispatch();

  const handleDrop = useCallback(
    (droppedImage: ImageDTO) => {
      if (droppedImage.image_name === image?.image_name) {
        return;
      }
      dispatch(imageSelected(droppedImage));
    },
    [dispatch, image?.image_name]
  );

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
          draggable={false}
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
        <Flex
          sx={{
            width: 'full',
            height: 'full',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <IAIDndImage
            image={image}
            onDrop={handleDrop}
            fallback={<IAIImageFallback sx={{ bg: 'none' }} />}
            isUploadDisabled={true}
          />
        </Flex>
      )}
      {shouldShowImageDetails && image && (
        <Box
          sx={{
            position: 'absolute',
            top: '0',
            width: 'full',
            height: 'full',
            borderRadius: 'base',
            overflow: 'scroll',
          }}
        >
          <ImageMetadataViewer image={image} />
        </Box>
      )}
      {!shouldShowImageDetails && image && (
        <Box
          sx={{
            position: 'absolute',
            top: '0',
            width: 'full',
            height: 'full',
            pointerEvents: 'none',
          }}
        >
          <NextPrevImageButtons />
        </Box>
      )}
    </Flex>
  );
};

export default memo(CurrentImagePreview);
