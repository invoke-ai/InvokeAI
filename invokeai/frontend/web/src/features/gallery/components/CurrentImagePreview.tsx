import { Box, Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/storeHooks';
import { useGetUrl } from 'common/util/getUrl';
import { systemSelector } from 'features/system/store/systemSelectors';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';

import { selectedImageSelector } from '../store/gallerySelectors';
import CurrentImageFallback from './CurrentImageFallback';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from './NextPrevImageButtons';
import CurrentImageHidden from './CurrentImageHidden';

export const imagesSelector = createSelector(
  [uiSelector, selectedImageSelector, systemSelector],
  (ui, selectedImage, system) => {
    const { shouldShowImageDetails, shouldHidePreview } = ui;
    const { progressImage } = system;

    // TODO: Clean this up, this is really gross
    const imageToDisplay = progressImage
      ? {
          url: progressImage.dataURL,
          width: progressImage.width,
          height: progressImage.height,
          isProgressImage: true,
          image: progressImage,
        }
      : selectedImage
      ? {
          url: selectedImage.url,
          width: selectedImage.metadata.width,
          height: selectedImage.metadata.height,
          isProgressImage: false,
          image: selectedImage,
        }
      : null;

    return {
      shouldShowImageDetails,
      shouldHidePreview,
      imageToDisplay,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function CurrentImagePreview() {
  const { shouldShowImageDetails, imageToDisplay, shouldHidePreview } =
    useAppSelector(imagesSelector);
  const { getUrl } = useGetUrl();

  return (
    <Flex
      sx={{
        position: 'relative',
        justifyContent: 'center',
        alignItems: 'center',
        width: '100%',
        height: '100%',
      }}
    >
      {imageToDisplay && (
        <Image
          src={
            shouldHidePreview
              ? undefined
              : imageToDisplay.isProgressImage
              ? imageToDisplay.url
              : getUrl(imageToDisplay.url)
          }
          width={imageToDisplay.width}
          height={imageToDisplay.height}
          fallback={
            shouldHidePreview ? (
              <CurrentImageHidden />
            ) : !imageToDisplay.isProgressImage ? (
              <CurrentImageFallback />
            ) : undefined
          }
          sx={{
            objectFit: 'contain',
            maxWidth: '100%',
            maxHeight: '100%',
            height: 'auto',
            position: 'absolute',
            imageRendering: imageToDisplay.isProgressImage
              ? 'pixelated'
              : 'initial',
            borderRadius: 'base',
          }}
        />
      )}
      {!shouldShowImageDetails && <NextPrevImageButtons />}
      {shouldShowImageDetails &&
        imageToDisplay &&
        'metadata' in imageToDisplay.image && (
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
            <ImageMetadataViewer image={imageToDisplay.image} />
          </Box>
        )}
    </Flex>
  );
}
