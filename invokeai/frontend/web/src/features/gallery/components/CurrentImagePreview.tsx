import { Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/storeHooks';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';

import { gallerySelector } from '../store/gallerySelectors';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from './NextPrevImageButtons';

export const imagesSelector = createSelector(
  [gallerySelector, uiSelector],
  (gallery: GalleryState, ui) => {
    const { currentImage, intermediateImage } = gallery;
    const { shouldShowImageDetails } = ui;

    return {
      imageToDisplay: intermediateImage ? intermediateImage : currentImage,
      isIntermediate: Boolean(intermediateImage),
      shouldShowImageDetails,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function CurrentImagePreview() {
  const { shouldShowImageDetails, imageToDisplay, isIntermediate } =
    useAppSelector(imagesSelector);

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
          src={imageToDisplay.url}
          sx={{
            objectFit: 'contain',
            maxWidth: '100%',
            maxHeight: '100%',
            height: 'auto',
            position: 'absolute',
            imageRendering: isIntermediate ? 'pixelated' : 'initial',
            borderRadius: 'base',
          }}
          {...(isIntermediate && {
            width: imageToDisplay.width,
            height: imageToDisplay.height,
          })}
        />
      )}
      {!shouldShowImageDetails && <NextPrevImageButtons />}
      {shouldShowImageDetails && imageToDisplay && (
        <ImageMetadataViewer
          image={imageToDisplay}
          styleClass="current-image-metadata"
        />
      )}
    </Flex>
  );
}
