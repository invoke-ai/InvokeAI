import { Box, Flex, Image, Skeleton, useBoolean } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useGetUrl } from 'common/util/getUrl';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';

import { gallerySelector } from '../store/gallerySelectors';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from './NextPrevImageButtons';
import CurrentImageHidden from './CurrentImageHidden';
import { DragEvent, memo, useCallback } from 'react';
import { systemSelector } from 'features/system/store/systemSelectors';
import CurrentImageFallback from './CurrentImageFallback';

export const imagesSelector = createSelector(
  [uiSelector, gallerySelector, systemSelector],
  (ui, gallery, system) => {
    const {
      shouldShowImageDetails,
      shouldHidePreview,
      shouldShowProgressInViewer,
    } = ui;
    const { selectedImage } = gallery;
    const { progressImage } = system;
    return {
      shouldShowImageDetails,
      shouldHidePreview,
      image: selectedImage,
      progressImage,
      shouldShowProgressInViewer,
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
  } = useAppSelector(imagesSelector);
  const { getUrl } = useGetUrl();

  const [isLoaded, { on, off }] = useBoolean();

  const handleDragStart = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      if (!image) {
        return;
      }
      e.dataTransfer.setData('invokeai/imageName', image.name);
      e.dataTransfer.setData('invokeai/imageType', image.type);
      e.dataTransfer.effectAllowed = 'move';
    },
    [image]
  );

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
          }}
        />
      ) : (
        image && (
          <Image
            onDragStart={handleDragStart}
            fallbackStrategy="beforeLoadOrError"
            src={shouldHidePreview ? undefined : getUrl(image.url)}
            width={image.metadata.width || 'auto'}
            height={image.metadata.height || 'auto'}
            fallback={
              shouldHidePreview ? (
                <CurrentImageHidden />
              ) : (
                <CurrentImageFallback />
              )
            }
            sx={{
              objectFit: 'contain',
              maxWidth: '100%',
              maxHeight: '100%',
              height: 'auto',
              position: 'absolute',
              borderRadius: 'base',
            }}
          />
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
