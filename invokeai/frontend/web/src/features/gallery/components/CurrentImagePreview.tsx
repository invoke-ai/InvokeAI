import { Box, Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { isEqual } from 'lodash-es';
import { memo, useMemo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from './NextPrevImageButtons';

export const imagesSelector = createSelector(
  [stateSelector, selectLastSelectedImage],
  ({ ui, system }, lastSelectedImage) => {
    const {
      shouldShowImageDetails,
      shouldHidePreview,
      shouldShowProgressInViewer,
    } = ui;
    const { progressImage, shouldAntialiasProgressImage } = system;
    return {
      shouldShowImageDetails,
      shouldHidePreview,
      imageName: lastSelectedImage,
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
    imageName,
    progressImage,
    shouldShowProgressInViewer,
    shouldAntialiasProgressImage,
  } = useAppSelector(imagesSelector);

  const {
    currentData: imageDTO,
    isLoading,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(imageName ?? skipToken);

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

  return (
    <Flex
      sx={{
        width: 'full',
        height: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
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
        <IAIDndImage
          imageDTO={imageDTO}
          droppableData={droppableData}
          draggableData={draggableData}
          isUploadDisabled={true}
          fitContainer
          dropLabel="Set as Current Image"
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
            overflow: 'scroll',
          }}
        >
          <ImageMetadataViewer image={imageDTO} />
        </Box>
      )}
      {!shouldShowImageDetails && imageDTO && (
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
