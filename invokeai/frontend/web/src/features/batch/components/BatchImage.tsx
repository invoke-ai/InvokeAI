import { Box } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { TypesafeDraggableData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIErrorLoadingImageFallback from 'common/components/IAIErrorLoadingImageFallback';
import IAIFillSkeleton from 'common/components/IAIFillSkeleton';
import {
  batchImageRangeEndSelected,
  batchImageSelected,
  batchImageSelectionToggled,
  imageRemovedFromBatch,
} from 'features/batch/store/batchSlice';
import ImageContextMenu from 'features/gallery/components/ImageContextMenu';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const makeSelector = (image_name: string) =>
  createSelector(
    [stateSelector],
    (state) => ({
      selectionCount: state.batch.selection.length,
      selection: state.batch.selection,
      isSelected: state.batch.selection.includes(image_name),
    }),
    defaultSelectorOptions
  );

type BatchImageProps = {
  imageName: string;
};

const BatchImage = (props: BatchImageProps) => {
  const dispatch = useAppDispatch();
  const { imageName } = props;
  const {
    currentData: imageDTO,
    isLoading,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(imageName);
  const selector = useMemo(() => makeSelector(imageName), [imageName]);

  const { isSelected, selectionCount, selection } = useAppSelector(selector);

  const handleClickRemove = useCallback(() => {
    dispatch(imageRemovedFromBatch(imageName));
  }, [dispatch, imageName]);

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (e.shiftKey) {
        dispatch(batchImageRangeEndSelected(imageName));
      } else if (e.ctrlKey || e.metaKey) {
        dispatch(batchImageSelectionToggled(imageName));
      } else {
        dispatch(batchImageSelected(imageName));
      }
    },
    [dispatch, imageName]
  );

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (selectionCount > 1) {
      return {
        id: 'batch',
        payloadType: 'IMAGE_NAMES',
        payload: { image_names: selection },
      };
    }

    if (imageDTO) {
      return {
        id: 'batch',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO, selection, selectionCount]);

  if (isLoading) {
    return <IAIFillSkeleton />;
  }

  if (isError || !imageDTO) {
    return <IAIErrorLoadingImageFallback />;
  }

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none' }}>
      <ImageContextMenu imageDTO={imageDTO}>
        {(ref) => (
          <Box
            position="relative"
            key={imageName}
            userSelect="none"
            ref={ref}
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              aspectRatio: '1/1',
            }}
          >
            <IAIDndImage
              onClick={handleClick}
              imageDTO={imageDTO}
              draggableData={draggableData}
              isSelected={isSelected}
              minSize={0}
              onClickReset={handleClickRemove}
              isDropDisabled={true}
              imageSx={{ w: 'full', h: 'full' }}
              isUploadDisabled={true}
              resetTooltip="Remove from batch"
              withResetIcon
              thumbnail
            />
          </Box>
        )}
      </ImageContextMenu>
    </Box>
  );
};

export default memo(BatchImage);
