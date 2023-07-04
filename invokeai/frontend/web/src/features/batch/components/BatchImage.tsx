import { Box, Icon, Skeleton } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { TypesafeDraggableData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import {
  batchImageRangeEndSelected,
  batchImageSelected,
  batchImageSelectionToggled,
  imageRemovedFromBatch,
} from 'features/batch/store/batchSlice';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { FaExclamationCircle } from 'react-icons/fa';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const makeSelector = (image_name: string) =>
  createSelector(
    [stateSelector],
    (state) => ({
      selection: state.batch.selection,
      isSelected: state.batch.selection.includes(image_name),
    }),
    defaultSelectorOptions
  );

type BatchImageProps = {
  imageName: string;
};

const BatchImage = (props: BatchImageProps) => {
  const {
    currentData: imageDTO,
    isFetching,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(props.imageName);
  const dispatch = useAppDispatch();

  const selector = useMemo(
    () => makeSelector(props.imageName),
    [props.imageName]
  );

  const { isSelected, selection } = useAppSelector(selector);

  const handleClickRemove = useCallback(() => {
    dispatch(imageRemovedFromBatch(props.imageName));
  }, [dispatch, props.imageName]);

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (e.shiftKey) {
        dispatch(batchImageRangeEndSelected(props.imageName));
      } else if (e.ctrlKey || e.metaKey) {
        dispatch(batchImageSelectionToggled(props.imageName));
      } else {
        dispatch(batchImageSelected(props.imageName));
      }
    },
    [dispatch, props.imageName]
  );

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (selection.length > 1) {
      return {
        id: 'batch',
        payloadType: 'IMAGE_NAMES',
        payload: {
          imageNames: selection,
        },
      };
    }

    if (imageDTO) {
      return {
        id: 'batch',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO, selection]);

  if (isError) {
    return <Icon as={FaExclamationCircle} />;
  }

  if (isFetching) {
    return (
      <Skeleton>
        <Box w="full" h="full" aspectRatio="1/1" />
      </Skeleton>
    );
  }

  return (
    <Box sx={{ position: 'relative', aspectRatio: '1/1' }}>
      <IAIDndImage
        imageDTO={imageDTO}
        draggableData={draggableData}
        isDropDisabled={true}
        isUploadDisabled={true}
        imageSx={{
          w: 'full',
          h: 'full',
        }}
        onClick={handleClick}
        isSelected={isSelected}
        onClickReset={handleClickRemove}
        resetTooltip="Remove from batch"
        withResetIcon
        thumbnail
      />
    </Box>
  );
};

export default memo(BatchImage);
