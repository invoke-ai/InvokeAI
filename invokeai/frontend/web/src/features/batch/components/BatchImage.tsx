import { Box, Icon, Skeleton } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { FaExclamationCircle } from 'react-icons/fa';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import {
  batchImageRangeEndSelected,
  batchImageSelected,
  batchImageSelectionToggled,
  imageRemovedFromBatch,
} from 'features/batch/store/batchSlice';
import IAIDndImage from 'common/components/IAIDndImage';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { TypesafeDraggableData } from 'app/components/ImageDnd/typesafeDnd';

const isSelectedSelector = createSelector(
  [stateSelector, (state: RootState, imageName: string) => imageName],
  (state, imageName) => ({
    selection: state.batch.selection,
    isSelected: state.batch.selection.includes(imageName),
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

  const { isSelected, selection } = useAppSelector((state) =>
    isSelectedSelector(state, props.imageName)
  );

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
