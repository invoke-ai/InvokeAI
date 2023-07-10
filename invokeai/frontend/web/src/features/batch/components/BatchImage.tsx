import { Box } from '@chakra-ui/react';
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
import ImageContextMenu from 'features/gallery/components/ImageContextMenu';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { ImageDTO } from 'services/api/types';

const makeSelector = (image_name: string) =>
  createSelector(
    [stateSelector],
    (state) => ({
      selectionCount: state.batch.selection.length,
      isSelected: state.batch.selection.includes(image_name),
    }),
    defaultSelectorOptions
  );

type BatchImageProps = {
  imageDTO: ImageDTO;
};

const BatchImage = (props: BatchImageProps) => {
  const dispatch = useAppDispatch();

  const { imageDTO } = props;
  const { image_name } = imageDTO;

  const selector = useMemo(() => makeSelector(image_name), [image_name]);

  const { isSelected, selectionCount } = useAppSelector(selector);

  const handleClickRemove = useCallback(() => {
    dispatch(imageRemovedFromBatch(image_name));
  }, [dispatch, image_name]);

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (e.shiftKey) {
        dispatch(batchImageRangeEndSelected(image_name));
      } else if (e.ctrlKey || e.metaKey) {
        dispatch(batchImageSelectionToggled(image_name));
      } else {
        dispatch(batchImageSelected(image_name));
      }
    },
    [dispatch, image_name]
  );

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (selectionCount > 1) {
      return {
        id: 'batch',
        payloadType: 'BATCH_SELECTION',
      };
    }

    if (imageDTO) {
      return {
        id: 'batch',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO, selectionCount]);

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none' }}>
      <ImageContextMenu image={imageDTO}>
        {(ref) => (
          <Box
            position="relative"
            key={image_name}
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
            />
          </Box>
        )}
      </ImageContextMenu>
    </Box>
  );
};

export default memo(BatchImage);
