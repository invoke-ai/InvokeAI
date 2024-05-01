import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Spinner, useShiftModifier } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { selectControlAdaptersSlice } from 'features/controlAdapters/store/controlAdaptersSlice';
import { heightChanged, widthChanged } from 'features/controlLayers/store/controlLayersSlice';
import type { ImageWithDims } from 'features/controlLayers/util/controlAdapters';
import type { ControlLayerDropData, ImageDraggableData } from 'features/dnd/types';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiFloppyDiskBold, PiRulerBold } from 'react-icons/pi';
import {
  useAddImageToBoardMutation,
  useChangeImageIsIntermediateMutation,
  useGetImageDTOQuery,
  useRemoveImageFromBoardMutation,
} from 'services/api/endpoints/images';
import type { ControlLayerAction, ImageDTO } from 'services/api/types';

type Props = {
  image: ImageWithDims | null;
  processedImage: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  hasProcessor: boolean;
  layerId: string; // required for the dnd/upload interactions
};

const selectPendingControlImages = createMemoizedSelector(
  selectControlAdaptersSlice,
  (controlAdapters) => controlAdapters.pendingControlImages
);

export const CALayerImagePreview = memo(({ image, processedImage, onChangeImage, hasProcessor, layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoAddBoardId = useAppSelector((s) => s.gallery.autoAddBoardId);
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const pendingControlImages = useAppSelector(selectPendingControlImages);
  const shift = useShiftModifier();

  const [isMouseOverImage, setIsMouseOverImage] = useState(false);

  const { currentData: controlImage, isError: isErrorControlImage } = useGetImageDTOQuery(
    image?.imageName ?? skipToken
  );
  const { currentData: processedControlImage, isError: isErrorProcessedControlImage } = useGetImageDTOQuery(
    processedImage?.imageName ?? skipToken
  );

  const [changeIsIntermediate] = useChangeImageIsIntermediateMutation();
  const [addToBoard] = useAddImageToBoardMutation();
  const [removeFromBoard] = useRemoveImageFromBoardMutation();
  const handleResetControlImage = useCallback(() => {
    onChangeImage(null);
  }, [onChangeImage]);

  const handleSaveControlImage = useCallback(async () => {
    if (!processedControlImage) {
      return;
    }

    await changeIsIntermediate({
      imageDTO: processedControlImage,
      is_intermediate: false,
    }).unwrap();

    if (autoAddBoardId !== 'none') {
      addToBoard({
        imageDTO: processedControlImage,
        board_id: autoAddBoardId,
      });
    } else {
      removeFromBoard({ imageDTO: processedControlImage });
    }
  }, [processedControlImage, changeIsIntermediate, autoAddBoardId, addToBoard, removeFromBoard]);

  const handleSetControlImageToDimensions = useCallback(() => {
    if (!controlImage) {
      return;
    }

    if (activeTabName === 'unifiedCanvas') {
      dispatch(setBoundingBoxDimensions({ width: controlImage.width, height: controlImage.height }, optimalDimension));
    } else {
      if (shift) {
        const { width, height } = controlImage;
        dispatch(widthChanged({ width, updateAspectRatio: true }));
        dispatch(heightChanged({ height, updateAspectRatio: true }));
      } else {
        const { width, height } = calculateNewSize(
          controlImage.width / controlImage.height,
          optimalDimension * optimalDimension
        );
        dispatch(widthChanged({ width, updateAspectRatio: true }));
        dispatch(heightChanged({ height, updateAspectRatio: true }));
      }
    }
  }, [controlImage, activeTabName, dispatch, optimalDimension, shift]);

  const handleMouseEnter = useCallback(() => {
    setIsMouseOverImage(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsMouseOverImage(false);
  }, []);

  const draggableData = useMemo<ImageDraggableData | undefined>(() => {
    if (controlImage) {
      return {
        id: layerId,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO: controlImage },
      };
    }
  }, [controlImage, layerId]);

  const droppableData = useMemo<ControlLayerDropData>(
    () => ({
      id: layerId,
      actionType: 'SET_CONTROL_LAYER_IMAGE',
      context: { layerId },
    }),
    [layerId]
  );

  const postUploadAction = useMemo<ControlLayerAction>(() => ({ type: 'SET_CONTROL_LAYER_IMAGE', layerId }), [layerId]);

  const shouldShowProcessedImage =
    controlImage &&
    processedControlImage &&
    !isMouseOverImage &&
    !pendingControlImages.includes(layerId) &&
    hasProcessor;

  useEffect(() => {
    if (isConnected && (isErrorControlImage || isErrorProcessedControlImage)) {
      handleResetControlImage();
    }
  }, [handleResetControlImage, isConnected, isErrorControlImage, isErrorProcessedControlImage]);

  return (
    <Flex
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      position="relative"
      w="full"
      h={36}
      alignItems="center"
      justifyContent="center"
    >
      <IAIDndImage
        draggableData={draggableData}
        droppableData={droppableData}
        imageDTO={controlImage}
        isDropDisabled={shouldShowProcessedImage}
        postUploadAction={postUploadAction}
      />

      <Box
        position="absolute"
        top={0}
        insetInlineStart={0}
        w="full"
        h="full"
        opacity={shouldShowProcessedImage ? 1 : 0}
        transitionProperty="common"
        transitionDuration="normal"
        pointerEvents="none"
      >
        <IAIDndImage
          draggableData={draggableData}
          droppableData={droppableData}
          imageDTO={processedControlImage}
          isUploadDisabled={true}
        />
      </Box>

      <>
        <IAIDndImageIcon
          onClick={handleResetControlImage}
          icon={controlImage ? <PiArrowCounterClockwiseBold size={16} /> : undefined}
          tooltip={t('controlnet.resetControlImage')}
        />
        <IAIDndImageIcon
          onClick={handleSaveControlImage}
          icon={controlImage ? <PiFloppyDiskBold size={16} /> : undefined}
          tooltip={t('controlnet.saveControlImage')}
          styleOverrides={saveControlImageStyleOverrides}
        />
        <IAIDndImageIcon
          onClick={handleSetControlImageToDimensions}
          icon={controlImage ? <PiRulerBold size={16} /> : undefined}
          tooltip={shift ? t('controlnet.setControlImageDimensionsForce') : t('controlnet.setControlImageDimensions')}
          styleOverrides={setControlImageDimensionsStyleOverrides}
        />
      </>

      {pendingControlImages.includes(layerId) && (
        <Flex
          position="absolute"
          top={0}
          insetInlineStart={0}
          w="full"
          h="full"
          alignItems="center"
          justifyContent="center"
          opacity={0.8}
          borderRadius="base"
          bg="base.900"
        >
          <Spinner size="xl" color="base.400" />
        </Flex>
      )}
    </Flex>
  );
});

CALayerImagePreview.displayName = 'CALayerImagePreview';

const saveControlImageStyleOverrides: SystemStyleObject = { mt: 6 };
const setControlImageDimensionsStyleOverrides: SystemStyleObject = { mt: 12 };
