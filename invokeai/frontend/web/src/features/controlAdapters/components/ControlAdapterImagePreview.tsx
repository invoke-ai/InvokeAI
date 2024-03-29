import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Spinner } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { useControlAdapterControlImage } from 'features/controlAdapters/hooks/useControlAdapterControlImage';
import { useControlAdapterProcessedControlImage } from 'features/controlAdapters/hooks/useControlAdapterProcessedControlImage';
import { useControlAdapterProcessorType } from 'features/controlAdapters/hooks/useControlAdapterProcessorType';
import {
  controlAdapterImageChanged,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { heightChanged, selectOptimalDimension, widthChanged } from 'features/parameters/store/generationSlice';
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
import type { PostUploadAction } from 'services/api/types';

type Props = {
  id: string;
  isSmall?: boolean;
};

const selectPendingControlImages = createMemoizedSelector(
  selectControlAdaptersSlice,
  (controlAdapters) => controlAdapters.pendingControlImages
);

const ControlAdapterImagePreview = ({ isSmall, id }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const controlImageName = useControlAdapterControlImage(id);
  const processedControlImageName = useControlAdapterProcessedControlImage(id);
  const processorType = useControlAdapterProcessorType(id);
  const autoAddBoardId = useAppSelector((s) => s.gallery.autoAddBoardId);
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const pendingControlImages = useAppSelector(selectPendingControlImages);

  const [isMouseOverImage, setIsMouseOverImage] = useState(false);

  const { currentData: controlImage, isError: isErrorControlImage } = useGetImageDTOQuery(
    controlImageName ?? skipToken
  );

  const { currentData: processedControlImage, isError: isErrorProcessedControlImage } = useGetImageDTOQuery(
    processedControlImageName ?? skipToken
  );

  const [changeIsIntermediate] = useChangeImageIsIntermediateMutation();
  const [addToBoard] = useAddImageToBoardMutation();
  const [removeFromBoard] = useRemoveImageFromBoardMutation();
  const handleResetControlImage = useCallback(() => {
    dispatch(controlAdapterImageChanged({ id, controlImage: null }));
  }, [id, dispatch]);

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
      const { width, height } = calculateNewSize(
        controlImage.width / controlImage.height,
        optimalDimension * optimalDimension
      );
      dispatch(widthChanged(width));
      dispatch(heightChanged(height));
    }
  }, [controlImage, activeTabName, dispatch, optimalDimension]);

  const handleMouseEnter = useCallback(() => {
    setIsMouseOverImage(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsMouseOverImage(false);
  }, []);

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (controlImage) {
      return {
        id,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO: controlImage },
      };
    }
  }, [controlImage, id]);

  const droppableData = useMemo<TypesafeDroppableData | undefined>(
    () => ({
      id,
      actionType: 'SET_CONTROL_ADAPTER_IMAGE',
      context: { id },
    }),
    [id]
  );

  const postUploadAction = useMemo<PostUploadAction>(() => ({ type: 'SET_CONTROL_ADAPTER_IMAGE', id }), [id]);

  const shouldShowProcessedImage =
    controlImage &&
    processedControlImage &&
    !isMouseOverImage &&
    !pendingControlImages.includes(id) &&
    processorType !== 'none';

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
      h={isSmall ? 32 : 366} // magic no touch
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
          tooltip={t('controlnet.setControlImageDimensions')}
          styleOverrides={setControlImageDimensionsStyleOverrides}
        />
      </>

      {pendingControlImages.includes(id) && (
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
};

export default memo(ControlAdapterImagePreview);

const saveControlImageStyleOverrides: SystemStyleObject = { mt: 6 };
const setControlImageDimensionsStyleOverrides: SystemStyleObject = { mt: 12 };
