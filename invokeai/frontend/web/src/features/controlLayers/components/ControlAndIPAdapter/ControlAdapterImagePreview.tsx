import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Spinner, useShiftModifier } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { heightChanged, widthChanged } from 'features/controlLayers/store/controlLayersSlice';
import type { ControlNetConfig, T2IAdapterConfig } from 'features/controlLayers/util/controlAdapters';
import type { ImageDraggableData, TypesafeDroppableData } from 'features/dnd/types';
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
import type { ImageDTO, PostUploadAction } from 'services/api/types';

type Props = {
  controlAdapter: ControlNetConfig | T2IAdapterConfig;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  droppableData: TypesafeDroppableData;
  postUploadAction: PostUploadAction;
};

export const ControlAdapterImagePreview = memo(
  ({ controlAdapter, onChangeImage, droppableData, postUploadAction }: Props) => {
    const { t } = useTranslation();
    const dispatch = useAppDispatch();
    const autoAddBoardId = useAppSelector((s) => s.gallery.autoAddBoardId);
    const isConnected = useAppSelector((s) => s.system.isConnected);
    const activeTabName = useAppSelector(activeTabNameSelector);
    const optimalDimension = useAppSelector(selectOptimalDimension);
    const shift = useShiftModifier();

    const [isMouseOverImage, setIsMouseOverImage] = useState(false);

    const { currentData: controlImage, isError: isErrorControlImage } = useGetImageDTOQuery(
      controlAdapter.image?.imageName ?? skipToken
    );
    const { currentData: processedControlImage, isError: isErrorProcessedControlImage } = useGetImageDTOQuery(
      controlAdapter.processedImage?.imageName ?? skipToken
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
        dispatch(
          setBoundingBoxDimensions({ width: controlImage.width, height: controlImage.height }, optimalDimension)
        );
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
          id: controlAdapter.id,
          payloadType: 'IMAGE_DTO',
          payload: { imageDTO: controlImage },
        };
      }
    }, [controlImage, controlAdapter.id]);

    const shouldShowProcessedImage =
      controlImage &&
      processedControlImage &&
      !isMouseOverImage &&
      !controlAdapter.isProcessingImage &&
      controlAdapter.processorConfig !== null;

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

        {controlAdapter.isProcessingImage && (
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
  }
);

ControlAdapterImagePreview.displayName = 'ControlAdapterImagePreview';

const saveControlImageStyleOverrides: SystemStyleObject = { mt: 6 };
const setControlImageDimensionsStyleOverrides: SystemStyleObject = { mt: 12 };
